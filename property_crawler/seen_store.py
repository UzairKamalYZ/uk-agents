"""
SQLite record of every listing the crawler has indexed, so each pass embeds
only what is new or changed.

The `listings` table also keeps each listing's details (title, price, city, deal,
availability, …) so other apps can list and filter them without the vector index —
my-stats' Properties page reads it read-only. Those columns are a contract: see
README.md → "The listings table".
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .extractor import Listing


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SeenStore:

    def __init__(self, db_file: str | Path):
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_file))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS listings (
                   id TEXT PRIMARY KEY,
                   site TEXT NOT NULL,
                   url TEXT NOT NULL,
                   content_hash TEXT NOT NULL,
                   first_seen TEXT NOT NULL,
                   last_seen TEXT NOT NULL
               )"""
        )
        # Sites that blocked the crawler: skipped until blocked_until; strikes doubles the wait
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS site_backoff (
                   site TEXT PRIMARY KEY,
                   strikes INTEGER NOT NULL,
                   blocked_until TEXT NOT NULL,
                   reason TEXT NOT NULL
               )"""
        )
        # Bookkeeping, e.g. when the last pass started (so a missed scheduled pass is caught up)
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._migrate()
        self._conn.commit()

    # Listing details, added after the first version of the table: (column, SQL type)
    DETAIL_COLUMNS = [
        ("title", "TEXT"), ("description", "TEXT"), ("city", "TEXT"), ("postcode", "TEXT"),
        ("price", "REAL"), ("price_text", "TEXT"), ("currency", "TEXT"), ("rooms", "REAL"),
        ("area", "REAL"), ("property_type", "TEXT"), ("deal", "TEXT"), ("availability", "TEXT"),
        ("photo", "TEXT"), ("extra", "TEXT"),          # extra: JSON object of the site's other fields
        ("updated_at", "TEXT"),                         # when the details last changed
        ("fingerprint", "TEXT"),                        # the property, whichever site lists it
        ("duplicate_of", "TEXT"),                       # id of the same property on another site
    ]

    def _migrate(self) -> None:
        """Add the detail columns to a seen.db made before they existed (rows keep NULL until re-seen)."""
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(listings)")}
        for name, sql_type in self.DETAIL_COLUMNS:
            if name not in have:
                self._conn.execute(f"ALTER TABLE listings ADD COLUMN {name} {sql_type}")
        self._conn.execute("CREATE INDEX IF NOT EXISTS listings_first_seen ON listings (first_seen)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS listings_fingerprint ON listings (fingerprint)")

    # --- passes --------------------------------------------------------------

    def last_pass(self) -> datetime | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'last_pass'").fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def record_pass(self, started: datetime) -> None:
        self._conn.execute("INSERT INTO meta (key, value) VALUES ('last_pass', ?) "
                           "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                           (started.isoformat(timespec="seconds"),))
        self._conn.commit()

    # --- back-off ------------------------------------------------------------

    def blocked_until(self, site: str, now: datetime | None = None) -> datetime | None:
        """When the site may be tried again, or None if it may be tried now."""
        row = self._conn.execute("SELECT blocked_until FROM site_backoff WHERE site = ?", (site,)).fetchone()
        if row is None:
            return None
        until = datetime.fromisoformat(row[0])
        return until if until > (now or datetime.now(timezone.utc)) else None

    def record_block(self, site: str, reason: str, base: timedelta, cap: timedelta,
                     retry_after: timedelta | None = None, now: datetime | None = None) -> datetime:
        """Back off: base, 2×base, 4×base … up to cap (or longer, if the site's Retry-After asks)."""
        row = self._conn.execute("SELECT strikes FROM site_backoff WHERE site = ?", (site,)).fetchone()
        strikes = (row[0] if row else 0) + 1
        wait = min(base * 2 ** (strikes - 1), cap)
        if retry_after is not None:
            wait = max(wait, retry_after)
        until = (now or datetime.now(timezone.utc)) + wait
        self._conn.execute(
            """INSERT INTO site_backoff (site, strikes, blocked_until, reason) VALUES (?, ?, ?, ?)
               ON CONFLICT(site) DO UPDATE SET strikes = excluded.strikes,
                   blocked_until = excluded.blocked_until, reason = excluded.reason""",
            (site, strikes, until.isoformat(timespec="seconds"), reason),
        )
        self._conn.commit()
        return until

    def clear_block(self, site: str) -> None:
        """The site answered normally again: forget its strikes."""
        self._conn.execute("DELETE FROM site_backoff WHERE site = ?", (site,))
        self._conn.commit()

    # --- listings ------------------------------------------------------------

    def unseen(self, listings: list[Listing]) -> list[Listing]:
        """Listings never indexed, or whose content changed since they were."""
        fresh = []
        for listing in listings:
            row = self._conn.execute(
                "SELECT content_hash FROM listings WHERE id = ?", (listing.id,)
            ).fetchone()
            if row is None or row[0] != listing.content_hash():
                fresh.append(listing)
        return fresh

    def original_of(self, listing: Listing) -> str | None:
        """The id of the same property already stored from another site, or None if this one is the first."""
        fingerprint = listing.fingerprint
        if fingerprint is None:
            return None
        row = self._conn.execute(
            "SELECT id FROM listings WHERE fingerprint = ? AND site != ? AND duplicate_of IS NULL AND id != ? "
            "ORDER BY first_seen LIMIT 1",
            (fingerprint, listing.site, listing.id),
        ).fetchone()
        return row[0] if row else None

    def touch(self, listings: list[Listing], when: str | None = None,
              duplicate_of: dict[str, str] | None = None) -> None:
        """Store new or changed listings with their details, seen now (call after they are safely
        in the vector store). first_seen is kept from the first time a listing was stored.
        duplicate_of: listing id → id of the same property on another site (that one is indexed)."""
        when = when or utc_now()
        duplicate_of = duplicate_of or {}
        rows = []
        for l in listings:
            rows.append((
                l.id, l.site, l.url, l.content_hash(), when, when,
                l.title or None, l.description or None, l.city, l.postcode, l.price, l.price_text or None,
                l.currency, l.rooms, l.area, l.property_type, l.deal, l.availability, l.photo,
                json.dumps(l.extra, ensure_ascii=False) if l.extra else None, when,
                l.fingerprint, duplicate_of.get(l.id),
            ))
        self._conn.executemany(
            """INSERT INTO listings (id, site, url, content_hash, first_seen, last_seen,
                   title, description, city, postcode, price, price_text, currency, rooms, area,
                   property_type, deal, availability, photo, extra, updated_at, fingerprint, duplicate_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   content_hash = excluded.content_hash, last_seen = excluded.last_seen,
                   url = excluded.url, title = excluded.title, description = excluded.description,
                   city = excluded.city, postcode = excluded.postcode, price = excluded.price,
                   price_text = excluded.price_text, currency = excluded.currency, rooms = excluded.rooms,
                   area = excluded.area, property_type = excluded.property_type, deal = excluded.deal,
                   availability = excluded.availability, photo = excluded.photo, extra = excluded.extra,
                   updated_at = excluded.updated_at, fingerprint = excluded.fingerprint,
                   duplicate_of = excluded.duplicate_of""",
            rows,
        )
        self._conn.commit()

    def seen_again(self, listings: list[Listing], when: str | None = None) -> None:
        """Unchanged listings still on the site: only last_seen moves."""
        if not listings:
            return
        when = when or utc_now()
        self._conn.executemany("UPDATE listings SET last_seen = ? WHERE id = ?",
                               [(when, l.id) for l in listings])
        self._conn.commit()

    def count(self, site: str | None = None) -> int:
        if site:
            return self._conn.execute("SELECT COUNT(*) FROM listings WHERE site = ?", (site,)).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
