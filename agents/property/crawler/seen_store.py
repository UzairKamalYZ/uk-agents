"""
SQLite record of every listing the crawler has indexed, so each hourly pass embeds
only what is new or changed.
"""
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

    def touch(self, listings: list[Listing], when: str | None = None) -> None:
        """Mark listings as seen now (call after they are safely in the vector store)."""
        when = when or utc_now()
        self._conn.executemany(
            """INSERT INTO listings (id, site, url, content_hash, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET content_hash = excluded.content_hash,
                                             last_seen = excluded.last_seen""",
            [(l.id, l.site, l.url, l.content_hash(), when, when) for l in listings],
        )
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
