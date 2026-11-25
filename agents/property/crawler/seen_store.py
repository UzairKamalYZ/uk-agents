"""
SQLite record of every listing the crawler has indexed, so each hourly pass embeds
only what is new or changed.
"""
import sqlite3
from datetime import datetime, timezone
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
        self._conn.commit()

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
