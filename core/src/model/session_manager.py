import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent      TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    message    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (agent, session_id, id);
"""


class SessionManager:
    """SQLite-backed chat history: one conversation per (agent, session_id), survives restarts.

    Each call opens a short-lived connection (WAL mode, busy timeout), so several agents —
    or processes — can share one database file.
    """

    def __init__(self, db_file: str | Path) -> None:
        self._db_path = Path(db_file)
        self._schema_ready = False

    def load(self, agent: str, session_id: str, limit: int) -> list[BaseMessage]:
        """The last *limit* messages of a conversation, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT message FROM (SELECT id, message FROM messages WHERE agent = ? AND session_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id",
                (agent, session_id, limit),
            ).fetchall()
        return messages_from_dict([json.loads(r[0]) for r in rows])

    def append(self, agent: str, session_id: str, messages: list[BaseMessage]) -> None:
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO messages (agent, session_id, created_at, message) VALUES (?, ?, ?, ?)",
                [(agent, session_id, now, json.dumps(d)) for d in messages_to_dict(messages)],
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            if not self._schema_ready:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                self._schema_ready = True
                logger.info("session history: using database %s", self._db_path)
            with conn:
                yield conn
        finally:
            conn.close()
