import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_ROLES: dict[str, type[BaseMessage]] = {"human": HumanMessage, "ai": AIMessage}


class HistoryStore(ABC):
    """Per-agent, per-session chat history. Implementations must be thread-safe."""

    @abstractmethod
    def get(self, agent: str, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        """The session's messages, oldest first; only the last `limit` when given."""

    @abstractmethod
    def append(self, agent: str, session_id: str, messages: list[BaseMessage]) -> None: ...

    def close(self) -> None:
        return None


class InMemoryHistoryStore(HistoryStore):
    """Process-local history — lost on restart. Used when no store is configured (e.g. tests)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.histories: dict[tuple[str, str], list[BaseMessage]] = {}

    def get(self, agent: str, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        with self._lock:
            history = list(self.histories.get((agent, session_id), []))
        return history[-limit:] if limit else history

    def append(self, agent: str, session_id: str, messages: list[BaseMessage]) -> None:
        with self._lock:
            self.histories.setdefault((agent, session_id), []).extend(messages)


class SessionManager(HistoryStore):
    """SQLite-backed history that survives restarts.

    One connection shared across threads behind a lock; the file is opened on first use,
    so constructing the store never touches the disk.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def get(self, agent: str, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT role, content FROM (SELECT id, role, content FROM messages"
                " WHERE agent = ? AND session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (agent, session_id, limit if limit else -1),
            ).fetchall()
        return [_ROLES[role](content=content) for role, content in rows if role in _ROLES]

    def append(self, agent: str, session_id: str, messages: list[BaseMessage]) -> None:
        rows = [(agent, session_id, m.type, m.text) for m in messages if m.type in _ROLES]
        with self._lock:
            conn = self._connection()
            with conn:
                conn.executemany(
                    "INSERT INTO messages (agent, session_id, role, content) VALUES (?, ?, ?, ?)", rows
                )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT NOT NULL, session_id TEXT NOT NULL,"
                " role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS messages_session ON messages (agent, session_id, id)")
            self._conn = conn
        return self._conn


_shared: SessionManager | None = None
_shared_lock = threading.Lock()


def shared_session_store(db_path: str | Path) -> SessionManager:
    """The process-wide store every agent writes to (one connection, one lock)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SessionManager(db_path)
        return _shared
