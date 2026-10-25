import threading

from langchain_core.messages import AIMessage, HumanMessage

from core.src.model.session_manager import InMemoryHistoryStore, SessionManager


def _turn(i):
    return [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]


def test_sqlite_history_survives_reopen_and_keeps_order(tmp_path):
    db = tmp_path / "s.db"
    store = SessionManager(db)
    store.append("AgentA", "s1", _turn(1))
    store.append("AgentA", "s1", _turn(2))
    store.close()
    history = SessionManager(db).get("AgentA", "s1")
    assert [m.content for m in history] == ["q1", "a1", "q2", "a2"]
    assert [type(m) for m in history] == [HumanMessage, AIMessage] * 2


def test_sqlite_history_limit_returns_most_recent(tmp_path):
    store = SessionManager(tmp_path / "s.db")
    for i in range(5):
        store.append("A", "s", _turn(i))
    assert [m.content for m in store.get("A", "s", limit=4)] == ["q3", "a3", "q4", "a4"]


def test_sqlite_history_isolated_by_agent_and_session(tmp_path):
    store = SessionManager(tmp_path / "s.db")
    store.append("A", "s1", _turn(1))
    assert store.get("B", "s1") == []
    assert store.get("A", "s2") == []


def test_sqlite_store_does_not_touch_disk_until_used(tmp_path):
    SessionManager(tmp_path / "sub" / "s.db")
    assert not (tmp_path / "sub").exists()


def test_concurrent_appends_lose_nothing(tmp_path):
    for store in (SessionManager(tmp_path / "s.db"), InMemoryHistoryStore()):
        threads = [threading.Thread(target=lambda i=i: [store.append("A", "s", _turn(i)) for _ in range(20)])
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(store.get("A", "s")) == 8 * 20 * 2


def test_sqlite_history_migrates_legacy_json_message_table(tmp_path):
    import json
    import sqlite3

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT NOT NULL,"
        " session_id TEXT NOT NULL, created_at TEXT NOT NULL, message TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX messages_by_conversation ON messages (agent, session_id, id)")
    for kind, text in [("human", "q1"), ("ai", "a1")]:
        conn.execute(
            "INSERT INTO messages (agent, session_id, created_at, message) VALUES (?, ?, ?, ?)",
            ("A", "s", "2026-10-23 22:24:47", json.dumps({"type": kind, "data": {"content": text}})),
        )
    conn.commit()
    conn.close()

    store = SessionManager(db)
    store.append("A", "s", _turn(2))
    history = store.get("A", "s")
    assert [m.content for m in history] == ["q1", "a1", "q2", "a2"]
    assert [type(m) for m in history] == [HumanMessage, AIMessage] * 2


def test_sqlite_history_keeps_message_names_across_reopen(tmp_path):
    db = tmp_path / "s.db"
    SessionManager(db).append("A", "s", [HumanMessage(content="q"), AIMessage(content="a", name="expenditure_tracker")])
    history = SessionManager(db).get("A", "s")
    assert [m.name for m in history] == [None, "expenditure_tracker"]


def test_sqlite_history_adds_name_column_to_an_existing_table(tmp_path):
    import sqlite3

    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT NOT NULL, session_id TEXT NOT NULL,"
        " role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO messages (agent, session_id, role, content) VALUES ('A', 's', 'human', 'old')")
    conn.commit()
    conn.close()
    store = SessionManager(db)
    store.append("A", "s", [AIMessage(content="new", name="x")])
    assert [(m.content, m.name) for m in store.get("A", "s")] == [("old", None), ("new", "x")]
