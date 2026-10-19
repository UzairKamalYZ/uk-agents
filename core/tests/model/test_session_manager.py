from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from core.src.model.llm_model_graph import LlmModelGraph
from core.src.model.session_manager import SessionManager


def test_history_survives_a_new_manager(tmp_path):
    db = tmp_path / "sessions.db"
    SessionManager(db).append("PropertyAgent", "chat-1", [HumanMessage(content="hi"), AIMessage(content="hello")])
    restored = SessionManager(db).load("PropertyAgent", "chat-1", limit=10)
    assert [(type(m), m.content) for m in restored] == [(HumanMessage, "hi"), (AIMessage, "hello")]


def test_load_returns_the_latest_window_oldest_first(tmp_path):
    store = SessionManager(tmp_path / "s.db")
    store.append("A", "s", [HumanMessage(content=str(i)) for i in range(5)])
    assert [m.content for m in store.load("A", "s", limit=2)] == ["3", "4"]


def test_conversations_are_separate_per_agent_and_session(tmp_path):
    store = SessionManager(tmp_path / "s.db")
    store.append("A", "s1", [HumanMessage(content="a1")])
    store.append("B", "s1", [HumanMessage(content="b1")])
    store.append("A", "s2", [HumanMessage(content="a2")])
    assert [m.content for m in store.load("A", "s1", 10)] == ["a1"]
    assert [m.content for m in store.load("B", "s1", 10)] == ["b1"]
    assert store.load("C", "s1", 10) == []


def _graph(store):
    with patch("core.src.model.llm_model_graph.Config") as cfg:
        cfg.MAX_HISTORY_TURNS = 10
        g = LlmModelGraph(MagicMock(), system_prompt="sys", rag_context_manager=MagicMock(),
                          tools=[], agent_name="TestAgent", history_store=store)
    g.graph = MagicMock()
    return g


def test_llm_model_graph_reads_and_writes_the_store(tmp_path):
    db = tmp_path / "sessions.db"
    with patch("core.src.model.llm_model_graph.Config") as cfg:
        cfg.MAX_HISTORY_TURNS = 10
        first = _graph(SessionManager(db))
        first.graph.invoke.return_value = {"answer": "a1"}
        first.ask("q1", session_id="chat")
        restarted = _graph(SessionManager(db))  # e.g. after a bot restart
        restarted.graph.invoke.return_value = {"answer": "a2"}
        restarted.ask("q2", session_id="chat")
    history = restarted.graph.invoke.call_args[0][0]["session_history"]
    assert [m.content for m in history] == ["q1", "a1"]
    assert [m.content for m in SessionManager(db).load("TestAgent", "chat", 10)] == ["q1", "a1", "q2", "a2"]


def test_broken_store_does_not_stop_the_answer():
    store = MagicMock()
    store.load.side_effect = OSError("disk")
    store.append.side_effect = OSError("disk")
    with patch("core.src.model.llm_model_graph.Config") as cfg:
        cfg.MAX_HISTORY_TURNS = 10
        g = _graph(store)
        g.graph.invoke.return_value = {"answer": "ok"}
        assert g.ask("q", session_id="s") == "ok"
