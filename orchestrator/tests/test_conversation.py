from orchestrator.conversation import HISTORY_TURNS, ConversationMemory
from core.src.model.session_manager import SessionManager


def test_recent_renders_turns_with_agent_labels(tmp_path):
    memory = ConversationMemory(SessionManager(tmp_path / "s.db"))
    assert memory.recent("chat") == ""
    memory.record("chat", "Aldi 10", "Saved.\nWant the breakdown?", ["expenditure_tracker"])
    assert memory.recent("chat") == "user: Aldi 10\nexpenditure_tracker: Saved. Want the breakdown?"


def test_recent_keeps_only_the_last_turns_and_truncates(tmp_path):
    memory = ConversationMemory(SessionManager(tmp_path / "s.db"))
    for i in range(HISTORY_TURNS + 2):
        memory.record("chat", f"q{i}", "x" * 1000 if i == HISTORY_TURNS + 1 else f"a{i}", ["a"])
    lines = memory.recent("chat").splitlines()
    assert len(lines) == HISTORY_TURNS * 2
    assert lines[0] == "user: q2"
    assert lines[-1].endswith("…") and len(lines[-1]) < 700


def test_history_survives_restart_and_empty_answers_are_skipped(tmp_path):
    db = tmp_path / "s.db"
    ConversationMemory(SessionManager(db)).record("chat", "hi", "", ["a"])
    ConversationMemory(SessionManager(db)).record("chat", "hi", "Hello!", ["conversational_agent", "finance_agent"])
    assert ConversationMemory(SessionManager(db)).recent("chat") == (
        "user: hi\nconversational_agent+finance_agent: Hello!"
    )


def test_store_failures_never_raise():
    class Broken:
        def get(self, *a):
            raise OSError("disk")

        def append(self, *a):
            raise OSError("disk")

    memory = ConversationMemory(Broken())
    assert memory.recent("chat") == ""
    memory.record("chat", "hi", "Hello!", ["a"])


def test_recent_hides_tool_call_markup_left_in_older_turns(tmp_path):
    store = SessionManager(tmp_path / "s.db")
    from langchain_core.messages import AIMessage, HumanMessage
    store.append("orchestrator", "chat", [
        HumanMessage(content="is my return filed?"),
        AIMessage(content='Checking.<invoke name="Skill"><parameter name="a">x</parameter></invoke>', name="fbr"),
    ])
    assert ConversationMemory(store).recent("chat") == "user: is my return filed?\nfbr: Checking."
