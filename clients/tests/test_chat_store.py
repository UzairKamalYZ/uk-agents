import json
from unittest.mock import patch

from clients.streamlit import main
from clients.streamlit.chat_store import ChatStore, SavedChat

MESSAGES = [{"role": "user", "content": "Aldi 10"}, {"role": "assistant", "content": "✅ Saved"}]


class FakeSessionState(dict):
    """st.session_state supports both item and attribute access."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_load_returns_empty_chat_when_no_file(tmp_path):
    assert ChatStore(tmp_path / "chat.json").load() == SavedChat()


def test_save_then_load_round_trips(tmp_path):
    store = ChatStore(tmp_path / "nested" / "chat.json")
    store.save("abc", MESSAGES)
    assert store.load() == SavedChat("abc", MESSAGES)
    assert not list((tmp_path / "nested").glob("*.tmp"))


def test_load_ignores_corrupt_file(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text("{not json", encoding="utf-8")
    assert ChatStore(path).load() == SavedChat()


def test_load_drops_malformed_messages(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({"session_id": 5, "messages": [
        MESSAGES[0], {"role": "system", "content": "x"}, {"role": "user"}, "junk",
    ]}), encoding="utf-8")
    assert ChatStore(path).load() == SavedChat(None, [MESSAGES[0]])


def test_clear_removes_saved_chat(tmp_path):
    store = ChatStore(tmp_path / "chat.json")
    store.save("abc", MESSAGES)
    store.clear()
    store.clear()  # clearing twice is fine
    assert store.load() == SavedChat()


def test_new_browser_session_resumes_saved_chat(tmp_path):
    store = ChatStore(tmp_path / "chat.json")
    store.save("abc", MESSAGES)
    state = FakeSessionState()
    with patch.object(main, "_STORE", store), patch.object(main.st, "session_state", state):
        main._resume_chat()
    assert state["messages"] == MESSAGES
    assert state["session_id"] == "abc"


def test_resume_does_not_overwrite_a_running_session(tmp_path):
    store = ChatStore(tmp_path / "chat.json")
    store.save("abc", MESSAGES)
    state = FakeSessionState(messages=[])
    with patch.object(main, "_STORE", store), patch.object(main.st, "session_state", state):
        main._resume_chat()
    assert state["messages"] == [] and "session_id" not in state


def test_new_chat_clears_session_and_saved_chat(tmp_path):
    store = ChatStore(tmp_path / "chat.json")
    store.save("abc", MESSAGES)
    state = FakeSessionState(messages=MESSAGES, session_id="abc")
    with patch.object(main, "_STORE", store), patch.object(main.st, "session_state", state):
        main._new_chat()
    assert state["messages"] == [] and "session_id" not in state
    assert store.load() == SavedChat()
