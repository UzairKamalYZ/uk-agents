"""End-to-end graph tests: real LangGraph wiring, scripted fake LLM, mock agents."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from orchestrator.agent_registry import AgentEntry

SUMMARY = "2026-10-18 | Colruyt | 90.00"
PROMPTS = {
    "normaliser": "NORMALISER today={today} pending={pending} history={history}",
    "supervisor": "SUPERVISOR {agents} history={history} {user_input} {agent_outputs}",
    "synthesiser": "SYNTHESISER",
}


class ScriptedLLM:
    """Answers by inspecting the system prompt to tell which node is calling."""

    def __init__(self, normaliser_reply, supervisor_replies):
        self.normaliser_reply = normaliser_reply
        self.supervisor_replies = list(supervisor_replies)
        self.supervisor_calls = 0

    def invoke(self, messages):
        system = messages[0].content
        reply = MagicMock()
        if system.startswith("NORMALISER"):
            reply.text = self.normaliser_reply
        elif system.startswith("SUPERVISOR"):
            self.supervisor_calls += 1
            reply.text = self.supervisor_replies.pop(0) if self.supervisor_replies else '{"next": "FINISH"}'
        else:
            raise AssertionError("synthesiser LLM should not be called (single-output passthrough)")
        return reply


def _load_prompt(path):
    for key, text in PROMPTS.items():
        if key in str(path):
            return text
    raise AssertionError(f"unexpected prompt path {path}")


def _agents(pending=None):
    conv = MagicMock()
    conv.ask.return_value = "Hello!"
    conv.pending_summary.return_value = None
    exp = MagicMock()
    exp.ask.return_value = "Pending: 2026-10-18 | Colruyt | 80.00 — confirm?"
    exp.pending_summary.return_value = pending
    exp.confirm_pending.return_value = f"Saved {SUMMARY}\nLast month: €0.00\nThis month: €90.00"
    return [
        AgentEntry("conversational_agent", "chat", conv, "anthropic", "m"),
        AgentEntry("expenditure_tracker", "spend", exp, "anthropic", "m", owner_only=True),
    ]


# Patched over orchestrator.graph.Config: clients/tests/conftest.py replaces
# core.src.config.config with a MagicMock for the whole session, so the real
# Config cannot be relied on when the full suite runs.
_CONFIG = SimpleNamespace(
    LLM_PROVIDER="anthropic",
    LLM_MODEL_NAME="test-model",
    NORMALISER_PROMPT_FILE="normaliser.txt",
    SUPERVISOR_PROMPT_FILE="supervisor.txt",
    SYNTHESISER_PROMPT_FILE="synthesiser.txt",
)


class MemoryStore:
    """In-memory stand-in for SessionManager (same load/append interface)."""

    def __init__(self):
        self.rows: dict[tuple[str, str], list] = {}

    def load(self, agent, session_id, limit):
        return self.rows.get((agent, session_id), [])[-limit:]

    def append(self, agent, session_id, messages):
        self.rows.setdefault((agent, session_id), []).extend(messages)


def _build(llm, agents, store=None):
    with patch("orchestrator.graph.Config", _CONFIG), \
         patch("orchestrator.graph.create_llm", return_value=llm), \
         patch("orchestrator.graph.load_agents", return_value=agents), \
         patch("orchestrator.graph.load_prompt", side_effect=_load_prompt):
        from orchestrator.graph import MultiAgentGraph
        return MultiAgentGraph(history_store=store or MemoryStore())


def test_yes_without_pending_goes_to_supervisor():
    agents = _agents(pending=None)
    llm = ScriptedLLM(json.dumps({"text": "yes", "pending": None}),
                      ['{"next": "conversational_agent"}', '{"next": "FINISH"}'])
    answer = _build(llm, agents).ask("yes", "s1", is_owner=True)
    assert answer == "Hello!"
    agents[0].instance.ask.assert_called_once_with("yes", session_id="s1")
    agents[1].instance.ask.assert_not_called()
    agents[1].instance.confirm_pending.assert_not_called()
    assert llm.supervisor_calls >= 1


def test_confirm_skips_supervisor():
    agents = _agents(pending=SUMMARY)
    llm = ScriptedLLM(json.dumps({"text": "yes", "pending": "confirm"}), [])
    answer = _build(llm, agents).ask("yes", "s1", is_owner=True)
    assert answer == agents[1].instance.confirm_pending.return_value
    agents[1].instance.confirm_pending.assert_called_once_with("s1")
    agents[1].instance.ask.assert_not_called()
    assert llm.supervisor_calls == 0


def test_correction_runs_agent_once_then_synthesiser():
    agents = _agents(pending=SUMMARY)
    # If the supervisor's decision were honoured after the agent ran, it would
    # route to conversational_agent — the direct flag must override it.
    llm = ScriptedLLM(json.dumps({"text": "it was €80", "pending": "correction"}),
                      ['{"next": "conversational_agent"}'])
    answer = _build(llm, agents).ask("it was eighty", "s1", is_owner=True)
    exp = agents[1].instance
    exp.ask.assert_called_once_with(
        f"it was €80\n\n(Correction to the pending expense: {SUMMARY})", session_id="s1"
    )
    agents[0].instance.ask.assert_not_called()
    assert answer == exp.ask.return_value


def test_garbage_normaliser_reply_with_pending_reasks_without_supervisor():
    agents = _agents(pending=SUMMARY)
    llm = ScriptedLLM("not json at all", ['{"next": "conversational_agent"}'])
    answer = _build(llm, agents).ask("yes", "s1", is_owner=True)
    assert answer.startswith("Sorry, I didn't catch that. Pending: " + SUMMARY)
    assert llm.supervisor_calls == 0
    exp = agents[1].instance
    exp.discard_pending.assert_not_called()
    exp.confirm_pending.assert_not_called()
    exp.ask.assert_not_called()


def test_on_route_announces_agent_before_it_runs():
    agents = _agents(pending=None)
    events = []
    agents[1].instance.ask.side_effect = lambda *a, **k: events.append("agent ran") or "Saved"
    llm = ScriptedLLM(json.dumps({"text": "Aldi €10", "pending": None}),
                      ['{"next": "expenditure_tracker", "reason": "an expense"}', '{"next": "FINISH"}'])
    answer = _build(llm, agents).ask(
        "Aldi 10", "s1", is_owner=True, on_route=lambda entry: events.append(f"route {entry.name}")
    )
    assert answer == "Saved"
    assert events == ["route expenditure_tracker", "agent ran"]


def test_on_route_fires_for_pending_confirm():
    agents = _agents(pending=SUMMARY)
    routed = []
    llm = ScriptedLLM(json.dumps({"text": "yes", "pending": "confirm"}), [])
    _build(llm, agents).ask("yes", "s1", is_owner=True, on_route=lambda e: routed.append(e.name))
    assert routed == ["expenditure_tracker"]


def test_on_route_failure_does_not_break_the_request():
    agents = _agents(pending=None)
    llm = ScriptedLLM(json.dumps({"text": "hi", "pending": None}), ['{"next": "conversational_agent"}'])

    def boom(_entry):
        raise RuntimeError("telegram down")

    assert _build(llm, agents).ask("hi", "s1", on_route=boom) == "Hello!"


def test_routing_is_logged_with_request_id_and_reason(caplog):
    agents = _agents(pending=None)
    llm = ScriptedLLM(json.dumps({"text": "Aldi €10", "pending": None}),
                      ['{"next": "expenditure_tracker", "reason": "an expense"}'])
    with caplog.at_level("INFO", logger="orchestrator.tracing"), \
         patch("orchestrator.graph.new_request_id", return_value="req123"):
        _build(llm, agents).ask("Aldi 10", "s1", is_owner=True)
    assert "[route] request=req123 supervisor → expenditure_tracker (an expense)" in caplog.text


class RecordingLLM(ScriptedLLM):
    """ScriptedLLM that also keeps every system prompt it was sent."""

    def __init__(self, *args):
        super().__init__(*args)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return super().invoke(messages)


def test_follow_up_sees_previous_turn_and_reaches_the_agent_that_asked():
    agents = _agents(pending=None)
    exp = agents[1].instance
    store = MemoryStore()
    exp.ask.return_value = "Saved. Want the per-shop breakdown for October?"
    first = RecordingLLM(json.dumps({"text": "Aldi €10", "pending": None}),
                         ['{"next": "expenditure_tracker"}', '{"next": "FINISH"}'])
    _build(first, agents, store).ask("Aldi 10", "chat-1", is_owner=True)
    assert "history=None" in first.prompts[0]  # first turn: nothing to go on yet

    # Second turn: the normaliser rewrites "yes" using the history; the supervisor sees it too
    exp.ask.return_value = "Breakdown: Aldi €10"
    second = RecordingLLM(json.dumps({"text": "Yes, show me the per-shop breakdown for October.", "pending": None}),
                          ['{"next": "expenditure_tracker"}', '{"next": "FINISH"}'])
    answer = _build(second, agents, store).ask("yes", "chat-1", is_owner=True)
    assert answer == "Breakdown: Aldi €10"
    exp.ask.assert_called_with("Yes, show me the per-shop breakdown for October.", session_id="chat-1")
    expected_history = ("user: Aldi 10\n"
                        "expenditure_tracker: Saved. Want the per-shop breakdown for October?")
    assert expected_history in second.prompts[0]                      # normaliser
    assert any(expected_history in p for p in second.prompts if p.startswith("SUPERVISOR"))


def test_history_is_kept_per_session():
    agents = _agents(pending=None)
    store = MemoryStore()
    llm = RecordingLLM(json.dumps({"text": "hi", "pending": None}), ['{"next": "conversational_agent"}'])
    _build(llm, agents, store).ask("hi", "chat-A")
    other = RecordingLLM(json.dumps({"text": "hi", "pending": None}), ['{"next": "conversational_agent"}'])
    _build(other, agents, store).ask("hi", "chat-B")
    assert "history=None" in other.prompts[0]
    assert [m.content for m in store.rows[("orchestrator", "chat-A")]] == ["hi", "Hello!"]
