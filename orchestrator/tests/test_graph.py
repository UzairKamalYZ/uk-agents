"""End-to-end graph tests: real LangGraph wiring, scripted fake LLM, mock agents."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from orchestrator.agent_registry import AgentEntry

SUMMARY = "2026-10-18 | Colruyt | 90.00"
PROMPTS = {
    "normaliser": "NORMALISER today={today} pending={pending}",
    "supervisor": "SUPERVISOR {agents} {user_input} {agent_outputs}",
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
    exp.confirm_pending.return_value = f"Saved {SUMMARY}\nprev_month: €0, current_month: €90"
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


def _build(llm, agents):
    with patch("orchestrator.graph.Config", _CONFIG), \
         patch("orchestrator.graph.create_llm", return_value=llm), \
         patch("orchestrator.graph.load_agents", return_value=agents), \
         patch("orchestrator.graph.load_prompt", side_effect=_load_prompt):
        from orchestrator.graph import MultiAgentGraph
        return MultiAgentGraph()


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
