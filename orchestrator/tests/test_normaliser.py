import json
from datetime import date
from unittest.mock import MagicMock

from orchestrator.agent_registry import AgentEntry
from orchestrator.nodes.normaliser import NormaliserNode

TEMPLATE = "Today is {today}. Pending: {pending}"
SUMMARY = "2026-10-18 | Colruyt | 90.00"


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        reply = MagicMock()
        reply.text = self.text
        return reply


def _entry(name, pending=None, owner_only=False):
    inst = MagicMock()
    inst.pending_summary.return_value = pending
    inst.confirm_pending.return_value = f"Saved: {pending}"
    return AgentEntry(name, "desc", inst, "anthropic", "m", owner_only=owner_only)


def _reply(text, pending=None):
    return json.dumps({"text": text, "pending": pending})


def _state(raw="yes", is_owner=True):
    return {"user_input": raw, "raw_input": raw, "session_id": "s1",
            "agent_outputs": {}, "turns": 0, "is_owner": is_owner, "direct": False}


def _node(llm, agents):
    return NormaliserNode(llm, TEMPLATE, agents, today=lambda: date(2026, 10, 18))


def test_cleans_text_without_pending():
    llm = FakeLLM(_reply("Colruyt €90 on 2026-10-18"))
    out = _node(llm, [_entry("conversational_agent")])(_state("colruit ninety euros today"))
    assert out == {"user_input": "Colruyt €90 on 2026-10-18"}
    system, human = llm.calls[0]
    assert "2026-10-18 (Sunday)" in system.content
    assert "Pending: None" in system.content
    assert human.content == "colruit ninety euros today"


def test_confirm_calls_confirm_pending():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    llm = FakeLLM(_reply("yes", "confirm"))
    out = _node(llm, [_entry("conversational_agent"), exp])(_state())
    exp.instance.confirm_pending.assert_called_once_with("s1")
    assert out == {"agent_outputs": {"expenditure_tracker": f"Saved: {SUMMARY}"},
                   "next": "synthesiser"}
    assert SUMMARY in llm.calls[0][0].content


def test_cancel_discards():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    out = _node(FakeLLM(_reply("no", "cancel")), [exp])(_state("no"))
    exp.instance.discard_pending.assert_called_once_with("s1")
    exp.instance.confirm_pending.assert_not_called()
    assert out == {"agent_outputs": {"expenditure_tracker": "Discarded."}, "next": "synthesiser"}


def test_correction_routes_direct_with_summary():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    out = _node(FakeLLM(_reply("it was €80", "correction")), [exp])(_state("it was eighty"))
    assert out == {
        "user_input": f"it was €80\n\n(Correction to the pending expense: {SUMMARY})",
        "next": "expenditure_tracker",
        "direct": True,
    }
    exp.instance.discard_pending.assert_not_called()


def test_unrelated_discards_and_goes_to_supervisor():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    out = _node(FakeLLM(_reply("what is AAPL at?", "unrelated")), [exp])(_state("what is apple at"))
    exp.instance.discard_pending.assert_called_once_with("s1")
    assert out == {"user_input": "what is AAPL at?"}


def test_null_while_pending_treated_as_unrelated():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    out = _node(FakeLLM(_reply("hello", None)), [exp])(_state("hello"))
    exp.instance.discard_pending.assert_called_once_with("s1")
    assert out == {"user_input": "hello"}


def test_parse_failure_keeps_raw_and_pending():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    out = _node(FakeLLM("sorry, I cannot"), [exp])(_state("yes"))
    assert out == {"user_input": "yes"}
    exp.instance.discard_pending.assert_not_called()
    exp.instance.confirm_pending.assert_not_called()


def test_fenced_json_is_parsed():
    llm = FakeLLM("```json\n" + _reply("hi there") + "\n```")
    out = _node(llm, [_entry("conversational_agent")])(_state("hi ther"))
    assert out == {"user_input": "hi there"}


def test_llm_error_behaves_like_parse_failure():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")
    out = _node(llm, [exp])(_state("yes"))
    assert out == {"user_input": "yes"}
    exp.instance.discard_pending.assert_not_called()


def test_owner_only_pending_ignored_for_non_owner():
    exp = _entry("expenditure_tracker", SUMMARY, owner_only=True)
    llm = FakeLLM(_reply("yes"))
    out = _node(llm, [exp])(_state("yes", is_owner=False))
    exp.instance.pending_summary.assert_not_called()
    assert "Pending: None" in llm.calls[0][0].content
    assert out == {"user_input": "yes"}
