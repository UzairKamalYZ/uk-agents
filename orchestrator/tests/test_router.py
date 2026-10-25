import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda

from orchestrator.nodes.router import RouterNode


def _node(reply, names=("a", "b", "c")):
    llm = MagicMock()

    def run(messages):
        if isinstance(reply, Exception):
            raise reply
        return reply

    llm.with_structured_output.return_value = RunnableLambda(run)
    node = RouterNode(llm, "{agents} {today}", "", set(names), today=lambda: date(2026, 10, 18))
    return node, llm


def _run(node, raw="raw message"):
    return asyncio.run(node({"user_input": raw, "raw_input": raw}))


def test_uses_native_json_schema_output():
    node, llm = _node(SimpleNamespace(text="t", agents=[]))
    assert llm.with_structured_output.call_args.kwargs == {"method": "json_schema"}


def test_returns_cleaned_text_and_plan():
    node, _ = _node(SimpleNamespace(text="  Colruyt €90  ", agents=["b", "a"]))
    assert _run(node) == {"user_input": "Colruyt €90", "plan": ["b", "a"]}


def test_blank_text_keeps_the_raw_message():
    node, _ = _node(SimpleNamespace(text="  ", agents=["a"]))
    assert _run(node)["user_input"] == "raw message"


def test_duplicates_dropped_keeping_first_order():
    node, _ = _node(SimpleNamespace(text="t", agents=["b", "a", "b"]))
    assert _run(node)["plan"] == ["b", "a"]


def test_plan_capped_at_max_agent_turns():
    node, _ = _node(SimpleNamespace(text="t", agents=["a", "b", "c"]))
    with patch("orchestrator.nodes.router.MAX_AGENT_TURNS", 2):
        assert _run(node)["plan"] == ["a", "b"]


def test_llm_error_passes_raw_text_with_empty_plan():
    node, _ = _node(RuntimeError("down"))
    assert _run(node) == {"user_input": "raw message", "plan": []}
