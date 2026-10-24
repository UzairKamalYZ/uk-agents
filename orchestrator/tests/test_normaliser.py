from datetime import date
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from orchestrator.nodes.normaliser import NormalisedMessage, NormaliserNode

TEMPLATE = "Today is {today}."


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs

        def run(messages):
            self.calls.append(messages)
            if isinstance(self.reply, Exception):
                raise self.reply
            return schema(text=self.reply)

        return RunnableLambda(run)


def _state(raw):
    return {"user_input": raw, "raw_input": raw, "session_id": "s1"}


def _node(llm):
    return NormaliserNode(llm, TEMPLATE, today=lambda: date(2026, 10, 18))


def test_cleans_text_with_structured_output():
    llm = FakeLLM("Colruyt €90 on 2026-10-18")
    out = _node(llm)(_state("colruit ninety euros today"))
    assert out == {"user_input": "Colruyt €90 on 2026-10-18"}
    assert llm.schema is NormalisedMessage
    assert llm.kwargs == {"method": "json_schema"}
    system, human = llm.calls[0]
    assert system.content == "Today is 2026-10-18 (Sunday)."
    assert human.content == "colruit ninety euros today"


def test_llm_error_passes_raw_through():
    assert _node(FakeLLM(RuntimeError("down")))(_state("hi")) == {"user_input": "hi"}


def test_blank_cleaned_text_passes_raw_through():
    assert _node(FakeLLM("   "))(_state("hi")) == {"user_input": "hi"}


def test_uses_raw_input_over_user_input():
    llm = FakeLLM("x")
    _node(llm)({"user_input": "already cleaned", "raw_input": "original"})
    assert llm.calls[0][1].content == "original"
