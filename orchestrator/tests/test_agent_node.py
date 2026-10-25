import asyncio
from unittest.mock import AsyncMock, MagicMock

from orchestrator.nodes.agent import AgentNode

def _state(**kw):
    base = {"user_input": "spent 5 on tea", "session_id": "s1", "agent_outputs": {}, "turns": 0}
    base.update(kw)
    return base


def test_agent_answer_is_stored():
    agent = MagicMock()
    agent.ask.return_value = "hi"
    out = asyncio.run(AgentNode("conversational_agent", agent)(_state()))
    agent.ask.assert_called_once()
    assert out["agent_outputs"]["conversational_agent"] == "hi"


def test_graph_initial_state():
    from orchestrator.graph import MultiAgentGraph

    st = MultiAgentGraph._initial_state("hello", "s", "user: hi")
    assert st["history"] == "user: hi"
    assert "is_owner" not in st
    assert st["raw_input"] == "hello"
    assert st["plan"] == []
    assert st["turns"] == 0


def test_awaits_agent_aask_when_available():
    class AsyncAgent:
        def __init__(self):
            self.ask = MagicMock()
            self.aask = AsyncMock(return_value="async reply")

        def pending_summary(self, session_id):
            return None

    agent = AsyncAgent()
    out = asyncio.run(AgentNode("a", agent)(_state()))
    agent.aask.assert_awaited_once_with("spent 5 on tea", session_id="s1")
    agent.ask.assert_not_called()
    assert out["agent_outputs"]["a"] == "async reply"
    assert "awaiting" not in out


def test_sync_only_agent_runs_off_the_event_loop():
    import threading
    seen = []

    class SyncAgent:
        def ask(self, prompt, session_id=None):
            seen.append(threading.current_thread())
            return "sync reply"

        def pending_summary(self, session_id):
            return "something pending"

    async def run():
        return threading.current_thread(), await AgentNode("a", SyncAgent())(_state())

    loop_thread, out = asyncio.run(run())
    assert out["agent_outputs"]["a"] == "sync reply"
    assert seen[0] is not loop_thread
    assert out["awaiting"] == "a"


LEAKED = '<invoke name="Skill">\n<parameter name="skill">fbr-iris-filing</parameter>\n</invoke>'


def test_answer_that_is_only_tool_call_markup_is_withheld(caplog):
    from core.src.output_guard import WITHHELD_REPLY

    agent = MagicMock()
    agent.ask.return_value = LEAKED
    agent.pending_summary.return_value = None
    with caplog.at_level("WARNING"):
        out = asyncio.run(AgentNode("fbr_tax_filing", agent)(_state()))
    assert out["agent_outputs"]["fbr_tax_filing"] == WITHHELD_REPLY
    assert out["messages"][0].content == WITHHELD_REPLY
    assert "[fbr_tax_filing] answer contained tool-call markup ['invoke']" in caplog.text


def test_markup_inside_an_answer_is_removed_and_the_rest_kept():
    agent = MagicMock()
    agent.ask.return_value = f"Checking…\n{LEAKED}\nYour 2025 return is submitted."
    agent.pending_summary.return_value = None
    out = asyncio.run(AgentNode("fbr_tax_filing", agent)(_state()))
    assert out["agent_outputs"]["fbr_tax_filing"] == "Checking…\n\nYour 2025 return is submitted."
