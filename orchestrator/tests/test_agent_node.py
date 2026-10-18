from unittest.mock import MagicMock

from orchestrator.nodes.agent import AgentNode

REFUSAL = "Sorry, this is only available to the bot's owner."


def _state(**kw):
    base = {"user_input": "spent 5 on tea", "session_id": "s1", "agent_outputs": {}, "turns": 0}
    base.update(kw)
    return base


def test_owner_only_refuses_non_owner():
    agent = MagicMock()
    out = AgentNode("expenditure_tracker", agent, owner_only=True)(_state(is_owner=False))
    agent.ask.assert_not_called()
    assert out["agent_outputs"]["expenditure_tracker"] == REFUSAL
    assert out["messages"][0].content == REFUSAL
    assert out["turns"] == 1


def test_owner_only_refuses_when_flag_missing():
    agent = MagicMock()
    out = AgentNode("x", agent, owner_only=True)(_state())
    agent.ask.assert_not_called()
    assert out["agent_outputs"]["x"] == REFUSAL


def test_owner_only_allows_owner():
    agent = MagicMock()
    agent.ask.return_value = "logged"
    out = AgentNode("expenditure_tracker", agent, owner_only=True)(_state(is_owner=True))
    agent.ask.assert_called_once_with("spent 5 on tea", session_id="s1")
    assert out["agent_outputs"]["expenditure_tracker"] == "logged"
    assert out["turns"] == 1


def test_regular_agent_ignores_owner_flag():
    agent = MagicMock()
    agent.ask.return_value = "hi"
    out = AgentNode("conversational_agent", agent)(_state(is_owner=False))
    agent.ask.assert_called_once()
    assert out["agent_outputs"]["conversational_agent"] == "hi"


def test_graph_initial_state_carries_owner_flag():
    from orchestrator.graph import MultiAgentGraph

    st = MultiAgentGraph._initial_state("hello", "s", True)
    assert st["is_owner"] is True
    assert st["raw_input"] == "hello"
    assert st["direct"] is False
    assert st["turns"] == 0
