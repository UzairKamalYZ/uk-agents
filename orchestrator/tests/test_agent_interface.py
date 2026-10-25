import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent_interface import OrchestratorAgent


@pytest.fixture
def facade():
    with patch("orchestrator.agent_interface.MultiAgentGraph") as graph_cls:
        graph = graph_cls.return_value
        graph.ask.return_value = "sync answer"
        graph.aask = AsyncMock(return_value="async answer")
        graph.stream_tokens.return_value = iter(["a", "b"])

        async def tokens(*args, **kwargs):
            yield "x"
            yield "y"

        graph.astream_tokens = MagicMock(side_effect=tokens)
        yield OrchestratorAgent(), graph


def test_ask_delegates_with_session_and_owner(facade):
    agent, graph = facade
    assert agent.ask("hi", session_id="s", is_owner=True) == "sync answer"
    graph.ask.assert_called_once_with("hi", session_id="s", is_owner=True)


def test_ask_stream_yields_tokens(facade):
    agent, graph = facade
    assert list(agent.ask("hi", stream=True, session_id="s")) == ["a", "b"]


def test_aask_awaits_the_graph(facade):
    agent, graph = facade
    assert asyncio.run(agent.aask("hi", session_id="s", is_owner=True)) == "async answer"
    graph.aask.assert_awaited_once_with("hi", session_id="s", is_owner=True)


def test_astream_yields_tokens(facade):
    agent, graph = facade

    async def collect():
        return [t async for t in agent.astream("hi", session_id="s")]

    assert asyncio.run(collect()) == ["x", "y"]
    graph.astream_tokens.assert_called_once_with("hi", session_id="s", is_owner=False)


def test_close_closes_the_graph(facade):
    agent, graph = facade
    with agent:
        pass
    graph.close.assert_called_once()
