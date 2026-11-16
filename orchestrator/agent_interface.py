"""
orchestrator/agent_interface.py — the facade every client talks to.

    agent.ask(prompt, session_id=...)            blocking, for sync callers (cron, scripts)
    agent.ask(prompt, stream=True)               blocking generator of answer tokens
    await agent.aask(prompt, session_id=...)     for clients with an event loop (REST, Telegram)
    agent.astream(prompt, session_id=...)        async generator of answer tokens

The graph runs on its own background loop, so all four can be called from anywhere.
Only the synthesiser's tokens are streamed; a single-agent answer arrives as one chunk.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Callable, Iterator

from langsmith import traceable

from .agent_registry import AgentEntry
from .graph import MultiAgentGraph
from .tracing import tag_current_trace

logger = logging.getLogger(__name__)

OnRoute = Callable[[AgentEntry], None]
_traced_request = traceable(name="orchestrator.request", run_type="chain", tags=["orchestrator"])


class OrchestratorAgent:
    """Loads the agents in agents.json and compiles the graph once; reused for every request."""

    def __init__(self) -> None:
        self._graph = MultiAgentGraph()
        logger.info("[OrchestratorAgent] ready")

    @_traced_request
    def ask(
        self, prompt: str, stream: bool = False, session_id: str = "default", *, on_route: OnRoute | None = None
    ) -> str | Iterator[str]:
        """Answer one message.  on_route (blocking path only) is called with each agent's
        AgentEntry just before it runs, e.g. to show who is working on the query."""
        if stream:
            return self._graph.stream_tokens(prompt, session_id=session_id)
        return self._graph.ask(prompt, session_id=session_id, on_route=self._tagging(on_route))

    @_traced_request
    async def aask(self, prompt: str, session_id: str = "default", *, on_route: OnRoute | None = None) -> str:
        """Async ask().  on_route runs on the orchestrator's loop thread — see MultiAgentGraph.aask."""
        return await self._graph.aask(prompt, session_id=session_id, on_route=self._tagging(on_route))

    def astream(self, prompt: str, session_id: str = "default") -> AsyncIterator[str]:
        """Async generator of answer tokens as the synthesiser writes them."""
        return self._graph.astream_tokens(prompt, session_id=session_id)

    @staticmethod
    def _tagging(on_route: OnRoute | None) -> OnRoute:
        """Tag each picked agent onto this request's LangSmith trace, then call on_route."""
        routed: list[str] = []

        def _on_route(entry: AgentEntry) -> None:
            routed.append(entry.name)
            tag_current_trace([f"agent:{entry.name}"], {"agents": list(routed)})
            if on_route is not None:
                on_route(entry)

        return _on_route

    def close(self) -> None:
        """Close the checkpointer and stop the orchestrator's event loop."""
        self._graph.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
