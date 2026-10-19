"""
orchestrator/agent_interface.py — client-facing facade over the multi-agent graph.

WHY THIS FILE EXISTS
--------------------
All clients (REST API, Telegram bot, cron job) were originally written against
LocalAgent, which exposes:

    agent.ask(prompt, stream=False, session_id="default") -> str | Generator

OrchestratorAgent provides the exact same interface so every client can switch
to the multi-agent pipeline by changing one import line — nothing else changes.

STREAMING NOTE
--------------
_stream_tokens() delegates to MultiAgentGraph.stream_tokens(), which uses
LangGraph's stream_mode="messages" to intercept AIMessageChunk objects from
the synthesiser node as the LLM produces them.  Agent-node tokens are filtered
out so only the final synthesis reaches the caller.
"""
from __future__ import annotations

import logging
from typing import Callable, Generator

from langsmith import traceable

from .agent_registry import AgentEntry
from .graph import MultiAgentGraph
from .tracing import tag_current_trace

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Thin facade over MultiAgentGraph.

    Matches the LocalAgent interface used by all clients:
        agent.ask(prompt)                      → str
        agent.ask(prompt, stream=True)         → Generator[str, None, None]
        agent.ask(prompt, session_id="abc")    → str
        agent.close()                          → None  (context-manager safe)

    Construction
    ------------
    Instantiating OrchestratorAgent triggers MultiAgentGraph.__init__, which:
      - loads agents from agents.json
      - instantiates each enabled agent (LLM + vector store connections)
      - compiles the LangGraph StateGraph
    This happens once at startup and is reused for every subsequent ask().
    """

    def __init__(self) -> None:
        # Build and compile the full multi-agent graph
        self._graph = MultiAgentGraph()
        logger.info("[OrchestratorAgent] ready")

    @traceable(
        name="orchestrator.request",
        run_type="chain",
        tags=["orchestrator"],
    )
    def ask(
        self,
        prompt: str,
        stream: bool = False,
        session_id: str = "default",
        *,
        is_owner: bool = False,
        on_route: Callable[[AgentEntry], None] | None = None,
    ) -> str | Generator[str, None, None]:
        """
        Ask the multi-agent pipeline a question.

        Parameters
        ----------
        prompt     : the user's raw question
        stream     : if True, returns a generator that yields text chunks;
                     if False (default), blocks until the full answer is ready
        session_id : passed through to each agent so per-session history is
                     maintained correctly across turns
        is_owner   : True when the caller is the bot's owner; agents marked
                     "owner_only" in agents.json refuse non-owner requests
        on_route   : blocking path only — called with each agent's AgentEntry as
                     soon as it is picked, before it runs (e.g. to show
                     "Expenditure Tracker is working on it…")

        Returns
        -------
        str                        when stream=False
        Generator[str, None, None] when stream=True
        """
        if stream:
            # Return a generator — the caller iterates it to receive chunks
            return self._stream_tokens(prompt, session_id, is_owner)

        # Blocking path — run the full graph and return the final string.
        # Each picked agent is also tagged onto this request's LangSmith trace.
        routed: list[str] = []

        def _on_route(entry: AgentEntry) -> None:
            routed.append(entry.name)
            tag_current_trace([f"agent:{entry.name}"], {"agents": list(routed)})
            if on_route is not None:
                on_route(entry)

        return self._graph.ask(prompt, session_id=session_id, is_owner=is_owner, on_route=_on_route)

    def _stream_tokens(
        self, prompt: str, session_id: str, is_owner: bool = False
    ) -> Generator[str, None, None]:
        """
        Yield real tokens from the synthesiser LLM as they are generated.

        Delegates to MultiAgentGraph.stream_tokens() which uses LangGraph's
        stream_mode="messages" to intercept token chunks from the synthesiser
        node in real time.  Agent-node tokens (property, finance, conversational)
        are filtered out inside stream_tokens() — only the final synthesis
        reaches the caller.

        The caller (REST SSE endpoint, Telegram live-edit) receives chunks
        immediately as the LLM writes them, not after the whole pipeline finishes.
        """
        yield from self._graph.stream_tokens(prompt, session_id=session_id, is_owner=is_owner)

    def close(self) -> None:
        """No persistent resources to release — graph is stateless between calls."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
