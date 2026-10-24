"""
orchestrator/nodes/agent.py — Agent node class.

Responsibility
--------------
Wraps a single specialist agent (property, finance, conversational, …) as a
LangGraph node.  Calls the agent's ask() method, stores the response in
state["agent_outputs"], and increments the turns counter so the router can
enforce MAX_AGENT_TURNS.

Usage
-----
    node = AgentNode("property_agent", agent_instance)
    graph_builder.add_node("property_agent", node)   # callable instance
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..state import GraphState

logger = logging.getLogger(__name__)

OWNER_ONLY_REFUSAL = "Sorry, this is only available to the bot's owner."


class AgentNode:
    """
    LangGraph node that delegates to one specialist agent.

    Parameters
    ----------
    agent_name     : routing key for this agent (e.g. "property_agent") —
                     used as the key in state["agent_outputs"] and in logs
    agent_instance : instantiated BaseAgent; must expose
                     ask(prompt: str, session_id: str) -> str
    owner_only     : if True, the agent is only called when state["is_owner"]
                     is truthy; otherwise a refusal message is returned and
                     the agent is never invoked
    """

    def __init__(
        self, agent_name: str, agent_instance: Any, owner_only: bool = False
    ) -> None:
        self._name = agent_name
        self._agent = agent_instance
        self._owner_only = owner_only
        # Give the instance a readable __name__ so LangGraph's visualisation
        # and debug logs show the agent name rather than "AgentNode.__call__"
        self.__name__ = agent_name

    # ------------------------------------------------------------------
    # LangGraph interface — called as agent_node(state)
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState) -> dict[str, Any]:
        prompt = state["user_input"]
        session_id = state.get("session_id", "default")

        update: dict[str, Any] = {}
        if self._owner_only and not state.get("is_owner", False):
            logger.warning("[%s] owner-only agent refused non-owner request", self._name)
            response_text = OWNER_ONLY_REFUSAL
        else:
            logger.info("[%s] calling with session=%s", self._name, session_id)
            response_text = self._agent.ask(prompt, session_id=session_id)
            logger.info("[%s] done, response_len=%d", self._name, len(response_text))
            if self._has_pending(session_id):
                # The agent asked the user to confirm something — pause after answering
                update["awaiting"] = self._name

        updated_outputs = dict(state.get("agent_outputs", {}))
        updated_outputs[self._name] = response_text

        return {
            **update,
            "agent_outputs": updated_outputs,
            "messages": [AIMessage(content=response_text, name=self._name)],
            "turns": state.get("turns", 0) + 1,
        }

    def _has_pending(self, session_id: str) -> bool:
        try:
            return self._agent.pending_summary(session_id) is not None
        except Exception:
            logger.exception("[%s] pending_summary failed — treating as nothing pending", self._name)
            return False
