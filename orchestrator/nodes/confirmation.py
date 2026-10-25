"""
orchestrator/nodes/confirmation.py — Confirmation node class.

Responsibility
--------------
Human-in-the-loop for pending items (e.g. an expense the agent wasn't sure
about).  Runs after the synthesiser whenever state["awaiting"] names an agent:

  1. Re-checks the agent's pending item; nothing pending → END.
  2. interrupt() — the graph pauses and is saved by the checkpointer.  The
     user sees the agent's question (already answered by the synthesiser).
  3. The user's next message resumes the graph here (MultiAgentGraph passes it
     as Command(resume=...)).  One structured LLM call cleans the reply and
     classifies it against the pending item.

LangGraph re-runs a node from the top on resume, so step 1 runs again then —
it only reads, which is safe.

Outcomes (state["next"] picks the following node)
--------
    confirm            → agent.confirm_pending() output        → synthesiser
    cancel             → agent.discard_pending(), "Discarded." → synthesiser
    correction         → cleaned reply + summary, plan=[agent] → that agent
    unrelated          → agent.discard_pending(), fresh turn   → router
    LLM error/refusal  → re-ask, pending kept                  → synthesiser
    confirm/discard raises → PENDING_ERROR_REPLY               → synthesiser

After confirm / correction / re-ask the agent may still hold a pending item
(e.g. the amount is still missing), so `awaiting` stays set and the graph
comes back here after the synthesiser to pause again.

Usage
-----
    node = ConfirmationNode(llm, prompt_template, agents)
    graph_builder.add_node("confirmation", node)   # callable instance
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any, Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ..agent_registry import AgentEntry
from ..state import GraphState

logger = logging.getLogger(__name__)

DISCARDED_REPLY = "Discarded."
REASK_REPLY = (
    "Sorry, I didn't catch that. Pending: {summary}. "
    "Reply yes to save, no to discard, or send the corrected version."
)
PENDING_ERROR_REPLY = "Sorry, something went wrong with the pending item. Please try again."


class ConfirmationReply(BaseModel):
    verdict: Literal["confirm", "cancel", "correction", "unrelated"] = Field(
        description="How the reply relates to the pending item"
    )
    text: str = Field(description="The cleaned reply")


class ConfirmationNode:
    """
    LangGraph node that pauses for the user's answer to a pending item.

    Parameters
    ----------
    llm             : chat model used to classify the reply
    prompt_template : text loaded from CONFIRMATION_PROMPT_FILE — must contain
                      {pending} and {today} placeholders
    agents          : enabled agents (pending_summary / confirm_pending /
                      discard_pending hooks from BaseAgent)
    today           : date provider, injectable for tests
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: str,
        agents: list[AgentEntry],
        today: Callable[[], date] = date.today,
    ) -> None:
        self._llm = llm.with_structured_output(ConfirmationReply, method="json_schema")
        self._prompt_template = prompt_template
        self._agents = {a.name: a for a in agents}
        self._today = today

    # ------------------------------------------------------------------
    # LangGraph interface — called as confirmation_node(state)
    # ------------------------------------------------------------------

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        name = state.get("awaiting")
        session_id = state.get("session_id", "default")
        entry = self._agents.get(name)
        summary = await asyncio.to_thread(self._pending_summary, entry, session_id) if entry else None
        if summary is None:
            return {"awaiting": None, "next": END}

        logger.info("[confirmation] pausing for %s: %s", name, summary)
        reply: str = interrupt({"agent": name, "summary": summary})

        # ---- resumed with the user's next message ----
        fresh_turn = {"raw_input": reply, "user_input": reply, "agent_outputs": {}, "turns": 0, "plan": []}
        classified = await self._classify(reply, summary)
        if classified is None:
            return {**fresh_turn, "agent_outputs": {name: REASK_REPLY.format(summary=summary)},
                    "awaiting": name, "next": "synthesiser"}

        verdict, text = classified.verdict, classified.text.strip() or reply
        logger.info("[confirmation] %s → %s", name, verdict)
        agent = entry.instance

        if verdict == "correction":
            return {**fresh_turn, "user_input": f"{text}\n\n(Correction to the pending expense: {summary})",
                    "plan": [name], "awaiting": None, "next": name}

        try:
            if verdict == "confirm":
                saved = await asyncio.to_thread(agent.confirm_pending, session_id)
                return {**fresh_turn, "agent_outputs": {name: saved},
                        "awaiting": name, "next": "synthesiser"}
            await asyncio.to_thread(agent.discard_pending, session_id)
        except Exception:
            logger.exception("[confirmation] %s failed resolving pending item (verdict=%s)", name, verdict)
            return {**fresh_turn, "agent_outputs": {name: PENDING_ERROR_REPLY}, "awaiting": name,
                    "next": "synthesiser"}

        if verdict == "cancel":
            return {**fresh_turn, "agent_outputs": {name: DISCARDED_REPLY}, "awaiting": None,
                    "next": "synthesiser"}
        # "unrelated" — the user moved on: handle the message as a brand-new turn
        return {**fresh_turn, "awaiting": None, "next": "router"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pending_summary(entry: AgentEntry, session_id: str) -> str | None:
        try:
            return entry.instance.pending_summary(session_id)
        except Exception:
            logger.exception("[confirmation] pending_summary failed for %s — treating as none", entry.name)
            return None

    async def _classify(self, reply: str, summary: str) -> ConfirmationReply | None:
        today = self._today()
        prompt = (
            self._prompt_template
            .replace("{pending}", summary)
            .replace("{today}", f"{today.isoformat()} ({today.strftime('%A')})")
        )
        try:
            return await self._llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=reply)])
        except Exception:
            logger.exception("[confirmation] classification failed — re-asking")
            return None
