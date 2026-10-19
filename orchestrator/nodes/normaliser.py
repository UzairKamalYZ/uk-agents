"""
orchestrator/nodes/normaliser.py — Normaliser node class.

Responsibility
--------------
First node of every turn.  One LLM call does two jobs:

  1. Cleans the raw message (transcription/spelling fixes, spoken numbers →
     digits, relative dates → YYYY-MM-DD) so the supervisor and agents see a
     tidy `user_input`.  `raw_input` keeps the original.
  2. When an agent holds a pending item awaiting confirmation (e.g. an
     expense), classifies the message against it.  The supervisor sees no
     chat history, so a bare "yes" can only be resolved here.

The LLM replies with JSON: {"text": "...", "pending": "confirm" | "cancel" |
"correction" | "unrelated" | null}.

Outcomes
--------
    no pending, parse failure / LLM error → raw text                       → supervisor
    no pending                            → cleaned text                   → supervisor
    pending, parse failure / LLM error /
      unknown verdict                     → re-ask, pending untouched      → synthesiser
    confirm                               → agent.confirm_pending() output → synthesiser
    cancel                                → agent.discard_pending()        → synthesiser
    correction                            → cleaned text + summary, direct → that agent
    unrelated / null                      → agent.discard_pending()        → supervisor
    pending_summary raises                → treated as no pending for that agent
    confirm/discard_pending raises        → PENDING_ERROR_REPLY            → synthesiser

Verdicts are matched case-insensitively after stripping whitespace.

Usage
-----
    node = NormaliserNode(llm, prompt_template, agents)
    graph_builder.add_node("normaliser", node)   # callable instance
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..agent_registry import AgentEntry
from ..state import GraphState
from ..tracing import record_route
from ._json import parse_json_object

logger = logging.getLogger(__name__)

DISCARDED_REPLY = "Discarded."
REASK_REPLY = (
    "Sorry, I didn't catch that. Pending: {summary}. "
    "Reply yes to save, no to discard, or send the corrected version."
)
PENDING_ERROR_REPLY = "Sorry, something went wrong with the pending item. Please try again."
_VERDICTS = {"confirm", "cancel", "correction", "unrelated"}


class NormaliserNode:
    """
    LangGraph node that cleans the user's message and resolves pending items.

    Parameters
    ----------
    llm             : chat model used for cleaning + classification
    prompt_template : text loaded from NORMALISER_PROMPT_FILE — must contain
                      {today} and {pending} placeholders
    agents          : enabled agents; each instance exposes pending_summary /
                      confirm_pending / discard_pending (BaseAgent hooks)
    today           : date provider, injectable for tests
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: str,
        agents: list[AgentEntry],
        today: Callable[[], date] = date.today,
    ) -> None:
        self._llm = llm
        self._prompt_template = prompt_template
        self._agents = agents
        self._today = today

    # ------------------------------------------------------------------
    # LangGraph interface — called as normaliser_node(state)
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState) -> dict[str, Any]:
        result = self._run(state)
        # When this node decided the agent itself (pending item), record it like a supervisor route.
        routed = result.get("next")
        if routed == "synthesiser" and result.get("agent_outputs"):
            routed = next(iter(result["agent_outputs"]))
        if routed and routed != "synthesiser":
            record_route("normaliser", routed, "pending item", state.get("request_id"))
        return result

    def _run(self, state: GraphState) -> dict[str, Any]:
        raw = state.get("raw_input") or state["user_input"]
        session_id = state.get("session_id", "default")
        pending_agent, summary = self._find_pending(session_id, state.get("is_owner", False))

        reply = self._classify(raw, summary, state.get("history") or "")
        if reply is not None and reply[0] != raw and state.get("history"):
            logger.info("[normaliser] request=%s rewrote %r → %r", state.get("request_id"), raw, reply[0])
        if pending_agent is None:
            return {"user_input": reply[0] if reply is not None else raw}

        name, agent = pending_agent.name, pending_agent.instance
        verdict = reply[1] if reply is not None else None
        if reply is None or (verdict is not None and verdict not in _VERDICTS):
            # Classification failed — never guess with a pending item at stake
            logger.warning("[normaliser] pending on %s, unclassified (verdict=%r) — re-asking", name, verdict)
            return {"agent_outputs": {name: REASK_REPLY.format(summary=summary)}, "next": "synthesiser"}

        text = reply[0]
        logger.info("[normaliser] pending on %s → %s", name, verdict)

        if verdict == "correction":
            return {
                "user_input": f"{text}\n\n(Correction to the pending expense: {summary})",
                "next": name,
                "direct": True,
            }

        try:
            if verdict == "confirm":
                return {"agent_outputs": {name: agent.confirm_pending(session_id)}, "next": "synthesiser"}
            agent.discard_pending(session_id)
        except Exception:
            logger.exception("[normaliser] %s failed resolving pending item (verdict=%s)", name, verdict)
            return {"agent_outputs": {name: PENDING_ERROR_REPLY}, "next": "synthesiser"}

        if verdict == "cancel":
            return {"agent_outputs": {name: DISCARDED_REPLY}, "next": "synthesiser"}
        # "unrelated" or null — the user moved on
        return {"user_input": text}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pending(self, session_id: str, is_owner: bool) -> tuple[AgentEntry | None, str | None]:
        """First agent with a pending item; owner-only agents skipped for non-owners."""
        for entry in self._agents:
            if entry.owner_only and not is_owner:
                continue
            try:
                summary = entry.instance.pending_summary(session_id)
            except Exception:
                logger.exception("[normaliser] pending_summary failed for %s — treating as no pending", entry.name)
                continue
            if summary is not None:
                return entry, summary
        return None, None

    def _classify(self, raw: str, summary: str | None, history: str = "") -> tuple[str, str | None] | None:
        """One LLM call → (cleaned_text, verdict), or None on any failure."""
        today = self._today()
        prompt = (
            self._prompt_template
            .replace("{today}", f"{today.isoformat()} ({today.strftime('%A')})")
            .replace("{pending}", summary if summary is not None else "None")
            .replace("{history}", history or "None")
        )
        try:
            content = self._llm.invoke([SystemMessage(content=prompt), HumanMessage(content=raw)]).text
        except Exception:
            logger.exception("[normaliser] LLM call failed — passing raw input through")
            return None

        data = parse_json_object(content, "text")
        if data is None or not isinstance(data["text"], str) or not data["text"].strip():
            logger.warning("[normaliser] could not parse reply %r — passing raw input through", content[:200])
            return None

        verdict = data.get("pending")
        if verdict is None:
            return data["text"], None
        return data["text"], str(verdict).strip().lower()
