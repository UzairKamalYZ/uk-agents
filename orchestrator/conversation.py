"""
orchestrator/conversation.py — short conversation memory for routing follow-ups.

Agents keep their own chat history, but the router used to see only the latest
message.  A reply such as "yes" or "do that for last month too",
answering the bot's own follow-up question, was routed as if it were a new
conversation.  ConversationMemory stores the last few user/assistant turns per
session (in the same sessions.db as agent history, under the pseudo-agent
"orchestrator") and renders them as a block of text for the prompts.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from core.src.model.session_manager import HistoryStore
from core.src.output_guard import scrub

logger = logging.getLogger(__name__)

HISTORY_TURNS = 3       # user + assistant pairs shown to the router
_MAX_CHARS = 600        # per message, so a long answer doesn't flood the routing call
_STORE_KEY = "orchestrator"


class ConversationMemory:
    """Recent turns of one session, for the router.  Failures never block a request."""

    def __init__(self, store: HistoryStore) -> None:
        self._store = store

    def recent(self, session_id: str) -> str:
        """The last turns as 'user: …' / '<agent>: …' lines, oldest first ("" if none)."""
        try:
            messages = self._store.get(_STORE_KEY, session_id, HISTORY_TURNS * 2)
        except Exception:
            logger.exception("[conversation] could not load history (session=%s)", session_id)
            return ""
        lines = []
        for m in messages:
            speaker = "user" if isinstance(m, HumanMessage) else (m.name or "assistant")
            # Older turns may hold leaked tool-call markup; never show it to the router again
            text = " ".join(scrub(str(m.content)).text.split())
            lines.append(f"{speaker}: {text[:_MAX_CHARS]}{'…' if len(text) > _MAX_CHARS else ''}")
        return "\n".join(lines)

    def record(self, session_id: str, user_text: str, answer: str, agents: list[str]) -> None:
        """Store one finished turn; the answer is attributed to the agent(s) that produced it."""
        if not answer:
            return
        try:
            self._store.append(_STORE_KEY, session_id, [
                HumanMessage(content=user_text),
                AIMessage(content=answer, name="+".join(agents) or "assistant"),
            ])
        except Exception:
            logger.exception("[conversation] could not save history (session=%s)", session_id)
