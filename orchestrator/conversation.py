"""
orchestrator/conversation.py — short conversation memory for routing follow-ups.

Agents keep their own chat history, but the normaliser and supervisor used to see
only the latest message.  A reply such as "yes" or "do that for last month too",
answering the bot's own follow-up question, was routed as if it were a new
conversation.  ConversationMemory stores the last few user/assistant turns per
session (in the same sessions.db as agent history, under the pseudo-agent
"orchestrator") and renders them as a block of text for the prompts.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from core.src.model.session_manager import SessionManager

logger = logging.getLogger(__name__)

HISTORY_TURNS = 3       # user + assistant pairs shown to the normaliser and supervisor
_MAX_CHARS = 600        # per message, so a long answer doesn't flood the routing prompts
_STORE_KEY = "orchestrator"


class ConversationMemory:
    """Recent turns of one session, for the routing prompts.  Failures never block a request."""

    def __init__(self, store: SessionManager) -> None:
        self._store = store

    def recent(self, session_id: str) -> str:
        """The last turns as 'user: …' / '<agent>: …' lines, oldest first ("" if none)."""
        try:
            messages = self._store.load(_STORE_KEY, session_id, HISTORY_TURNS * 2)
        except Exception:
            logger.exception("[conversation] could not load history (session=%s)", session_id)
            return ""
        lines = []
        for m in messages:
            speaker = "user" if isinstance(m, HumanMessage) else (m.name or "assistant")
            text = " ".join(str(m.content).split())
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
