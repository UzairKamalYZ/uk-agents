"""
orchestrator/nodes/normaliser.py — Normaliser node class.

Responsibility
--------------
First node of every fresh turn.  Cleans the raw message (transcription and
spelling fixes, spoken numbers → digits, relative dates → YYYY-MM-DD) so the
supervisor and agents see a tidy `user_input`.  `raw_input` keeps the original.

The reply is structured output (NormalisedMessage), so there is no JSON to
parse.  Any failure — LLM error, refusal, empty text — passes the raw message
through unchanged.

Replies to a pending confirmation never reach this node: the graph is paused
in the confirmation node and resumes there (see confirmation.py).

Usage
-----
    node = NormaliserNode(llm, prompt_template)
    graph_builder.add_node("normaliser", node)   # callable instance
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..state import GraphState

logger = logging.getLogger(__name__)


class NormalisedMessage(BaseModel):
    text: str = Field(description="The cleaned message")


class NormaliserNode:
    """
    LangGraph node that cleans the user's message.

    Parameters
    ----------
    llm             : chat model used for cleaning
    prompt_template : text loaded from NORMALISER_PROMPT_FILE — must contain
                      a {today} placeholder
    today           : date provider, injectable for tests
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: str,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._llm = llm.with_structured_output(NormalisedMessage, method="json_schema")
        self._prompt_template = prompt_template
        self._today = today

    # ------------------------------------------------------------------
    # LangGraph interface — called as normaliser_node(state)
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState) -> dict[str, Any]:
        raw = state.get("raw_input") or state["user_input"]
        today = self._today()
        prompt = self._prompt_template.replace("{today}", f"{today.isoformat()} ({today.strftime('%A')})")
        try:
            text = self._llm.invoke([SystemMessage(content=prompt), HumanMessage(content=raw)]).text.strip()
        except Exception:
            logger.exception("[normaliser] LLM call failed — passing raw input through")
            return {"user_input": raw}
        if not text:
            logger.warning("[normaliser] empty cleaned text — passing raw input through")
            return {"user_input": raw}
        return {"user_input": text}
