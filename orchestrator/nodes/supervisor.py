"""
orchestrator/nodes/supervisor.py — Supervisor node class.

Responsibility
--------------
After every agent turn the supervisor inspects the accumulated agent outputs
and decides which specialist agent to invoke next, or emits "FINISH" when
enough information has been gathered to synthesise a final answer.

The routing decision is returned as JSON: {"next": "<agent_name | FINISH>"}

Usage
-----
    node = SupervisorNode(llm, prompt_template)
    graph_builder.add_node("supervisor", node)   # callable instance
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState
from ._json import parse_json_object

logger = logging.getLogger(__name__)


class SupervisorNode:
    """
    LangGraph node that routes between specialist agents.

    Parameters
    ----------
    llm             : chat model used to produce the routing decision
    prompt_template : text loaded from SUPERVISOR_PROMPT_FILE, with the {agents}
                      list already filled in; the question and agent outputs
                      are sent separately as the human message
    """

    def __init__(self, llm: BaseChatModel, prompt_template: str, agent_names: set[str] = None) -> None:
        self._llm = llm
        self._prompt_template = prompt_template
        self._agent_names = agent_names or set()

    # ------------------------------------------------------------------
    # LangGraph interface — called as supervisor_node(state)
    # ------------------------------------------------------------------

    def __call__(self, state: GraphState) -> dict[str, Any]:
        agent_outputs_text = (
            "\n\n".join(
                f"[{name}]:\n{output}"
                for name, output in state.get("agent_outputs", {}).items()
            )
            or "None yet."
        )

        # User text and agent outputs stay out of the system prompt, so nothing in
        # them can pose as routing rules (prompt injection).
        messages = [
            SystemMessage(content=self._prompt_template),
            HumanMessage(content=(
                f"<user_question>\n{state['user_input']}\n</user_question>\n\n"
                f"<agent_outputs>\n{agent_outputs_text}\n</agent_outputs>\n\n"
                "Which agent should run next, or should we FINISH?"
            )),
        ]
        raw = self._llm.invoke(messages)
        content = raw.text

        next_agent = self._parse_next(content)
        logger.info("[supervisor] turns=%d → next=%s", state.get("turns", 0), next_agent)

        return {"next": next_agent}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_next(self, text: str) -> str:
        """
        Extract the routing token from the supervisor's raw LLM output.

        Priority:
          1-3. parse_json_object: strip fences, full JSON parse, then regex
               extraction of the first {...} block → extract "next" value.
          4. Fallback to "FINISH" (safe default — stops the loop).

        After extraction, the raw value is validated against known agent names.
        If the LLM returns something like "property_agent | FINISH", we scan the
        value for a known agent name and return that instead of defaulting to FINISH.
        """
        data = parse_json_object(text, "next")
        raw = str(data["next"]) if data is not None else None

        if raw is None:
            logger.warning("[supervisor] could not parse next from %r — defaulting to FINISH", text[:200])
            return "FINISH"

        # Exact match — clean output from a well-behaved LLM
        if raw in self._agent_names or raw == "FINISH":
            return raw

        # Fuzzy match — LLM included garbage alongside the agent name
        # e.g. "property_agent | FINISH", "call property_agent next"
        for name in self._agent_names:
            if name in raw:
                logger.warning("[supervisor] fuzzy-matched %r → %r", raw, name)
                return name

        logger.warning("[supervisor] unrecognised next=%r — defaulting to FINISH", raw)
        return "FINISH"
