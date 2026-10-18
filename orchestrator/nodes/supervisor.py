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

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import GraphState

logger = logging.getLogger(__name__)

# Strips markdown code fences (```json ... ```) before JSON parsing
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Fallback: extracts the first {...} block when full JSON parsing fails
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


class SupervisorNode:
    """
    LangGraph node that routes between specialist agents.

    Parameters
    ----------
    llm             : chat model used to produce the routing decision
    prompt_template : text loaded from SUPERVISOR_PROMPT_FILE — must contain
                      {user_input} and {agent_outputs} placeholders
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

        prompt = (
            self._prompt_template
            .replace("{user_input}", state["user_input"])
            .replace("{agent_outputs}", agent_outputs_text)
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Which agent should run next, or should we FINISH?"),
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
          1. Strip markdown fences.
          2. Full JSON parse → extract "next" value.
          3. Regex extraction of the first {...} block.
          4. Fallback to "FINISH" (safe default — stops the loop).

        After extraction, the raw value is validated against known agent names.
        If the LLM returns something like "property_agent | FINISH", we scan the
        value for a known agent name and return that instead of defaulting to FINISH.
        """
        cleaned = _FENCE_RE.sub(r"\1", text).strip()

        raw = None
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "next" in data:
                raw = str(data["next"])
        except json.JSONDecodeError:
            pass

        if raw is None:
            m = _JSON_RE.search(cleaned)
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, dict) and "next" in data:
                        raw = str(data["next"])
                except json.JSONDecodeError:
                    pass

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
