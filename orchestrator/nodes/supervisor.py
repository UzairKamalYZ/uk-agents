"""
orchestrator/nodes/supervisor.py — Supervisor node class.

Responsibility
--------------
After every agent turn the supervisor inspects the accumulated agent outputs
and decides which specialist agent to invoke next, or emits "FINISH" when
enough information has been gathered to synthesise a final answer.

The decision is structured output: a RouteDecision whose `next` field is
constrained to the enabled agent names plus "FINISH", so the model cannot
answer with anything else.  An LLM error or refusal falls back to "FINISH".

Usage
-----
    node = SupervisorNode(llm, prompt_template, agent_names)
    graph_builder.add_node("supervisor", node)   # callable instance
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

from ..state import GraphState

logger = logging.getLogger(__name__)


FINISH = "FINISH"


def route_decision_model(agent_names: set[str]) -> type[BaseModel]:
    """RouteDecision with `next` limited to the known agents and FINISH."""
    choices = tuple(sorted(agent_names)) + (FINISH,)
    return create_model(
        "RouteDecision",
        next=(Literal[choices], Field(description="The agent to call next, or FINISH")),
    )


class SupervisorNode:
    """
    LangGraph node that routes between specialist agents.

    Parameters
    ----------
    llm             : chat model used to produce the routing decision
    prompt_template : text loaded from SUPERVISOR_PROMPT_FILE, with the {agents}
                      list already filled in; the question and agent outputs
                      are sent separately as the human message
    agent_names     : enabled agent names — the only values `next` may take
    """

    def __init__(self, llm: BaseChatModel, prompt_template: str, agent_names: set[str]) -> None:
        self._llm = llm.with_structured_output(route_decision_model(agent_names), method="json_schema")
        self._prompt_template = prompt_template

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
        try:
            next_agent = self._llm.invoke(messages).next
        except Exception:
            logger.exception("[supervisor] routing call failed — defaulting to FINISH")
            next_agent = FINISH
        logger.info("[supervisor] turns=%d → next=%s", state.get("turns", 0), next_agent)

        return {"next": next_agent}
