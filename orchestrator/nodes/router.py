"""
orchestrator/nodes/router.py — Router node class.

Responsibility
--------------
First node of every fresh turn.  ONE structured LLM call does what used to
take a normaliser call plus a supervisor call before and after every agent:

  1. Cleans the raw message (transcription and spelling fixes, spoken numbers
     → digits, relative dates → YYYY-MM-DD) → state["user_input"].
  2. Plans the turn: the ordered list of agents to run → state["plan"].

The graph then runs the plan without asking an LLM again between agents
(graph._route_plan), and the synthesiser merges the outputs.

The reply is a RoutePlan whose `agents` items are constrained to the enabled
agent names.  Duplicates are dropped and the plan is capped at
MAX_AGENT_TURNS.  On any failure (LLM error, refusal) the raw message passes
through with an empty plan, so the synthesiser apologises.

Usage
-----
    node = RouterNode(llm, prompt_template, agents_block, agent_names)
    graph_builder.add_node("router", node)   # callable instance
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

from ..state import MAX_AGENT_TURNS, GraphState

logger = logging.getLogger(__name__)


def route_plan_model(agent_names: set[str]) -> type[BaseModel]:
    """RoutePlan with `agents` limited to the enabled agent names."""
    names = tuple(sorted(agent_names))
    return create_model(
        "RoutePlan",
        text=(str, Field(description="The cleaned message")),
        agents=(
            list[Literal[names]],
            Field(description="Agents to run, in order; each runs once"),
        ),
    )


class RouterNode:
    """
    LangGraph node that cleans the message and plans which agents run.

    Parameters
    ----------
    llm             : chat model used for the routing call
    prompt_template : text loaded from ROUTER_PROMPT_FILE — must contain
                      {agents} and {today} placeholders
    agents_block    : "- name: description" lines for the enabled agents
    agent_names     : enabled agent names — the only values the plan may hold
    today           : date provider, injectable for tests
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: str,
        agents_block: str,
        agent_names: set[str],
        today: Callable[[], date] = date.today,
    ) -> None:
        self._llm = llm.with_structured_output(route_plan_model(agent_names), method="json_schema")
        self._prompt_template = prompt_template.replace("{agents}", agents_block)
        self._today = today

    # ------------------------------------------------------------------
    # LangGraph interface — called as router_node(state)
    # ------------------------------------------------------------------

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        raw = state.get("raw_input") or state["user_input"]
        today = self._today()
        prompt = self._prompt_template.replace("{today}", f"{today.isoformat()} ({today.strftime('%A')})")
        try:
            # The user's text goes in the human message, never the system prompt
            reply = await self._llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=raw)])
        except Exception:
            logger.exception("[router] routing call failed — raw input, no agents")
            return {"user_input": raw, "plan": []}

        plan = list(dict.fromkeys(reply.agents))[:MAX_AGENT_TURNS]
        text = reply.text.strip() or raw
        logger.info("[router] plan=%s", plan)
        return {"user_input": text, "plan": plan}
