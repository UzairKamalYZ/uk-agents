"""
orchestrator/nodes/synthesiser.py — Synthesiser node class.

Responsibility
--------------
Terminal node in the pipeline.  Receives all agent outputs collected during
the current turn and asks the LLM to merge them into one clean, user-facing
answer.  The result is stored in state["agent_outputs"]["synthesiser"] so
that agent_interface.py can extract it after the graph finishes.

Optimisation
------------
When only one agent ran, no LLM call is made — the single output is passed
through directly.  This avoids latency and prevents smaller models from
blindly echoing the agent text.

Usage
-----
    node = SynthesiserNode(llm, system_prompt)
    graph_builder.add_node("synthesiser", node)   # callable instance
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..state import GraphState

logger = logging.getLogger(__name__)


class SynthesiserNode:
    """
    LangGraph node that merges all agent outputs into one final answer.

    Parameters
    ----------
    llm           : chat model used to write the synthesised response
    system_prompt : text loaded from SYNTHESISER_PROMPT_FILE
    """

    def __init__(self, llm: BaseChatModel, system_prompt: str) -> None:
        self._llm = llm
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # LangGraph interface — called as synthesiser_node(state)
    # ------------------------------------------------------------------

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        agent_outputs = state.get("agent_outputs", {})

        if len(agent_outputs) == 1:
            return self._passthrough(agent_outputs)

        return await self._synthesise(agent_outputs, state["user_input"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passthrough(self, agent_outputs: dict[str, str]) -> dict[str, Any]:
        """Single-agent shortcut — no LLM call needed."""
        answer = next(iter(agent_outputs.values()))
        logger.info("[synthesiser] single agent — passthrough, len=%d", len(answer))
        updated = dict(agent_outputs)
        updated["synthesiser"] = answer
        return {
            "agent_outputs": updated,
            "messages": [AIMessage(content=answer, name="synthesiser")],
        }

    async def _synthesise(self, agent_outputs: dict[str, str], user_input: str) -> dict[str, Any]:
        """Multi-agent path — ask the LLM to merge all outputs."""
        outputs_block = "\n\n".join(
            f"### {name}\n{text}" for name, text in agent_outputs.items()
        )
        messages = [
            SystemMessage(content=f"{self._system_prompt}\n\nAGENT OUTPUTS:\n\n{outputs_block}"),
            HumanMessage(content=user_input),
        ]
        response = await self._llm.ainvoke(messages)
        answer = response.text

        logger.info("[synthesiser] final answer len=%d", len(answer))
        updated = dict(agent_outputs)
        updated["synthesiser"] = answer
        return {
            "agent_outputs": updated,
            "messages": [AIMessage(content=answer, name="synthesiser")],
        }
