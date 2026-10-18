"""
orchestrator/state.py — shared state that flows through every node in the pipeline.

HOW STATE WORKS IN LANGGRAPH
-----------------------------
LangGraph passes a single state dict through every node.  Each node receives
the full current state and returns only the keys it changed.  LangGraph merges
those changes back automatically before the next node runs.

TURN LIFECYCLE
--------------
Each time the supervisor routes to an agent, that agent increments `turns` by 1.
When `turns` reaches MAX_AGENT_TURNS the router ignores the supervisor's decision
and forces the flow straight to the synthesiser, preventing infinite loops.

  User input
      │
      ▼
  supervisor ──► agent_node (turns += 1) ──► supervisor ...
      │                                           │
      │          (turns >= MAX_AGENT_TURNS)       │
      └──────────────── synthesiser ◄─────────────┘
"""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Hard cap on how many agent calls a single user turn may trigger.
# Prevents runaway loops if the supervisor keeps re-routing to agents.
MAX_AGENT_TURNS = 6


class GraphState(TypedDict):
    """
    The single state object that travels through the entire graph.

    Every node reads from this and returns a partial dict of updated keys.

    Fields
    ------
    user_input : str
        The raw question submitted by the user.  Set once at graph entry;
        never modified after that.

    session_id : str
        Identifies the conversation so per-agent session history is kept
        separate across turns.  Passed through to each agent's ask() call.

    messages : list[AnyMessage]
        Append-only message log for the full turn.  Uses LangGraph's
        add_messages reducer so concurrent writes are merged safely without
        overwriting each other.

    agent_outputs : dict[str, str]
        Maps agent name → its latest response text.
        The supervisor reads this dict to decide what to do next.
        The synthesiser reads it to build the final answer.

    next : str
        Routing token written by the supervisor node after each decision.
        Values: any agent name registered in agents.json, or "FINISH".
        The conditional edge in graph.py reads this to pick the next node.

    turns : int
        Running count of agent-node invocations this turn.
        Incremented by every agent node.  Checked by the router to enforce
        the MAX_AGENT_TURNS cap.

    is_owner : bool
        True when the caller is the bot's owner.  Agents flagged
        "owner_only" in agents.json refuse to run when this is False.

    raw_input : str
        The user's message exactly as received, before any normalisation.

    direct : bool
        True when a pre-supervisor step has already decided the routing
        (bypassing the supervisor LLM).  False at graph entry.
    """

    user_input: str
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_outputs: dict[str, str]
    next: str
    turns: int
    is_owner: bool
    raw_input: str
    direct: bool
