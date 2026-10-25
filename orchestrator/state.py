"""
orchestrator/state.py — shared state that flows through every node in the pipeline.

HOW STATE WORKS IN LANGGRAPH
-----------------------------
LangGraph passes a single state dict through every node.  Each node receives
the full current state and returns only the keys it changed.  LangGraph merges
those changes back automatically before the next node runs.

TURN LIFECYCLE
--------------
The router writes the turn's `plan` (ordered agent names).  After the router
and after each agent, graph._route_plan sends the flow to the first planned
agent that has not answered yet, or to the synthesiser when none is left.
Every agent node increments `turns`; at MAX_AGENT_TURNS the flow goes
straight to the synthesiser whatever the plan says.

  User input
      │
      ▼
  router ──► agent A (turns += 1) ──► agent B ... ──► synthesiser
"""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Hard cap on how many agent calls a single user turn may trigger.
MAX_AGENT_TURNS = 6


class GraphState(TypedDict):
    """
    The single state object that travels through the entire graph.

    Every node reads from this and returns a partial dict of updated keys.

    Fields
    ------
    user_input : str
        The user's question as the agents see it.  Set to the raw message at
        graph entry, then replaced by the router's cleaned text (or, on a
        pending correction, the text plus the pending summary).

    session_id : str
        Identifies the conversation so per-agent session history is kept
        separate across turns.  Passed through to each agent's ask() call.

    messages : list[AnyMessage]
        Append-only message log for the full turn.  Uses LangGraph's
        add_messages reducer so concurrent writes are merged safely without
        overwriting each other.

    agent_outputs : dict[str, str]
        Maps agent name → its response text this turn.  An agent with an
        entry here has run; the synthesiser merges them into the answer.

    plan : list[str]
        Agents to run this turn, in order — written by the router (or by the
        confirmation node for a correction: just that agent).

    next : str
        Where the confirmation node sends the flow after a resume: an agent
        name, "synthesiser", "router" or END.

    turns : int
        Running count of agent-node invocations this turn.
        Incremented by every agent node; at MAX_AGENT_TURNS the plan is cut
        short and the synthesiser answers.

    is_owner : bool
        True when the caller is the bot's owner.  Agents flagged
        "owner_only" in agents.json refuse to run when this is False.

    raw_input : str
        The user's message exactly as received, before any normalisation.

    awaiting : str | None
        Name of an agent that left a pending item needing the user's yes/no
        (set by its agent node).  After the synthesiser answers, the graph
        goes to the confirmation node, which pauses with interrupt() until the
        user's next message resumes it.  None when nothing is pending.
    """

    user_input: str
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_outputs: dict[str, str]
    plan: list[str]
    next: str
    turns: int
    is_owner: bool
    raw_input: str
    awaiting: str | None
