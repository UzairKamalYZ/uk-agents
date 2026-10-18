"""
orchestrator/graph.py — assembles and compiles the LangGraph multi-agent graph.

PIPELINE OVERVIEW
-----------------
Every user message travels through this sequence:

  START
    │
    ▼
  supervisor          ← Step 1: LLM decides which agent to call next (or FINISH)
    │
    ├──► conversational_agent ──┐
    ├──► property_agent        ─┤  ← Step 2: specialist agent runs, turns += 1
    ├──► finance_agent         ─┘
    │         │
    │         └──► supervisor   (loop back — supervisor reassesses after each agent)
    │
    └──► synthesiser            ← Step 3: LLM combines all outputs into one answer
          │
          ▼
         END

TURN CAP
--------
The router enforces MAX_AGENT_TURNS (defined in state.py).  Once that many
agent calls have been made in a single user turn, the router bypasses the
supervisor and sends the flow straight to the synthesiser regardless of what
the supervisor LLM said.  This prevents runaway loops.

ADDING A NEW AGENT
------------------
Edit agents.json (project root) only — no Python changes needed here.
Set "enabled": true and provide a "class" dotted path; graph.py will pick
it up automatically on the next startup.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from core.src.config.config import Config
from core.src.model.llm_factory import create_llm
from core.src.utils import load_prompt

# Register LangSmith env vars so traces from the orchestrator appear in the
# same project as the individual agent traces (set with setdefault so a value
# already exported by the shell or llm_model_graph.py is never overwritten).
os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)

from .agent_registry import load_agents
from .nodes import AgentNode, SupervisorNode, SynthesiserNode
from .state import MAX_AGENT_TURNS, GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router — conditional edge out of the supervisor node
# ---------------------------------------------------------------------------

def _make_router(agent_names: set[str]):
    """
    Returns the routing function used as a conditional edge from the supervisor.

    Called by LangGraph after every supervisor_node execution.  It reads the
    current state and returns the NAME of the next node to visit.

    Decision logic (in priority order):
      1. If turns >= MAX_AGENT_TURNS  →  force "synthesiser" (loop guard).
      2. If state["next"] is a known agent name  →  route to that agent.
      3. Otherwise (supervisor said FINISH or returned garbage)  →  "synthesiser".

    Parameters
    ----------
    agent_names : set of enabled agent names loaded from agents.json.
                  Used to validate the supervisor's routing decision.
    """

    def _route(state: GraphState) -> str:
        # --- Guard: too many turns → skip supervisor, go straight to synthesis ---
        if state.get("turns", 0) >= MAX_AGENT_TURNS:
            logger.warning(
                "[router] turn cap reached (%d/%d) — forcing synthesiser",
                state["turns"],
                MAX_AGENT_TURNS,
            )
            return "synthesiser"

        # --- Normal path: honour the supervisor's routing decision ---
        next_node = state.get("next", "FINISH")
        already_ran = set(state.get("agent_outputs", {}).keys())

        # Block re-routing to an agent that already produced output this turn.
        # Prevents smaller models from looping on the same agent indefinitely.
        if next_node in agent_names and next_node in already_ran:
            logger.warning(
                "[router] turns=%d — supervisor re-routed to %r which already ran; forcing synthesiser",
                state.get("turns", 0), next_node,
            )
            return "synthesiser"

        logger.info("[router] turns=%d → routing to %r", state.get("turns", 0), next_node)
        if next_node in agent_names:
            return next_node

        return "synthesiser"

    return _route


# ---------------------------------------------------------------------------
# MultiAgentGraph
# ---------------------------------------------------------------------------

class MultiAgentGraph:
    """
    Builds, wires, and compiles the full LangGraph supervisor multi-agent graph.

    Construction happens once at startup (__init__).  After that, stream() and
    ask() can be called repeatedly without rebuilding the graph.

    Build order inside __init__
    ---------------------------
    1. Create the shared LLM (used by supervisor and synthesiser).
    2. Load enabled agents from agents.json via agent_registry.
    3. Build the supervisor prompt with the live agent list injected.
    4. Create node functions via factories in nodes.py.
    5. Wire nodes and edges into a StateGraph.
    6. Compile the graph.
    """

    def __init__(self) -> None:

        # -----------------------------------------------------------------
        # 1. Shared LLM — same model drives both supervisor and synthesiser
        # -----------------------------------------------------------------
        llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)

        # -----------------------------------------------------------------
        # 2. Load agents — reads agents.json, skips disabled entries,
        #    and returns live instantiated agent objects
        # -----------------------------------------------------------------
        agents = load_agents()
        agent_names = {a.name for a in agents}   # used by router + route_map

        # -----------------------------------------------------------------
        # 3. Build the agent description block injected into the supervisor
        #    prompt so the LLM knows which agents exist and what they do
        # -----------------------------------------------------------------
        agents_block = "\n".join(
            f"- {a.name}: {a.description}" for a in agents
        )
        supervisor_prompt = (
            load_prompt(Config.SUPERVISOR_PROMPT_FILE)
            .replace("{agents}", agents_block)   # fills the {agents} placeholder
        )
        synthesiser_prompt = load_prompt(Config.SYNTHESISER_PROMPT_FILE)

        # -----------------------------------------------------------------
        # 4. Instantiate node objects
        #    Each class closes over its dependencies (llm, prompt, agent
        #    instance) and is callable as a plain function(state) -> dict.
        # -----------------------------------------------------------------
        supervisor_node = SupervisorNode(llm, supervisor_prompt, agent_names)
        synthesiser_node = SynthesiserNode(llm, synthesiser_prompt)
        # Agent nodes are created in the loop below (one per enabled agent)

        # -----------------------------------------------------------------
        # 5. Wire the StateGraph
        # -----------------------------------------------------------------
        builder = StateGraph(GraphState)

        # Fixed nodes — always present regardless of agents.json contents
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("synthesiser", synthesiser_node)

        # Dynamic agent nodes — one per enabled entry in agents.json
        for agent in agents:
            node_fn = AgentNode(agent.name, agent.instance, owner_only=agent.owner_only)
            builder.add_node(agent.name, node_fn)
            # Every agent loops back to the supervisor after it finishes
            builder.add_edge(agent.name, "supervisor")

        # Entry point: graph always starts at the supervisor
        builder.add_edge(START, "supervisor")

        # Conditional edge from supervisor — router function decides the target.
        # route_map tells LangGraph every possible return value of _route()
        # and which node name it maps to.
        route_map = {name: name for name in agent_names}
        route_map["synthesiser"] = "synthesiser"
        builder.add_conditional_edges("supervisor", _make_router(agent_names), route_map)

        # Synthesiser is terminal — no further routing after it finishes
        builder.add_edge("synthesiser", END)

        # -----------------------------------------------------------------
        # 6. Compile — LangGraph validates the graph topology here
        # -----------------------------------------------------------------
        self._graph = builder.compile()
        logger.info("[MultiAgentGraph] compiled with agents: %s", sorted(agent_names))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_state(user_input: str, session_id: str, is_owner: bool) -> GraphState:
        """Fresh state for one user turn: everything empty, turns at zero."""
        return {
            "user_input": user_input,
            "session_id": session_id,
            "messages": [],
            "agent_outputs": {},
            "next": "",
            "turns": 0,
            "is_owner": is_owner,
            "raw_input": user_input,
            "direct": False,
        }

    def stream(self, user_input: str, session_id: str = "default", *, is_owner: bool = False):
        """
        Run the full pipeline and yield LangGraph update-event dicts.

        Each yielded event is {"node_name": {state_delta}} — one dict per
        node execution.  Used internally by ask() to collect the final state.
        """
        initial_state = self._initial_state(user_input, session_id, is_owner)
        yield from self._graph.stream(initial_state, stream_mode="updates")

    def stream_tokens(self, user_input: str, session_id: str = "default", *, is_owner: bool = False):
        """
        Yield text tokens from the synthesiser LLM as they are generated.
        """
        initial_state = self._initial_state(user_input, session_id, is_owner)

        # stream_mode="messages" yields (AIMessageChunk, metadata) tuples.
        # metadata["langgraph_node"] tells us which graph node produced the chunk.
        for chunk, metadata in self._graph.invoke(
            initial_state, stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "synthesiser":
                # .text skips thinking / tool-call blocks (empty for those chunks)
                if chunk.text:
                    yield chunk.text

    def ask(self, user_input: str, session_id: str = "default", *, is_owner: bool = False) -> str:
        """
        Blocking call — runs the full pipeline and returns the synthesiser's answer.

        Internally calls stream() and collects all events.  The synthesiser
        stores its answer in agent_outputs["synthesiser"], which this method
        extracts and returns as a plain string.
        """
        final_state: dict[str, Any] = {}
        for event in self.stream(user_input, session_id=session_id, is_owner=is_owner):
            final_state.update(event)   # accumulate all node updates

        # Pull agent_outputs out of each node's update dict and merge them
        outputs = {}
        for node_update in final_state.values():
            if isinstance(node_update, dict):
                outputs.update(node_update.get("agent_outputs", {}))

        answer = outputs.get("synthesiser", "")
        if not answer:
            logger.warning("[MultiAgentGraph] no synthesiser output in final state")
        return answer
