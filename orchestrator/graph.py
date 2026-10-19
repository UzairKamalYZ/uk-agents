"""
orchestrator/graph.py — assembles and compiles the LangGraph multi-agent graph.

PIPELINE OVERVIEW
-----------------
Every user message travels through this sequence:

  START
    │
    ▼
  normaliser          ← Step 0: LLM cleans the message; resolves any pending item
    │
    ├──► synthesiser            (pending confirmed / cancelled — answer is ready)
    ├──► <agent>                (pending corrected — "direct", skips supervisor routing)
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

NORMALISER
----------
The supervisor sees no chat history, so a bare "yes" to "save this expense?"
cannot be routed by it.  The normaliser asks each agent for a pending item
(BaseAgent.pending_summary) and classifies the message against it in the same
LLM call that cleans the text.  On a correction it sets state["direct"], and
the supervisor router then sends the flow straight to the synthesiser once
that agent has answered.

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
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from core.src.config.config import Config
from core.src.model.llm_factory import create_llm
from core.src.model.session_manager import SessionManager
from core.src.utils import load_prompt

# Register LangSmith env vars so traces from the orchestrator appear in the
# same project as the individual agent traces (set with setdefault so a value
# already exported by the shell or llm_model_graph.py is never overwritten).
os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)

from .agent_registry import AgentEntry, load_agents
from .conversation import ConversationMemory
from .nodes import AgentNode, NormaliserNode, SupervisorNode, SynthesiserNode
from .state import MAX_AGENT_TURNS, GraphState
from .tracing import new_request_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router — conditional edge out of the normaliser node
# ---------------------------------------------------------------------------

def _make_normaliser_router(agent_names: set[str]):
    """
    Returns the routing function used as a conditional edge from the normaliser.

      - state["next"] is a known agent name  →  that agent (pending correction)
      - state["next"] == "synthesiser"       →  synthesiser (confirm / cancel)
      - anything else                        →  "supervisor" (normal path)
    """

    def _route(state: GraphState) -> str:
        next_node = state.get("next", "")
        if next_node in agent_names or next_node == "synthesiser":
            logger.info("[normaliser-router] → %r (skipping supervisor)", next_node)
            return next_node
        return "supervisor"

    return _route


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
      2. If state["direct"] and an agent has answered  →  "synthesiser"
         (the normaliser already chose the agent; nothing left to route).
      3. If state["next"] is a known agent name  →  route to that agent.
      4. Otherwise (supervisor said FINISH or returned garbage)  →  "synthesiser".

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

        # --- Direct path: normaliser picked the agent, which has now answered ---
        if state.get("direct") and state.get("agent_outputs"):
            logger.info("[router] direct turn complete → synthesiser")
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
    1. Create the shared LLM (used by normaliser, supervisor and synthesiser).
    2. Load enabled agents from agents.json via agent_registry.
    3. Build the supervisor prompt with the live agent list injected;
       load the normaliser and synthesiser prompts.
    4. Create node functions via factories in nodes.py.
    5. Wire nodes and edges into a StateGraph.
    6. Compile the graph.
    """

    def __init__(self, history_store: SessionManager | None = None) -> None:

        # Recent turns per session, so follow-ups ("yes", "do that too") keep their context
        self._memory = ConversationMemory(history_store or SessionManager(Config.SESSION_DB_FILE))

        # -----------------------------------------------------------------
        # 1. Shared LLM — same model drives normaliser, supervisor and synthesiser
        # -----------------------------------------------------------------
        llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)

        # -----------------------------------------------------------------
        # 2. Load agents — reads agents.json, skips disabled entries,
        #    and returns live instantiated agent objects
        # -----------------------------------------------------------------
        agents = load_agents()
        agent_names = {a.name for a in agents}   # used by router + route_map
        self._entries = {a.name: a for a in agents}   # name → registry entry (labels, remarks)

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
        normaliser_prompt = load_prompt(Config.NORMALISER_PROMPT_FILE)

        # -----------------------------------------------------------------
        # 4. Instantiate node objects
        #    Each class closes over its dependencies (llm, prompt, agent
        #    instance) and is callable as a plain function(state) -> dict.
        # -----------------------------------------------------------------
        normaliser_node = NormaliserNode(llm, normaliser_prompt, agents)
        supervisor_node = SupervisorNode(llm, supervisor_prompt, agent_names)
        synthesiser_node = SynthesiserNode(llm, synthesiser_prompt)
        # Agent nodes are created in the loop below (one per enabled agent)

        # -----------------------------------------------------------------
        # 5. Wire the StateGraph
        # -----------------------------------------------------------------
        builder = StateGraph(GraphState)

        # Fixed nodes — always present regardless of agents.json contents
        builder.add_node("normaliser", normaliser_node)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("synthesiser", synthesiser_node)

        # Dynamic agent nodes — one per enabled entry in agents.json
        for agent in agents:
            node_fn = AgentNode(agent.name, agent.instance, owner_only=agent.owner_only)
            builder.add_node(agent.name, node_fn)
            # Every agent loops back to the supervisor after it finishes
            builder.add_edge(agent.name, "supervisor")

        # Entry point: graph always starts at the normaliser
        builder.add_edge(START, "normaliser")

        # Conditional edges — router functions decide the target.
        # route_map tells LangGraph every possible return value of _route()
        # and which node name it maps to.
        route_map = {name: name for name in agent_names}
        route_map["synthesiser"] = "synthesiser"
        builder.add_conditional_edges(
            "normaliser",
            _make_normaliser_router(agent_names),
            {**route_map, "supervisor": "supervisor"},
        )
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

    def _initial_state(self, user_input: str, session_id: str, is_owner: bool) -> GraphState:
        """Fresh state for one user turn: a new request_id, the session's recent conversation,
        everything else empty and turns at zero."""
        return {
            "request_id": new_request_id(),
            "history": self._memory.recent(session_id),
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
        yield from self._graph.stream(
            initial_state, stream_mode="updates", config=self._run_config(initial_state)
        )

    @staticmethod
    def _run_config(state: GraphState) -> dict[str, Any]:
        """LangSmith run name, tags and metadata for one turn — every node run inherits them."""
        return {
            "run_name": "orchestrator.graph",
            "tags": ["orchestrator", f"session:{state['session_id']}"],
            "metadata": {
                "request_id": state["request_id"],
                "session_id": state["session_id"],
                "is_owner": state["is_owner"],
            },
        }

    def agent_entry(self, name: str) -> AgentEntry | None:
        """The registry entry for an agent name (display name, remarks…), or None."""
        return self._entries.get(name)

    def stream_tokens(self, user_input: str, session_id: str = "default", *, is_owner: bool = False):
        """
        Yield text tokens from the synthesiser LLM as they are generated.
        """
        initial_state = self._initial_state(user_input, session_id, is_owner)

        # stream_mode="messages" yields (AIMessageChunk, metadata) tuples.
        # metadata["langgraph_node"] tells us which graph node produced the chunk.
        for chunk, metadata in self._graph.invoke(
            initial_state, stream_mode="messages", config=self._run_config(initial_state)
        ):
            if metadata.get("langgraph_node") == "synthesiser":
                # .text skips thinking / tool-call blocks (empty for those chunks)
                if chunk.text:
                    yield chunk.text

    def ask(
        self,
        user_input: str,
        session_id: str = "default",
        *,
        is_owner: bool = False,
        on_route: Callable[[AgentEntry], None] | None = None,
    ) -> str:
        """
        Blocking call — runs the full pipeline and returns the synthesiser's answer.

        Internally calls stream() and collects all events.  The synthesiser
        stores its answer in agent_outputs["synthesiser"], which this method
        extracts and returns as a plain string.

        on_route, if given, is called with the AgentEntry of each agent as soon
        as it is picked — before that agent runs — so a client can tell the user
        who is working on their query.  Exceptions from on_route are logged and
        never interrupt the request.
        """
        final_state: dict[str, Any] = {}
        announced: set[str] = set()
        for event in self.stream(user_input, session_id=session_id, is_owner=is_owner):
            final_state.update(event)   # accumulate all node updates
            if on_route is not None:
                for name in self._routed_agents(event):
                    if name not in announced:
                        announced.add(name)
                        self._notify(on_route, name)

        # Pull agent_outputs out of each node's update dict and merge them
        outputs = {}
        for node_update in final_state.values():
            if isinstance(node_update, dict):
                outputs.update(node_update.get("agent_outputs", {}))

        answer = outputs.get("synthesiser", "")
        if not answer:
            logger.warning("[MultiAgentGraph] no synthesiser output in final state")
        self._memory.record(session_id, user_input, answer, [n for n in outputs if n != "synthesiser"])
        return answer

    def _routed_agents(self, event: dict[str, Any]) -> list[str]:
        """Agents picked by this update event: the supervisor's or normaliser's "next",
        or the agent whose pending item the normaliser just resolved."""
        names = []
        for node in ("normaliser", "supervisor"):
            update = event.get(node)
            if not isinstance(update, dict):
                continue
            nxt = update.get("next")
            if nxt in self._entries:
                names.append(nxt)
            names.extend(n for n in update.get("agent_outputs", {}) if n in self._entries)
        return names

    def _notify(self, on_route: Callable[[AgentEntry], None], name: str) -> None:
        try:
            on_route(self._entries[name])
        except Exception:
            logger.exception("[MultiAgentGraph] on_route callback failed for %s", name)
