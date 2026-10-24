"""
orchestrator/graph.py — assembles and compiles the LangGraph multi-agent graph.

PIPELINE OVERVIEW
-----------------
Every user message travels through this sequence:

  START
    │
    ▼
  normaliser          ← Step 0: LLM cleans the message (structured output)
    │
    ▼
  supervisor          ← Step 1: LLM picks the next agent or FINISH (structured output)
    │
    ├──► conversational_agent ──┐
    ├──► property_agent        ─┤  ← Step 2: specialist agent runs, turns += 1
    ├──► finance_agent         ─┘
    │         │
    │         └──► supervisor   (loop back — supervisor reassesses after each agent)
    │
    └──► synthesiser            ← Step 3: LLM combines all outputs into one answer
          │
          ├──► END
          │
          └──► confirmation     ← Step 4: an agent left a pending item → interrupt()
                 │                  (paused; the user's next message resumes here)
                 ├──► synthesiser   (confirmed / cancelled / re-asked)
                 ├──► <agent>       (correction — re-checked by that agent)
                 └──► normaliser    (unrelated — handled as a fresh turn)

CONFIRMATIONS (human-in-the-loop)
---------------------------------
When an agent leaves a pending item (BaseAgent.pending_summary), its node sets
state["awaiting"].  After the synthesiser has shown the agent's question, the
confirmation node calls interrupt(): the run stops and the checkpointer saves
it under thread_id = session_id.  ask()/stream_tokens() see the paused thread
on the next message and resume it with Command(resume=<message>) instead of
starting a new run.  Threads are deleted once a run finishes without pausing,
so the checkpoint database only ever holds paused conversations.

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
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessageChunk, RemoveMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

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
from .nodes import AgentNode, ConfirmationNode, NormaliserNode, SupervisorNode, SynthesiserNode
from .state import MAX_AGENT_TURNS, GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routers — conditional edges out of the synthesiser and confirmation nodes
# ---------------------------------------------------------------------------

def _route_after_synthesiser(state: GraphState) -> str:
    """Pause for a confirmation when an agent left a pending item; else finish."""
    return "confirmation" if state.get("awaiting") else END


def _make_confirmation_router(agent_names: set[str]):
    """Follow the confirmation node's state["next"]; anything unknown ends the run."""
    targets = agent_names | {"synthesiser", "normaliser"}

    def _route(state: GraphState) -> str:
        next_node = state.get("next", END)
        return next_node if next_node in targets else END

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
    1. Create the shared LLM (normaliser, supervisor, synthesiser, confirmation).
    2. Load enabled agents from agents.json via agent_registry.
    3. Build the supervisor prompt with the live agent list injected;
       load the normaliser, synthesiser and confirmation prompts.
    4. Create node functions via factories in nodes.py.
    5. Wire nodes and edges into a StateGraph.
    6. Compile the graph with a checkpointer (SQLite at CHECKPOINT_DB_FILE
       unless one is passed in), which is what lets a run pause and resume.
    """

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None) -> None:

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
        confirmation_prompt = load_prompt(Config.CONFIRMATION_PROMPT_FILE)

        # -----------------------------------------------------------------
        # 4. Instantiate node objects
        #    Each class closes over its dependencies (llm, prompt, agent
        #    instance) and is callable as a plain function(state) -> dict.
        # -----------------------------------------------------------------
        normaliser_node = NormaliserNode(llm, normaliser_prompt)
        supervisor_node = SupervisorNode(llm, supervisor_prompt, agent_names)
        synthesiser_node = SynthesiserNode(llm, synthesiser_prompt)
        confirmation_node = ConfirmationNode(llm, confirmation_prompt, agents)
        self._agents = {a.name: a for a in agents}
        # Agent nodes are created in the loop below (one per enabled agent)

        # -----------------------------------------------------------------
        # 5. Wire the StateGraph
        # -----------------------------------------------------------------
        builder = StateGraph(GraphState)

        # Fixed nodes — always present regardless of agents.json contents
        builder.add_node("normaliser", normaliser_node)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("synthesiser", synthesiser_node)
        builder.add_node("confirmation", confirmation_node)

        # Dynamic agent nodes — one per enabled entry in agents.json
        for agent in agents:
            node_fn = AgentNode(agent.name, agent.instance, owner_only=agent.owner_only)
            builder.add_node(agent.name, node_fn)
            # Every agent loops back to the supervisor after it finishes
            builder.add_edge(agent.name, "supervisor")

        # Entry point: a fresh turn always starts at the normaliser
        builder.add_edge(START, "normaliser")
        builder.add_edge("normaliser", "supervisor")

        # Conditional edges — router functions decide the target.
        # route_map tells LangGraph every possible return value of _route()
        # and which node name it maps to.
        route_map = {name: name for name in agent_names}
        route_map["synthesiser"] = "synthesiser"
        builder.add_conditional_edges("supervisor", _make_router(agent_names), route_map)

        # After answering: finish, or pause for a pending item's confirmation
        builder.add_conditional_edges(
            "synthesiser", _route_after_synthesiser, {"confirmation": "confirmation", END: END}
        )
        builder.add_conditional_edges(
            "confirmation",
            _make_confirmation_router(agent_names),
            {**route_map, "normaliser": "normaliser", END: END},
        )

        # -----------------------------------------------------------------
        # 6. Compile — LangGraph validates the graph topology here
        # -----------------------------------------------------------------
        self._checkpointer = checkpointer or self._sqlite_checkpointer()
        self._graph = builder.compile(checkpointer=self._checkpointer)
        logger.info("[MultiAgentGraph] compiled with agents: %s", sorted(agent_names))

    @staticmethod
    def _sqlite_checkpointer() -> SqliteSaver:
        path = Path(Config.CHECKPOINT_DB_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        # SqliteSaver serialises access with its own lock, so one shared connection is safe
        return SqliteSaver(sqlite3.connect(path, check_same_thread=False))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_state(user_input: str, session_id: str, is_owner: bool) -> GraphState:
        """Fresh state for one user turn: everything empty, turns at zero."""
        return {
            "user_input": user_input,
            "session_id": session_id,
            # Clears messages left by an earlier run on this thread (add_messages reducer)
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "agent_outputs": {},
            "next": "",
            "turns": 0,
            "is_owner": is_owner,
            "raw_input": user_input,
            "direct": False,
            "awaiting": None,
        }

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _pending_interrupt(self, thread_id: str) -> dict | None:
        """The payload of the interrupt this thread is paused on, if any."""
        interrupts = self._graph.get_state(self._config(thread_id)).interrupts
        return interrupts[0].value if interrupts else None

    def _start(self, user_input: str, session_id: str, is_owner: bool) -> tuple[Any, dict]:
        """(graph input, config) for this message: resume a paused thread, or start fresh."""
        paused = self._pending_interrupt(session_id)
        if paused is not None and not self._still_pending(paused.get("agent"), session_id):
            # Resolved some other way while paused — this message starts a new turn
            logger.info("[MultiAgentGraph] %s: pending item on %s is gone — fresh turn",
                        session_id, paused.get("agent"))
            paused = None
        if paused is not None:
            if paused.get("owner_only") and not is_owner:
                # Someone else in the chat can't answer the owner's question — run their
                # message on a side thread and leave the owner's confirmation paused.
                logger.info("[MultiAgentGraph] %s paused on owner-only %s — non-owner turn on side thread",
                            session_id, paused.get("agent"))
                thread_id = f"{session_id}#guest"
                self._checkpointer.delete_thread(thread_id)
                return self._initial_state(user_input, session_id, is_owner), self._config(thread_id)
            logger.info("[MultiAgentGraph] resuming %s (pending on %s)", session_id, paused.get("agent"))
            return Command(resume=user_input, update={"is_owner": is_owner}), self._config(session_id)
        # Nothing paused: drop any leftovers from a run that crashed mid-way
        self._checkpointer.delete_thread(session_id)
        return self._initial_state(user_input, session_id, is_owner), self._config(session_id)

    def _still_pending(self, agent_name: str | None, session_id: str) -> bool:
        entry = self._agents.get(agent_name)
        if entry is None:
            return False
        try:
            return entry.instance.pending_summary(session_id) is not None
        except Exception:
            logger.exception("[MultiAgentGraph] pending_summary failed for %s — not resuming", agent_name)
            return False

    def _finish(self, config: dict) -> None:
        """Keep a thread only while it is paused for a confirmation."""
        thread_id = config["configurable"]["thread_id"]
        if self._pending_interrupt(thread_id) is None:
            self._checkpointer.delete_thread(thread_id)

    def stream(self, user_input: str, session_id: str = "default", *, is_owner: bool = False):
        """
        Run the pipeline (or resume a paused one) and yield LangGraph update-event dicts.

        Each yielded event is {"node_name": {state_delta}} — one dict per
        node execution.  Used internally by ask() to collect the final state.
        """
        graph_input, config = self._start(user_input, session_id, is_owner)
        try:
            yield from self._graph.stream(graph_input, config, stream_mode="updates")
        finally:
            self._finish(config)

    def stream_tokens(self, user_input: str, session_id: str = "default", *, is_owner: bool = False):
        """
        Yield text tokens from the synthesiser LLM as they are generated.
        """
        graph_input, config = self._start(user_input, session_id, is_owner)

        # stream_mode="messages" yields (message, metadata) tuples as they happen:
        # AIMessageChunk tokens while the synthesiser LLM writes, plus the complete
        # AIMessage each node returns.  metadata["langgraph_node"] names the node.
        # The synthesiser can run more than once per call (e.g. confirm, then the
        # next question), so "already streamed" is tracked per synthesiser run.
        streamed_runs: set[str] = set()
        try:
            for chunk, metadata in self._graph.stream(graph_input, config, stream_mode="messages"):
                if metadata.get("langgraph_node") != "synthesiser":
                    continue
                run = metadata.get("langgraph_checkpoint_ns", "")
                # The final message repeats the streamed tokens — only use it when
                # nothing was streamed (single-agent passthrough makes no LLM call).
                if isinstance(chunk, AIMessageChunk):
                    streamed_runs.add(run)
                elif run in streamed_runs:
                    continue
                # .text skips thinking / tool-call blocks (empty for those chunks)
                if chunk.text:
                    yield chunk.text
        finally:
            self._finish(config)

    def ask(self, user_input: str, session_id: str = "default", *, is_owner: bool = False) -> str:
        """
        Blocking call — runs the pipeline (or resumes it) and returns the synthesiser's answer.

        The synthesiser stores its answer in agent_outputs["synthesiser"]; the
        last one written during this call is the reply (a run that pauses for
        a confirmation has already answered with the agent's question).
        """
        answer = ""
        for event in self.stream(user_input, session_id=session_id, is_owner=is_owner):
            update = event.get("synthesiser")
            if isinstance(update, dict):
                answer = update.get("agent_outputs", {}).get("synthesiser", answer)

        if not answer:
            logger.warning("[MultiAgentGraph] no synthesiser output in final state")
        return answer
