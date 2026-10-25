"""
orchestrator/graph.py — assembles and compiles the LangGraph multi-agent graph.

PIPELINE OVERVIEW
-----------------
Every user message travels through this sequence:

  START
    │
    ▼
  router              ← Step 1: ONE LLM call cleans the message and plans the
    │                   agents to run, in order (structured output)
    ├──► agent A ──► agent B ...  ← Step 2: the plan runs; no LLM decides between
    │                               agents (turns += 1 per agent)
    ▼
  synthesiser         ← Step 3: one agent → its answer as is; several → LLM merge
    │
    ├──► END
    │
    └──► confirmation ← Step 4: an agent left a pending item → interrupt()
           │              (paused; the user's next message resumes here)
           ├──► synthesiser   (confirmed / cancelled / re-asked)
           ├──► <agent>       (correction — re-checked by that agent)
           └──► router        (unrelated — handled as a fresh turn)

LLM CALLS PER MESSAGE
---------------------
Orchestration costs one call (router) for a single-agent message, plus one
synthesiser call when several agents answered — the agents' own calls aside.

CONFIRMATIONS (human-in-the-loop)
---------------------------------
When an agent leaves a pending item (BaseAgent.pending_summary), its node sets
state["awaiting"].  After the synthesiser has shown the agent's question, the
confirmation node calls interrupt(): the run stops and the checkpointer saves
it under thread_id = session_id.  ask()/stream_tokens() see the paused thread
on the next message and resume it with Command(resume=<message>) instead of
starting a new run.  Threads are deleted once a run finishes without pausing,
so the checkpoint database only ever holds paused conversations.

CONVERSATION MEMORY
-------------------
The last few user/assistant turns of each session (ConversationMemory, kept in
SESSION_DB_FILE) go into state["history"] for the router, so a follow-up such
as "yes" or "and for last month?" is routed with its context.  Every finished
run (or pause) records its turn.

ANNOUNCING THE AGENT
--------------------
ask()/aask() take on_route: it is called with each agent's AgentEntry just
before that agent runs, so a client can show who is working on the query.

TURN CAP
--------
_route_plan enforces MAX_AGENT_TURNS (defined in state.py): once that many
agent calls have run in one turn the flow goes to the synthesiser.

ADDING A NEW AGENT
------------------
Edit agents.json (project root) only — no Python changes needed here.
Set "enabled": true and provide a "class" dotted path; graph.py will pick
it up automatically on the next startup.
"""
from __future__ import annotations

import asyncio
import logging
import os
import weakref
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Callable

import aiosqlite

from langchain_core.messages import AIMessageChunk, RemoveMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from core.src.aio import BackgroundLoop
from core.src.config.config import Config
from core.src.model.llm_factory import create_llm
from core.src.model.session_manager import HistoryStore, shared_session_store
from core.src.output_guard import StreamScrubber
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
from .nodes import AgentNode, ConfirmationNode, RouterNode, SynthesiserNode
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
    targets = agent_names | {"synthesiser", "router"}

    def _route(state: GraphState) -> str:
        next_node = state.get("next", END)
        return next_node if next_node in targets else END

    return _route


# ---------------------------------------------------------------------------
# Plan follower — conditional edge out of the router and every agent
# ---------------------------------------------------------------------------

def _route_plan(state: GraphState) -> str:
    """The first planned agent that hasn't answered yet, else the synthesiser."""
    if state.get("turns", 0) >= MAX_AGENT_TURNS:
        logger.warning("[plan] turn cap reached (%d/%d) — synthesiser", state["turns"], MAX_AGENT_TURNS)
        return "synthesiser"
    done = state.get("agent_outputs", {})
    for name in state.get("plan", []):
        if name not in done:
            return name
    return "synthesiser"


# ---------------------------------------------------------------------------
# MultiAgentGraph
# ---------------------------------------------------------------------------

class MultiAgentGraph:
    """
    Builds, wires, and compiles the full LangGraph router multi-agent graph.

    Construction happens once at startup (__init__).  After that, stream() and
    ask() can be called repeatedly without rebuilding the graph.

    Build order inside __init__
    ---------------------------
    1. Create the shared LLM (router, synthesiser, confirmation).
    2. Load enabled agents from agents.json via agent_registry.
    3. Build the agent list for the router prompt; load the router,
       synthesiser and confirmation prompts.
    4. Create node functions via factories in nodes.py.
    5. Wire nodes and edges into a StateGraph.
    6. Compile the graph with a checkpointer (SQLite at CHECKPOINT_DB_FILE
       unless one is passed in), which is what lets a run pause and resume.
    """

    def __init__(
        self, checkpointer: BaseCheckpointSaver | None = None, history_store: HistoryStore | None = None
    ) -> None:

        # Recent turns per session, so follow-ups ("yes", "do that too") keep their context
        self._memory = ConversationMemory(history_store or shared_session_store(Config.SESSION_DB_FILE))

        # -----------------------------------------------------------------
        # 1. Shared LLM — same model drives router, synthesiser and confirmation
        # -----------------------------------------------------------------
        llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)

        # -----------------------------------------------------------------
        # 2. Load agents — reads agents.json, skips disabled entries,
        #    and returns live instantiated agent objects
        # -----------------------------------------------------------------
        agents = load_agents()
        agent_names = {a.name for a in agents}   # used by router + route_map

        # -----------------------------------------------------------------
        # 3. Build the agent description block injected into the router
        #    prompt so the LLM knows which agents exist and what they do
        # -----------------------------------------------------------------
        agents_block = "\n".join(
            f"- {a.name}: {a.description}" for a in agents
        )
        router_prompt = load_prompt(Config.ROUTER_PROMPT_FILE)
        synthesiser_prompt = load_prompt(Config.SYNTHESISER_PROMPT_FILE)
        confirmation_prompt = load_prompt(Config.CONFIRMATION_PROMPT_FILE)

        # -----------------------------------------------------------------
        # 4. Instantiate node objects
        #    Each class closes over its dependencies (llm, prompt, agent
        #    instance) and is callable as a plain function(state) -> dict.
        # -----------------------------------------------------------------
        router_node = RouterNode(llm, router_prompt, agents_block, agent_names)
        synthesiser_node = SynthesiserNode(llm, synthesiser_prompt)
        confirmation_node = ConfirmationNode(llm, confirmation_prompt, agents)
        self._agents = {a.name: a for a in agents}
        # Agent nodes are created in the loop below (one per enabled agent)

        # -----------------------------------------------------------------
        # 5. Wire the StateGraph
        # -----------------------------------------------------------------
        builder = StateGraph(GraphState)

        # Fixed nodes — always present regardless of agents.json contents
        builder.add_node("router", router_node)
        builder.add_node("synthesiser", synthesiser_node)
        builder.add_node("confirmation", confirmation_node)

        # Entry point: a fresh turn always starts at the router
        builder.add_edge(START, "router")

        # Conditional edges — route_map tells LangGraph every possible return
        # value of a routing function and which node name it maps to.
        route_map = {name: name for name in agent_names}
        route_map["synthesiser"] = "synthesiser"
        builder.add_conditional_edges("router", _route_plan, route_map)

        # Dynamic agent nodes — one per enabled entry in agents.json; after each
        # one the plan decides what runs next (no LLM call in between)
        for agent in agents:
            node_fn = AgentNode(agent.name, agent.instance)
            builder.add_node(agent.name, node_fn)
            builder.add_conditional_edges(agent.name, _route_plan, route_map)

        # After answering: finish, or pause for a pending item's confirmation
        builder.add_conditional_edges(
            "synthesiser", _route_after_synthesiser, {"confirmation": "confirmation", END: END}
        )
        builder.add_conditional_edges(
            "confirmation",
            _make_confirmation_router(agent_names),
            {**route_map, "router": "router", END: END},
        )

        # -----------------------------------------------------------------
        # 6. Compile — LangGraph validates the graph topology here
        # -----------------------------------------------------------------
        # The graph runs on its own event loop: the aiosqlite checkpointer must
        # stay on one loop, and sync + async callers can then both use it.
        self._runtime = BackgroundLoop("orchestrator-loop")
        self._owns_checkpointer = checkpointer is None
        self._checkpointer = checkpointer or self._runtime.run(self._open_sqlite_checkpointer())
        self._graph = builder.compile(checkpointer=self._checkpointer)
        # One run at a time per session (created and used only on the runtime loop)
        self._session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        logger.info("[MultiAgentGraph] compiled with agents: %s", sorted(agent_names))

    @staticmethod
    async def _open_sqlite_checkpointer() -> AsyncSqliteSaver:
        path = Path(Config.CHECKPOINT_DB_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        return AsyncSqliteSaver(await aiosqlite.connect(path))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_state(user_input: str, session_id: str, history: str = "") -> GraphState:
        """Fresh state for one user turn: the session's recent conversation, everything else
        empty and turns at zero."""
        return {
            "user_input": user_input,
            "session_id": session_id,
            # Clears messages left by an earlier run on this thread (add_messages reducer)
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "agent_outputs": {},
            "plan": [],
            "next": "",
            "turns": 0,
            "raw_input": user_input,
            "awaiting": None,
            "history": history,
        }

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    # ------------------------------------------------------------------
    # Run lifecycle (coroutines — always on the runtime loop)
    # ------------------------------------------------------------------

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = self._session_locks[session_id] = asyncio.Lock()
        return lock

    async def _pending_interrupt(self, thread_id: str) -> dict | None:
        """The payload of the interrupt this thread is paused on, if any."""
        interrupts = (await self._graph.aget_state(self._config(thread_id))).interrupts
        return interrupts[0].value if interrupts else None

    async def _start(self, user_input: str, session_id: str) -> tuple[Any, dict]:
        """(graph input, config) for this message: resume a paused thread, or start fresh."""
        paused = await self._pending_interrupt(session_id)
        if paused is not None and not await asyncio.to_thread(self._still_pending, paused.get("agent"), session_id):
            # Resolved some other way while paused — this message starts a new turn
            logger.info("[MultiAgentGraph] %s: pending item on %s is gone — fresh turn",
                        session_id, paused.get("agent"))
            paused = None
        if paused is not None:
            logger.info("[MultiAgentGraph] resuming %s (pending on %s)", session_id, paused.get("agent"))
            return Command(resume=user_input), self._config(session_id)
        # Nothing paused: drop any leftovers from a run that crashed mid-way
        await self._checkpointer.adelete_thread(session_id)
        history = await asyncio.to_thread(self._memory.recent, session_id)
        return self._initial_state(user_input, session_id, history), self._config(session_id)

    def _still_pending(self, agent_name: str | None, session_id: str) -> bool:
        entry = self._agents.get(agent_name)
        if entry is None:
            return False
        try:
            return entry.instance.pending_summary(session_id) is not None
        except Exception:
            logger.exception("[MultiAgentGraph] pending_summary failed for %s — not resuming", agent_name)
            return False

    async def _record_turn(self, config: dict, user_input: str, answer: str) -> None:
        """Remember this turn for the router, attributed to the agents that answered it."""
        try:
            outputs = (await self._graph.aget_state(config)).values.get("agent_outputs", {})
        except Exception:  # memory is a nicety — never fail the answer over it
            logger.exception("[MultiAgentGraph] could not read the finished run's outputs")
            return
        agents = [name for name in outputs if name != "synthesiser"]
        session_id = config["configurable"]["thread_id"]
        await asyncio.to_thread(self._memory.record, session_id, user_input, answer, agents)

    def _announce(self, on_route: Callable[[AgentEntry], None], name: str) -> None:
        try:
            on_route(self._agents[name])
        except Exception:  # a client callback must never break the request
            logger.exception("[MultiAgentGraph] on_route callback failed for %s", name)

    async def _finish(self, config: dict) -> None:
        """Keep a thread only while it is paused for a confirmation."""
        thread_id = config["configurable"]["thread_id"]
        if await self._pending_interrupt(thread_id) is None:
            await self._checkpointer.adelete_thread(thread_id)

    async def _run_ask(
        self, user_input: str, session_id: str,
        on_route: Callable[[AgentEntry], None] | None = None,
    ) -> str:
        """
        Run the pipeline (or resume it) and return the synthesiser's answer.

        The synthesiser stores its answer in agent_outputs["synthesiser"]; the
        last one written during this run is the reply (a run that pauses for
        a confirmation has already answered with the agent's question).

        on_route is called with each agent's entry just before it runs: the next
        agent in the router's plan, or the one a confirmation hands over to.
        """
        async with self._session_lock(session_id):
            graph_input, config = await self._start(user_input, session_id)
            answer = ""
            plan: list[str] = []
            ran: set[str] = set()
            announced: set[str] = set()
            try:
                async for event in self._graph.astream(graph_input, config, stream_mode="updates"):
                    update = event.get("synthesiser")
                    if isinstance(update, dict):
                        answer = update.get("agent_outputs", {}).get("synthesiser", answer)
                    if on_route is None:
                        continue
                    for node, node_update in event.items():
                        if node == "router" and isinstance(node_update, dict):
                            plan = node_update.get("plan", [])
                        elif node == "confirmation" and isinstance(node_update, dict):
                            plan = [node_update.get("next")]
                        elif node in self._agents:
                            ran.add(node)
                    upcoming = next((n for n in plan if n in self._agents and n not in ran), None)
                    if upcoming is not None and upcoming not in announced:
                        announced.add(upcoming)
                        self._announce(on_route, upcoming)
                await self._record_turn(config, user_input, answer)
            finally:
                await self._finish(config)
        if not answer:
            logger.warning("[MultiAgentGraph] no synthesiser output in final state")
        return answer

    async def _run_tokens(self, user_input: str, session_id: str) -> AsyncIterator[str]:
        """Yield text tokens from the synthesiser LLM as they are generated."""
        async with self._session_lock(session_id):
            graph_input, config = await self._start(user_input, session_id)
            # stream_mode="messages" yields (message, metadata) tuples as they happen:
            # AIMessageChunk tokens while the synthesiser LLM writes, plus the complete
            # AIMessage each node returns.  metadata["langgraph_node"] names the node.
            # The synthesiser can run more than once per call (e.g. confirm, then the
            # next question), so "already streamed" is tracked per synthesiser run.
            streamed_runs: set[str] = set()
            answer: list[str] = []
            # Agent outputs are scrubbed in AgentNode; the merge LLM's tokens are scrubbed here
            guard = StreamScrubber()
            try:
                async for chunk, metadata in self._graph.astream(graph_input, config, stream_mode="messages"):
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
                    if chunk.text and (safe := guard.feed(chunk.text)):
                        answer.append(safe)
                        yield safe
                if rest := guard.flush():
                    answer.append(rest)
                    yield rest
                if guard.found:
                    logger.warning("[MultiAgentGraph] streamed answer contained tool-call markup %s — removed",
                                   sorted(guard.found))
                await self._record_turn(config, user_input, "".join(answer))
            finally:
                await self._finish(config)

    # ------------------------------------------------------------------
    # Public interface — async (REST, Telegram) and sync (cron, scripts)
    # ------------------------------------------------------------------

    async def aask(
        self, user_input: str, session_id: str = "default", *,
        on_route: Callable[[AgentEntry], None] | None = None,
    ) -> str:
        """Answer one message. Messages on the same session are handled one at a time.

        on_route runs on the orchestrator's own loop thread, so it must not block;
        hand any client work back to the client's loop (e.g. run_coroutine_threadsafe).
        """
        return await self._runtime.arun(self._run_ask(user_input, session_id, on_route))

    async def astream_tokens(
        self, user_input: str, session_id: str = "default"
    ) -> AsyncIterator[str]:
        """Async generator of answer tokens as the synthesiser writes them."""
        async for token in self._runtime.aiterate(self._run_tokens(user_input, session_id)):
            yield token

    def ask(
        self, user_input: str, session_id: str = "default", *,
        on_route: Callable[[AgentEntry], None] | None = None,
    ) -> str:
        """Blocking aask() for sync callers."""
        return self._runtime.run(self._run_ask(user_input, session_id, on_route))

    def stream_tokens(self, user_input: str, session_id: str = "default") -> Iterator[str]:
        """Blocking astream_tokens() for sync callers."""
        yield from self._runtime.iterate(self._run_tokens(user_input, session_id))

    def pending_interrupt(self, session_id: str) -> dict | None:
        """What this session is paused on (agent, summary), or None."""
        return self._runtime.run(self._pending_interrupt(session_id))

    def close(self) -> None:
        """Close the checkpointer connection (if this graph opened it) and stop the loop."""
        if self._owns_checkpointer:
            self._runtime.run(self._checkpointer.conn.close())
        self._runtime.stop()
