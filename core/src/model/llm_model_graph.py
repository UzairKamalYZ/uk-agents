import asyncio
import json
import logging
import uuid
from typing import List, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

from langgraph.graph import StateGraph, START, END

from ..config.config import Config
from ..utils import load_prompt
from .embedder import Embedder
from .mcp_tools import close_mcp, _mcp
from .rag_context_manager import RagContextManager
from .session_manager import HistoryStore, InMemoryHistoryStore

logger = logging.getLogger(__name__)


class State(TypedDict):
    user_prompt: str
    context: str
    answer: str
    session_history: List[AnyMessage]


class LlmModelGraph:
    """One agent's pipeline: retrieve (RAG context) → generate (LLM + tool loop).

    Every step has a sync and an async form: ask() / graph.invoke() use the first,
    aask() / graph.ainvoke() the second.
    """

    MAX_TOOL_CALLS = 5  # rounds of tool calls per answer — prevents infinite loops

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = None,
        rag_context_manager: RagContextManager = None,
        tools: list | None = None,
        agent_name: str = "agent",
        rag_enabled: bool = False,
        history_store: HistoryStore | None = None,
    ):
        self.agent_name = agent_name
        self._rag_enabled = rag_enabled
        self.system_prompt = system_prompt or load_prompt(Config.PROMPT_FILE)
        self.rag_context_manager = rag_context_manager or RagContextManager(Embedder())
        # Thread-safe; BaseAgent passes the shared SQLite store so history survives restarts
        self._history = history_store or InMemoryHistoryStore()

        # None = every MCP tool; [] = no tools
        self._tools = tools if tools is not None else _mcp.langchain_tools()
        logger.info("[%s] tools loaded: %s", agent_name, [t.name for t in self._tools])
        self.llm = llm.bind_tools(self._tools) if self._tools else llm

        self.graph = self._build_graph()

    # ------------------- PUBLIC API -------------------

    def ask(self, user_query: str, session_id: str = None) -> str:
        session_id = session_id or str(uuid.uuid4())

        @self._traced_agent(session_id)
        def _run(query: str) -> str:
            answer = self.graph.invoke(self._initial_state(query, self._get_history(session_id)))["answer"]
            self._remember(session_id, query, answer)
            return answer

        return _run(user_query)

    async def aask(self, user_query: str, session_id: str = None) -> str:
        """Async ask(): LLM and tool calls are awaited; history I/O runs in a worker thread."""
        session_id = session_id or str(uuid.uuid4())

        @self._traced_agent(session_id)
        async def _run(query: str) -> str:
            history = await asyncio.to_thread(self._get_history, session_id)
            answer = (await self.graph.ainvoke(self._initial_state(query, history)))["answer"]
            await asyncio.to_thread(self._remember, session_id, query, answer)
            return answer

        return await _run(user_query)

    def ask_stream(self, user_query: str, session_id: str = None):
        """The whole answer as a single chunk (agents don't stream token by token)."""
        yield self.ask(user_query, session_id=session_id)

    def close(self):
        close_mcp()

    # ------------------- GRAPH -------------------

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("retrieve", RunnableLambda(self._retrieve_node, afunc=self._aretrieve_node))
        graph.add_node("generate", RunnableLambda(self._generate_node, afunc=self._agenerate_node))
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _retrieve_node(self, state: State) -> dict:
        context = self.rag_context_manager.get_context(state["user_prompt"])
        logger.info("[%s] retrieve → context_len=%d", self.agent_name, len(context))
        return {"context": context}

    async def _aretrieve_node(self, state: State) -> dict:
        # Embedding + vector search are blocking — keep them off the event loop
        return await asyncio.to_thread(self._retrieve_node, state)

    def _generate_node(self, state: State) -> dict:
        messages = self._initial_messages(state)
        response = self.llm.invoke(messages)
        for _ in range(self.MAX_TOOL_CALLS):
            calls = response.tool_calls or []
            if not calls:
                break
            messages.append(response)
            messages.extend(
                ToolMessage(content=self._invoke_tool(call["name"], call.get("args") or {}), tool_call_id=call["id"])
                for call in calls
            )
            response = self.llm.invoke(messages)
        return self._answer(response)

    async def _agenerate_node(self, state: State) -> dict:
        messages = self._initial_messages(state)
        response = await self.llm.ainvoke(messages)
        for _ in range(self.MAX_TOOL_CALLS):
            calls = response.tool_calls or []
            if not calls:
                break
            messages.append(response)
            # Independent tool calls from one turn run concurrently
            results = await asyncio.gather(*(
                self._ainvoke_tool(call["name"], call.get("args") or {}) for call in calls
            ))
            messages.extend(
                ToolMessage(content=result, tool_call_id=call["id"]) for call, result in zip(calls, results)
            )
            response = await self.llm.ainvoke(messages)
        return self._answer(response)

    # ------------------- HELPERS -------------------

    def _initial_state(self, user_query: str, history: list = None) -> State:
        return {
            "user_prompt": user_query,
            "context": "",
            "answer": "",
            "session_history": list(history) if history else [],
        }

    def _initial_messages(self, state: State) -> list:
        messages: list = [SystemMessage(content=self.system_prompt)]
        if state["context"]:
            messages.append(SystemMessage(content=f"Relevant listings:\n{state['context']}"))
        elif self._rag_enabled:
            messages.append(SystemMessage(content=(
                "No matching listings were found in the database for this query. "
                "You MUST inform the user that no results are available. "
                "Do NOT invent, estimate, or describe any property not present above."
            )))
        messages.extend(state["session_history"])
        messages.append(HumanMessage(content=state["user_prompt"]))
        return messages

    def _answer(self, response) -> dict:
        if response.tool_calls:
            logger.warning("[%s] max tool iterations reached, forcing final answer", self.agent_name)
        # .text joins the text blocks — Claude responses also carry thinking blocks
        return {"answer": response.text}

    def _get_history(self, session_id: str) -> list:
        return self._history.get(self.agent_name, session_id, limit=int(Config.MAX_HISTORY_TURNS) * 2)

    def _remember(self, session_id: str, query: str, answer: str) -> None:
        self._history.append(self.agent_name, session_id, [HumanMessage(content=query), AIMessage(content=answer)])

    def _traced_agent(self, session_id: str):
        return traceable(
            name=f"agent.{self.agent_name}",
            run_type="chain",
            tags=["agent"],
            metadata={"agent": self.agent_name, "session_id": session_id},
        )

    # ------------------- TOOLS -------------------

    @staticmethod
    def _coerce_args(args: dict) -> dict:
        """Decode string values that hold a JSON array or object.

        LLMs sometimes emit ``{"tickers": "[\"AAPL\"]"}`` instead of ``{"tickers": ["AAPL"]}``.
        """
        coerced = {}
        for key, value in args.items():
            if isinstance(value, str) and value.strip().startswith(("[", "{")):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            coerced[key] = value
        return coerced

    def _find_tool(self, name: str):
        return next((tool for tool in self._tools if tool.name == name), None)

    def _traced_tool(self, name: str):
        return traceable(
            name=f"tool.{name}", run_type="tool", tags=["tool"], metadata={"agent": self.agent_name, "tool": name}
        )

    def _tool_missing(self, name: str) -> str:
        logger.warning("[%s] tool '%s' not found", self.agent_name, name)
        return f"Tool '{name}' not found."

    def _tool_failed(self, name: str, error: Exception) -> str:
        """A tool error goes back to the LLM as the tool's result, so it can recover."""
        logger.error("[%s] tool '%s' failed: %s", self.agent_name, name, error)
        return f"Error executing tool {name}: {error}"

    def _invoke_tool(self, name: str, args: dict) -> str:
        tool = self._find_tool(name)
        if tool is None:
            return self._tool_missing(name)

        @self._traced_tool(name)
        def _call(**kwargs) -> str:
            return tool.invoke(kwargs)

        try:
            args = self._coerce_args(args)
            logger.info("[%s] calling tool '%s' with args=%s", self.agent_name, name, args)
            return _call(**args)
        except Exception as e:
            return self._tool_failed(name, e)

    async def _ainvoke_tool(self, name: str, args: dict) -> str:
        tool = self._find_tool(name)
        if tool is None:
            return self._tool_missing(name)

        @self._traced_tool(name)
        async def _call(**kwargs) -> str:
            return await tool.ainvoke(kwargs)

        try:
            args = self._coerce_args(args)
            logger.info("[%s] calling tool '%s' with args=%s", self.agent_name, name, args)
            return await _call(**args)
        except Exception as e:
            return self._tool_failed(name, e)
