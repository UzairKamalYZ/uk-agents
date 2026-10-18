import json
import logging
import os
import uuid
from typing import List, TypedDict

logger = logging.getLogger(__name__)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage
from langsmith import traceable

from langgraph.graph import StateGraph, START, END

from ..config.config import Config
from ..utils import load_prompt
from .embedder import Embedder
from .mcp_tools import close_mcp, _mcp
from .rag_context_manager import RagContextManager

os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)


# ------------------- STATE -------------------

class State(TypedDict):
    user_prompt: str
    context: str
    answer: str
    session_history: List[AnyMessage]


# ------------------- MODEL -------------------

class LlmModelGraph:

    MAX_TOOL_CALLS = 5  # 🔥 CRITICAL: prevents infinite loops

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str = None,
        rag_context_manager: RagContextManager = None,
        tools: list | None = None,
        agent_name: str = "agent",
        rag_enabled: bool = False,
    ):
        self.agent_name = agent_name
        self._rag_enabled = rag_enabled
        self.system_prompt = system_prompt or load_prompt(Config.PROMPT_FILE)

        self.rag_context_manager = rag_context_manager or RagContextManager(Embedder())
        self._histories: dict[str, list] = {}

        # Use caller-supplied tools when provided; fall back to the full MCP registry.
        # Pass an explicit empty list to disable all tool use for an agent.
        self._tools = tools if tools is not None else _mcp.langchain_tools()
        logger.info("[%s] tools loaded: %s", agent_name, [t.name for t in self._tools])
        self.llm = llm.bind_tools(self._tools) if self._tools else llm

        self.graph = self._build_graph()

    # ------------------- PUBLIC API -------------------

    def ask(self, user_query: str, session_id: str = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())

        logger.info("[%s] LlmModelGraph.ask() entered (session=%s)", self.agent_name, session_id)

        @traceable(
            name=f"agent.{self.agent_name}",
            run_type="chain",
            tags=["agent"],
            metadata={"agent": self.agent_name, "session_id": session_id},
        )
        def _run(query: str) -> str:
            history = self._get_history(session_id)
            state = self._initial_state(query, history)
            logger.info("[%s] graph.invoke → start (session=%s)", self.agent_name, session_id)
            result = self.graph.invoke(state)
            logger.info("[%s] graph.invoke → done", self.agent_name)
            answer = result["answer"]
            self._histories[session_id] = history + [
                HumanMessage(content=query),
                AIMessage(content=answer),
            ]
            return answer

        return _run(user_query)

    def ask_stream(self, user_query: str, session_id: str = None):
        if session_id is None:
            session_id = str(uuid.uuid4())

        answer = self.ask(user_query, session_id=session_id)
        yield answer

    def close(self):
        close_mcp()
        logger.debug("LlmModelGraph closed")

    # ------------------- GRAPH -------------------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    # ------------------- NODES -------------------

    def _retrieve_node(self, state: State) -> dict:
        logger.info("[%s] retrieve → querying RAG context", self.agent_name)
        context = self.rag_context_manager.get_context(state["user_prompt"])
        logger.info("[%s] retrieve → context_len=%d", self.agent_name, len(context))
        return {"context": context}

    def _generate_node(self, state: State) -> dict:
        logger.info("[%s] generate → invoking LLM", self.agent_name)
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

        response = self.llm.invoke(messages)

        iterations = 0  # 🔥 loop guard

        while iterations < self.MAX_TOOL_CALLS:
            iterations += 1

            tool_calls = response.tool_calls or []

            if not tool_calls:
                break

            logger.info("Tool iteration %d: detected %d tool call(s)", iterations, len(tool_calls))

            messages.append(response)

            for tool_call in tool_calls:
                name = tool_call["name"]
                args = tool_call.get("args") or {}

                logger.info("Calling tool '%s' with args=%s", name, args)

                result = self._invoke_tool(name, args)

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

                logger.info("Tool '%s' executed, result_len=%d", name, len(result))

            response = self.llm.invoke(messages)

        # 🔥 Safety fallback if loop exceeded
        if iterations >= self.MAX_TOOL_CALLS:
            logger.warning("Max tool iterations reached, forcing final answer")

        return {
            # .text joins the text blocks — Claude responses also carry thinking blocks
            "answer": response.text
        }

    # ------------------- HELPERS -------------------

    @staticmethod
    def _coerce_args(args: dict) -> dict:
        """
        Deserialize any string values that are JSON-encoded arrays or objects.

        LLMs sometimes emit ``{"tickers": "[\"AAPL\"]"}`` instead of
        ``{"tickers": ["AAPL"]}`` when constructing tool calls as plain text.
        """
        coerced = {}
        for k, v in args.items():
            if isinstance(v, str):
                stripped = v.strip()
                if stripped.startswith(("[", "{")):
                    try:
                        coerced[k] = json.loads(stripped)
                        continue
                    except json.JSONDecodeError:
                        pass
            coerced[k] = v
        return coerced

    def _invoke_tool(self, name: str, args: dict) -> str:
        for tool in self._tools:
            if tool.name == name:
                @traceable(
                    name=f"tool.{name}",
                    run_type="tool",
                    tags=["tool"],
                    metadata={"agent": self.agent_name, "tool": name},
                )
                def _call(**kwargs) -> str:
                    return tool.invoke(kwargs)

                try:
                    args = self._coerce_args(args)
                    logger.info("[%s] calling tool '%s' with args=%s", self.agent_name, name, args)
                    result = _call(**args)
                    logger.info("[%s] tool '%s' returned %d chars", self.agent_name, name, len(result))
                    return result
                except Exception as e:
                    logger.error("[%s] tool '%s' failed: %s", self.agent_name, name, e)
                    return f"Error executing tool {name}: {str(e)}"

        logger.warning("[%s] tool '%s' not found", self.agent_name, name)
        return f"Tool '{name}' not found."

    def _get_history(self, session_id: str) -> list:
        history = self._histories.setdefault(session_id, [])
        max_messages = int(Config.MAX_HISTORY_TURNS) * 2
        return history[-max_messages:] if len(history) > max_messages else history

    def _initial_state(self, user_query: str, history: list = None) -> State:
        return {
            "user_prompt": user_query,
            "context": "",
            "answer": "",
            "session_history": list(history) if history else [],
        }

