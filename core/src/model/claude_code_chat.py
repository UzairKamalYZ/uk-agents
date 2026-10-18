"""
LangChain chat model backed by the Claude Code CLI running on this machine.

Calls go through the Claude Agent SDK, which drives the locally installed
``claude`` CLI and its logged-in account (e.g. a Claude Pro/Max subscription)
instead of an Anthropic API key.

Tools bound with ``bind_tools()`` are exposed to Claude Code as an in-process
MCP server, so Claude Code runs the whole tool loop itself and every response
comes back as plain text with no ``tool_calls``.  Claude Code's built-in tools
(Bash, Read, Edit, …) and the user's own settings, hooks and MCP servers are
disabled — the model can only call the tools the agent was given.
"""

import asyncio
import logging
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    SdkMcpTool,
    create_sdk_mcp_server,
    query,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_TOOL_SERVER_NAME = "agent_tools"


class ChatClaudeCode(BaseChatModel):
    """Chat model that answers through the local Claude Code CLI."""

    model: str | None = None
    effort: str | None = None
    max_turns: int = 10
    bound_tools: list[BaseTool] = []

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def bind_tools(self, tools: list, **kwargs: Any) -> "ChatClaudeCode":
        return self.model_copy(update={"bound_tools": list(tools)})

    # ------------------------------------------------------------------
    # BaseChatModel hooks
    # ------------------------------------------------------------------

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system_prompt, prompt = self._split_messages(messages)
        logger.info("claude-code → query (model=%s, tools=%d)", self.model, len(self.bound_tools))

        result: ResultMessage | None = None
        async for message in query(prompt=prompt, options=self._options(system_prompt)):
            if isinstance(message, ResultMessage):
                result = message

        if result is None or result.is_error:
            errors = (result.errors or result.result) if result else "no result returned"
            raise RuntimeError(f"Claude Code request failed: {errors}")

        logger.info("claude-code → done (turns=%d, %d ms)", result.num_turns, result.duration_ms)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.result or ""))])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _options(self, system_prompt: str) -> ClaudeAgentOptions:
        mcp_servers = {}
        allowed_tools = []
        if self.bound_tools:
            mcp_servers[_TOOL_SERVER_NAME] = create_sdk_mcp_server(
                _TOOL_SERVER_NAME, tools=[self._to_sdk_tool(t) for t in self.bound_tools]
            )
            allowed_tools = [f"mcp__{_TOOL_SERVER_NAME}__{t.name}" for t in self.bound_tools]

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self.model,
            effort=self.effort,
            max_turns=self.max_turns,
            tools=[],                    # no built-in Bash / Read / Edit / web tools
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            strict_mcp_config=True,      # ignore the user's own MCP servers
            setting_sources=[],          # ignore user/project settings, hooks, CLAUDE.md
            permission_mode="dontAsk",   # anything not in allowed_tools is denied
            # .env is loaded into os.environ; blank the key so the CLI uses its own login
            env={"ANTHROPIC_API_KEY": ""},
        )

    @staticmethod
    def _to_sdk_tool(tool: BaseTool) -> SdkMcpTool:
        async def handler(args: dict) -> dict:
            try:
                result = await asyncio.to_thread(tool.invoke, args)
                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as e:
                logger.error("claude-code tool '%s' failed: %s", tool.name, e)
                return {"content": [{"type": "text", "text": f"Error executing tool {tool.name}: {e}"}], "is_error": True}

        return SdkMcpTool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.get_input_schema().model_json_schema(),
            handler=handler,
        )

    @staticmethod
    def _split_messages(messages: list[BaseMessage]) -> tuple[str, str]:
        """Collapse a LangChain message list into (system prompt, user prompt).

        Each call is a fresh Claude Code session, so earlier turns are replayed
        as a transcript ahead of the latest user message.
        """
        system_parts = [m.text for m in messages if isinstance(m, SystemMessage)]
        turns = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(turns) == 1:
            return "\n\n".join(system_parts), turns[0].text

        lines = ["<conversation_history>"]
        for m in turns[:-1]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.text}")
        lines.append("</conversation_history>")
        lines.append("")
        lines.append(turns[-1].text)
        return "\n\n".join(system_parts), "\n".join(lines)
