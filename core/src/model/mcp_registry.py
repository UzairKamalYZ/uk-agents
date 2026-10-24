"""
MCP client registry — built on the official ``mcp`` SDK client (``mcp.Client``).

Register any number of MCP servers (stdio commands, Streamable HTTP URLs, or —
in tests — an in-process ``MCPServer``):

    from .mcp_registry import MCPRegistry

    registry = MCPRegistry()
    registry.register("finance", ["npx", "--yes", "@easysolutions906/mcp-finance"])
    registry.register_http("weather", "https://example.com/mcp", headers={"Authorization": "..."})

Then get LangChain tools generated from every server's schema:

    tools = registry.langchain_tools()          # sync .invoke() and async .ainvoke()

HOW IT RUNS
-----------
Every server keeps one persistent connection, owned by a task on a shared
background event loop (one daemon thread for the whole registry).  Sync callers
block on that loop; async callers await it, so both work from any thread or
event loop.  The SDK does the protocol (handshake, versions, notifications,
JSON-RPC ids), so calls to one server run concurrently.

Each call is capped at the server's ``timeout``; a server that times out or
fails is disconnected (a stdio subprocess is stopped) and reconnected on the
next call.  Tool errors never raise — the LLM gets a short error string.
"""

import asyncio
import concurrent.futures
import keyword
import logging
import os
import threading
from collections.abc import Callable, Coroutine
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Optional
from urllib.parse import urlencode

from langchain_core.tools import StructuredTool
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# Seconds to wait for a tool reply, and for a server to start (npx may download first)
DEFAULT_TIMEOUT = 60.0
DEFAULT_START_TIMEOUT = 120.0

_JSON_SCHEMA_TO_PYTHON: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _sanitize_field_name(name: str) -> str:
    """Make a Python-safe identifier from an MCP parameter name."""
    if keyword.iskeyword(name) or not name.isidentifier():
        return name + "_param"
    return name


def _schema_type(schema: dict) -> str:
    """JSON-Schema type name; for an optional ``anyOf: [X, null]`` the non-null X."""
    if "type" in schema:
        return schema["type"]
    for option in schema.get("anyOf", []):
        if option.get("type") not in (None, "null"):
            return option["type"]
    return "string"


def _build_args_model(tool_name: str, input_schema: dict) -> type[BaseModel]:
    """
    Construct a Pydantic model from an MCP JSON-Schema inputSchema dict.

    Reserved Python keywords (e.g. ``from``) are renamed by appending
    ``_param``.  The original name is preserved in the ``_name_map`` class
    attribute so the tool invoker can remap them back.
    """
    properties: dict[str, dict] = input_schema.get("properties", {})
    required: set[str] = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}

    for orig_name, schema in properties.items():
        safe_name = _sanitize_field_name(orig_name)
        py_type = _JSON_SCHEMA_TO_PYTHON.get(_schema_type(schema), str)
        description = schema.get("description", orig_name)

        if orig_name in required:
            fields[safe_name] = (py_type, Field(description=description))
        else:
            default = schema.get("default", None)
            fields[safe_name] = (
                Optional[py_type],
                Field(default=default, description=description),
            )

    model = create_model(f"{tool_name}_Input", **fields)
    # Store the safe→original name mapping as a class attribute.
    model._name_map = {_sanitize_field_name(k): k for k in properties}  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Background event loop shared by every MCP connection
# ---------------------------------------------------------------------------


class _LoopThread:
    """One asyncio loop on a daemon thread; MCP connections live here."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="mcp-loop", daemon=True).start()
                self._loop = loop
            return self._loop

    def submit(self, coro: Coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure())

    def run(self, coro: Coroutine) -> Any:
        """Run on the MCP loop and block for the result (sync callers)."""
        return self.submit(coro).result()

    async def arun(self, coro: Coroutine) -> Any:
        """Run on the MCP loop and await the result (async callers, any loop)."""
        return await asyncio.wrap_future(self.submit(coro))


# ---------------------------------------------------------------------------
# MCPServerConnection — one server, one persistent SDK client
# ---------------------------------------------------------------------------


class MCPServerConnection:
    """
    A persistent ``mcp.Client`` connection to one server.

    The client is entered and exited inside a single long-lived task (the SDK's
    anyio scopes must close in the task that opened them); other tasks send
    requests through it.  All methods run on the registry's MCP loop.

    Parameters
    ----------
    name          : label used in logs
    target        : factory returning what ``mcp.Client`` connects to — a
                    transport context manager, URL or in-process server
    timeout       : seconds to wait for each reply
    start_timeout : seconds to wait for the connection + handshake
    command, cwd  : the stdio command line and working directory (informational)
    """

    def __init__(
        self,
        name: str,
        target: Callable[[], Any],
        timeout: float = DEFAULT_TIMEOUT,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        command: list[str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.cwd = cwd
        self._target = target
        self.timeout = timeout
        self.start_timeout = start_timeout
        self._client: Client | None = None
        self._tools: list[Tool] | None = None
        self._runner: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._connect_lock: asyncio.Lock | None = None

    # -- connection lifecycle (MCP loop only) -------------------------------

    async def _connected(self) -> Client:
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self._client is not None and self._runner is not None and not self._runner.done():
                return self._client
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            self._stop = asyncio.Event()
            self._runner = asyncio.create_task(self._hold(ready, self._stop), name=f"mcp-{self.name}")
            try:
                self._client = await asyncio.wait_for(asyncio.shield(ready), self.start_timeout)
            except BaseException:
                ready.cancel()  # nobody is waiting for it any more
                await self._disconnect(graceful=False)
                raise
            logger.info("mcp[%s]: connected", self.name)
            return self._client

    async def _hold(self, ready: asyncio.Future, stop: asyncio.Event) -> None:
        """Owns the client: enter, publish, wait for stop, exit — all in this task."""
        try:
            async with Client(self._target()) as client:
                if not ready.done():
                    ready.set_result(client)
                await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else ConnectionError(str(exc)))
            if not isinstance(exc, (Exception, asyncio.CancelledError)):
                raise

    async def _disconnect(self, graceful: bool = True) -> None:
        """Close the client (stops a stdio subprocess); the next request reconnects.

        graceful=False cancels straight away — for a connection that never came up.
        """
        runner, stop = self._runner, self._stop
        self._client, self._tools, self._runner = None, None, None
        if runner is None:
            return
        stop.set()
        if graceful:
            try:
                await asyncio.wait_for(asyncio.shield(runner), 10)
            except BaseException:
                pass
        if not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        logger.info("mcp[%s]: disconnected", self.name)

    # -- requests (MCP loop only) -------------------------------------------

    async def list_tools(self) -> list[Tool]:
        if self._tools is None:
            try:
                client = await self._connected()
                tools, cursor = [], None
                while True:
                    page = await asyncio.wait_for(client.list_tools(cursor=cursor), self.timeout)
                    tools.extend(page.tools)
                    cursor = page.next_cursor
                    if not cursor:
                        break
            except Exception:
                await self._disconnect()
                raise
            self._tools = tools
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Never raises: errors come back as a short message for the LLM."""
        try:
            client = await self._connected()
            logger.info("mcp[%s]: calling tool '%s' args=%s", self.name, name, arguments)
            result = await asyncio.wait_for(client.call_tool(name, arguments), self.timeout)
        except TimeoutError:
            logger.error("mcp[%s]: tool '%s' timed out after %.0fs — reconnecting on next call",
                         self.name, name, self.timeout)
            await self._disconnect()
            return f"Tool '{name}' timed out — please try again."
        except Exception:
            logger.exception("mcp[%s]: tool call failed name=%s args=%s", self.name, name, arguments)
            await self._disconnect()
            return f"Tool '{name}' is currently unavailable."
        text = _result_text(result)
        if result.is_error:
            logger.error("mcp[%s]: tool '%s' error: %s", self.name, name, text)
            return f"Tool '{name}' failed: {text or 'unknown error'}"
        logger.info("mcp[%s]: tool '%s' returned %d chars", self.name, name, len(text))
        return text

    async def close(self) -> None:
        await self._disconnect()


def _result_text(result: CallToolResult) -> str:
    return "\n".join(item.text for item in result.content if getattr(item, "type", None) == "text")


def _stdio_target(
    command: list[str], cwd: str | None, env: dict[str, str] | None
) -> Callable[[], AbstractAsyncContextManager]:
    # The SDK starts the server with a minimal environment (HOME, PATH, …) plus `env`
    params = StdioServerParameters(command=command[0], args=list(command[1:]), cwd=cwd, env=env)

    @asynccontextmanager
    async def target():
        # Server stderr (npx install noise, tracebacks) is discarded, as before
        with open(os.devnull, "w") as errlog:
            async with stdio_client(params, errlog=errlog) as streams:
                yield streams

    return target


def _http_target(url: str, params: dict[str, str] | None, headers: dict[str, str] | None) -> Callable[[], Any]:
    if params:
        url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"

    def target():
        if not headers:
            return url
        return streamable_http_client(url, http_client=create_mcp_http_client(headers=headers))

    return target


# ---------------------------------------------------------------------------
# MCPRegistry
# ---------------------------------------------------------------------------


class MCPRegistry:
    """
    Registry for multiple MCP servers.

    Typical usage::

        registry.register("finance", ["npx", "--yes", "@easysolutions906/mcp-finance"])
        tools = registry.langchain_tools()          # LangChain tools from every server
        registry.call_tool("currency_convert", {"from": "PLN", "to": "USD"})
        registry.close()
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConnection] = {}
        # Servers excluded from the default (server_name=None) tool set, e.g. ledger writers.
        self._owner_only: set[str] = set()
        self._loop = _LoopThread()

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        command: list[str],
        cwd: str | None = None,
        owner_only: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> "MCPRegistry":
        """Register a stdio subprocess MCP server.

        The subprocess gets only a minimal environment (HOME, PATH, …) plus `env`,
        so secrets in os.environ don't leak to third-party servers.
        """
        self._add(MCPServerConnection(name, _stdio_target(command, cwd, env), timeout, start_timeout,
                                      command=command, cwd=cwd), owner_only)
        logger.info("mcp-registry: registered stdio server '%s'", name)
        return self

    def register_http(
        self,
        name: str,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        owner_only: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> "MCPRegistry":
        """Register a Streamable HTTP MCP server (query params are appended to the URL)."""
        self._add(MCPServerConnection(name, _http_target(url, params, headers), timeout), owner_only)
        logger.info("mcp-registry: registered HTTP server '%s' → %s", name, url)
        return self

    def register_server(
        self,
        name: str,
        target: Any,
        owner_only: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        start_timeout: float = DEFAULT_START_TIMEOUT,
    ) -> "MCPRegistry":
        """Register anything ``mcp.Client`` accepts, e.g. an in-process ``MCPServer``."""
        self._add(MCPServerConnection(name, lambda: target, timeout, start_timeout), owner_only)
        return self

    def _add(self, server: MCPServerConnection, owner_only: bool) -> None:
        self._servers[server.name] = server
        if owner_only:
            self._owner_only.add(server.name)
        else:
            self._owner_only.discard(server.name)

    def is_owner_only(self, name: str) -> bool:
        return name in self._owner_only

    # ------------------------------------------------------------------ #
    # Tool dispatch                                                        #
    # ------------------------------------------------------------------ #

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool by name on whichever registered server offers it."""
        for server in self._servers.values():
            try:
                names = {t.name for t in self._loop.run(server.list_tools())}
            except Exception:
                logger.exception("mcp-registry: failed to list tools for server '%s'", server.name)
                continue
            if tool_name in names:
                return self._loop.run(server.call_tool(tool_name, arguments))
        return f"Tool '{tool_name}' is currently unavailable."

    # ------------------------------------------------------------------ #
    # LangChain integration                                                #
    # ------------------------------------------------------------------ #

    def langchain_tools(self, server_name: str | None = None) -> list[StructuredTool]:
        """
        Auto-generate LangChain StructuredTools from registered servers.

        Parameters
        ----------
        server_name : if given, only return tools from that server;
                      if None (default), return tools from all servers except
                      owner-only ones (those must be requested by name).
                      An unregistered name yields no tools (never all servers).
        """
        if server_name is None:
            servers = [s for n, s in self._servers.items() if n not in self._owner_only]
        elif server_name in self._servers:
            servers = [self._servers[server_name]]
        else:
            logger.warning("mcp-registry: server '%s' is not registered — returning no tools", server_name)
            return []

        tools: list[StructuredTool] = []
        for server in servers:
            try:
                mcp_tools = self._loop.run(server.list_tools())
            except Exception:
                logger.exception("mcp-registry: failed to list tools for server '%s'", server.name)
                continue
            tools.extend(self._make_langchain_tool(server, t) for t in mcp_tools)
        return tools

    def _make_langchain_tool(self, server: MCPServerConnection, mcp_tool: Tool) -> StructuredTool:
        tool_name = mcp_tool.name
        input_schema = mcp_tool.input_schema or {"type": "object", "properties": {}}
        args_model = _build_args_model(tool_name, input_schema)
        name_map: dict[str, str] = args_model._name_map  # type: ignore[attr-defined]
        accepted = set(input_schema.get("properties", {}))

        def _arguments(kwargs: dict) -> dict:
            # Remap safe Python names back to the MCP names; drop keys the tool doesn't declare
            remapped = {name_map.get(k, k): v for k, v in kwargs.items()}
            return {k: v for k, v in remapped.items() if k in accepted} if accepted else remapped

        def _invoke(**kwargs: Any) -> str:
            return self._loop.run(server.call_tool(tool_name, _arguments(kwargs)))

        async def _ainvoke(**kwargs: Any) -> str:
            return await self._loop.arun(server.call_tool(tool_name, _arguments(kwargs)))

        return StructuredTool.from_function(
            func=_invoke,
            coroutine=_ainvoke,
            name=tool_name,
            description=mcp_tool.description or tool_name,
            args_schema=args_model,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def close_all(self) -> None:
        """Disconnect every server (stdio subprocesses are stopped)."""
        for server in self._servers.values():
            try:
                self._loop.run(server.close())
            except Exception:
                logger.exception("mcp-registry: error closing '%s'", server.name)

    def close(self) -> None:
        """Alias for :meth:`close_all`."""
        self.close_all()
