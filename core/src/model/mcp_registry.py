"""
Generic stdio MCP client registry.

Register any number of MCP servers with a single command:

    from .mcp_registry import registry

    registry.register("finance", ["npx", "--yes", "@easysolutions906/mcp-finance"])
    registry.register("weather", ["npx", "--yes", "@acme/mcp-weather"])

Then call any tool by name (the registry routes to the right server automatically):

    result = registry.call_tool("currency_convert", {"from": "PLN", "to": "USD", "amount": 1})

Or get LangChain StructuredTools auto-generated from every server's schema:

    tools = registry.langchain_tools()
"""

import json
import keyword
import logging
import subprocess
import threading
from typing import Any, Optional

import httpx

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

_INIT_PAYLOAD = (
    json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "uk-agents", "version": "1.0"},
        },
    })
    + "\n"
)

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
        py_type = _JSON_SCHEMA_TO_PYTHON.get(schema.get("type", "string"), str)
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
# MCPProcess — manages a single stdio subprocess
# ---------------------------------------------------------------------------


class MCPProcess:
    """
    Manages a persistent stdio subprocess running one MCP server.

    The subprocess is started lazily on first use and kept alive for the
    lifetime of the application.  A threading lock serialises concurrent calls.

    Parameters
    ----------
    name    : Human-readable label used in log messages.
    command : Subprocess command list, e.g. ``["npx", "--yes", "@acme/mcp"]``.
    """

    def __init__(self, name: str, command: list[str]):
        self.name = name
        self._command = command
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._req_id = 1

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_running(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        logger.info("mcp[%s]: starting subprocess", self.name)
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._process.stdin.write(_INIT_PAYLOAD)
        self._process.stdin.flush()
        self._process.stdout.readline()  # discard init response
        logger.info("mcp[%s]: ready", self.name)

    def _rpc(self, method: str, params: dict) -> dict:
        self._req_id += 1
        payload = (
            json.dumps({
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": method,
                "params": params,
            })
            + "\n"
        )
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        return json.loads(self._process.stdout.readline())

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def list_tools(self) -> list[dict]:
        """Query the server for its available tools (``tools/list``)."""
        with self._lock:
            self._ensure_running()
            response = self._rpc("tools/list", {})
            return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call a tool on this MCP server.

        Thread-safe.  Never raises — exceptions are caught and an error
        string is returned so the LLM sees a clean message.
        """
        with self._lock:
            try:
                self._ensure_running()
                logger.info("mcp[%s]: calling tool '%s' args=%s", self.name, name, arguments)
                response = self._rpc("tools/call", {"name": name, "arguments": arguments})
                content = response.get("result", {}).get("content", [])
                result = "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
                logger.info("mcp[%s]: tool '%s' returned %d chars", self.name, name, len(result))
                return result
            except Exception:
                logger.exception("mcp[%s]: tool call failed name=%s args=%s", self.name, name, arguments)
                return f"Tool '{name}' is currently unavailable."

    def close(self) -> None:
        """Terminate the subprocess if it is running."""
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                logger.info("mcp[%s]: subprocess terminated", self.name)
            self._process = None


# ---------------------------------------------------------------------------
# MCPHttpProcess — manages a single HTTP-based MCP server
# ---------------------------------------------------------------------------


class MCPHttpProcess:
    """
    HTTP/JSON-RPC MCP client for remote MCP servers.

    Unlike MCPProcess (which spawns a subprocess), this sends JSON-RPC 2.0
    requests over HTTP POST.  The API key is passed as a query parameter.

    Parameters
    ----------
    name          : Human-readable label used in log messages.
    url           : Full base URL of the MCP endpoint.
    params        : Query parameters appended to every request (e.g. api key).
    headers       : Extra HTTP headers (Content-Type is set automatically).
    """

    def __init__(
        self,
        name: str,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self._url = url
        self._params = params or {}
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._lock = threading.Lock()
        self._req_id = 0
        self._tool_schemas: dict[str, set[str]] = {}  # tool_name → set of accepted param names

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._req_id += 1
        payload: dict = {"jsonrpc": "2.0", "id": self._req_id, "method": method}
        if params:
            payload["params"] = params

        response = httpx.post(
            self._url,
            json=payload,
            headers=self._headers,
            params=self._params,
            timeout=30,
        )

        if not response.is_success:
            logger.error(
                "mcp[%s]: HTTP %d for method=%r — body: %s",
                self.name, response.status_code, method, response.text[:500],
            )
            response.raise_for_status()

        return response.json()

    def list_tools(self) -> list[dict]:
        with self._lock:
            logger.info("mcp[%s]: listing tools", self.name)
            response = self._rpc("tools/list")
            tools = response.get("result", {}).get("tools", [])
            # Cache each tool's accepted parameter names for argument filtering
            for t in tools:
                props = t.get("inputSchema", {}).get("properties", {})
                self._tool_schemas[t["name"]] = set(props.keys())
            return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        with self._lock:
            try:
                # Strip arguments not declared in the tool's schema.
                # Guards against LLM passing extra keys (e.g. legacy 'function'
                # param from Alpha Vantage REST API) that the MCP implementation
                # does not accept.
                accepted = self._tool_schemas.get(name)
                if accepted is not None:
                    filtered = {k: v for k, v in arguments.items() if k in accepted}
                    dropped = set(arguments) - set(filtered)
                    if dropped:
                        logger.warning("mcp[%s]: dropped unknown args for '%s': %s", self.name, name, dropped)
                    arguments = filtered

                logger.info("mcp[%s]: calling tool '%s' args=%s", self.name, name, arguments)
                response = self._rpc("tools/call", {"name": name, "arguments": arguments})
                content = response.get("result", {}).get("content", [])
                result = "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
                logger.info("mcp[%s]: tool '%s' returned %d chars", self.name, name, len(result))
                return result
            except Exception:
                logger.exception("mcp[%s]: HTTP tool call failed name=%s args=%s", self.name, name, arguments)
                return f"Tool '{name}' is currently unavailable."

    def close(self) -> None:
        pass  # No persistent connection to close


# ---------------------------------------------------------------------------
# MCPRegistry — manages multiple MCPProcess instances
# ---------------------------------------------------------------------------


class MCPRegistry:
    """
    Registry for multiple MCP servers.

    Typical usage::

        registry.register("finance", ["npx", "--yes", "@easysolutions906/mcp-finance"])
        registry.register("weather", ["npx", "--yes", "@acme/mcp-weather"])

        # Route a tool call to whichever server handles it
        result = registry.call_tool("currency_convert", {"from": "PLN", "to": "USD"})

        # Get LangChain tools auto-generated from every server's schema
        tools = registry.langchain_tools()

        # Shutdown
        registry.close_all()
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPProcess] = {}

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(self, name: str, command: list[str]) -> "MCPRegistry":
        """Register a stdio subprocess MCP server."""
        self._servers[name] = MCPProcess(name, command)
        logger.info("mcp-registry: registered stdio server '%s'", name)
        return self

    def register_http(
        self,
        name: str,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> "MCPRegistry":
        """Register an HTTP-based MCP server."""
        self._servers[name] = MCPHttpProcess(name, url, params=params, headers=headers)
        logger.info("mcp-registry: registered HTTP server '%s' → %s", name, url)
        return self

    # ------------------------------------------------------------------ #
    # Tool dispatch                                                        #
    # ------------------------------------------------------------------ #

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool by name, routing to the correct registered server.

        Servers are tried in registration order.  The first server that
        returns a non-error result wins.  If no server handles the tool,
        an error string is returned.
        """
        for server in self._servers.values():
            result = server.call_tool(tool_name, arguments)
            if "unavailable" not in result.lower():
                return result
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
                      if None (default), return tools from all servers.
                      An unregistered name yields no tools (never all servers).
        """
        if server_name is None:
            servers = self._servers
        elif server_name in self._servers:
            servers = {server_name: self._servers[server_name]}
        else:
            logger.warning("mcp-registry: server '%s' is not registered — returning no tools", server_name)
            return []

        tools: list[StructuredTool] = []
        for server in servers.values():
            try:
                mcp_tools = server.list_tools()
            except Exception:
                logger.exception(
                    "mcp-registry: failed to list tools for server '%s'", server.name
                )
                continue

            for mcp_tool in mcp_tools:
                tools.append(self._make_langchain_tool(server, mcp_tool))

        return tools

    def _make_langchain_tool(self, server: MCPProcess, mcp_tool: dict) -> StructuredTool:
        tool_name: str = mcp_tool["name"]
        description: str = mcp_tool.get("description", tool_name)
        input_schema: dict = mcp_tool.get(
            "inputSchema", {"type": "object", "properties": {}}
        )

        args_model = _build_args_model(tool_name, input_schema)
        name_map: dict[str, str] = args_model._name_map  # type: ignore[attr-defined]

        def _invoke(**kwargs: Any) -> str:
            # Remap safe Python names back to original MCP parameter names.
            remapped = {name_map.get(k, k): v for k, v in kwargs.items()}
            return server.call_tool(tool_name, remapped)

        return StructuredTool.from_function(
            func=_invoke,
            name=tool_name,
            description=description,
            args_schema=args_model,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def close_all(self) -> None:
        """Terminate all registered MCP subprocesses."""
        for server in self._servers.values():
            server.close()

    def close(self) -> None:
        """Alias for :meth:`close_all`."""
        self.close_all()