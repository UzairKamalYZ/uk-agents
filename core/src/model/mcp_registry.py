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
import queue
import subprocess
import threading
import time
from typing import Any, Optional

import httpx

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2025-06-18"
_CLIENT_INFO = {"name": "uk-agents", "version": "1.0"}

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
# MCPProcess — manages a single stdio subprocess
# ---------------------------------------------------------------------------


class MCPProcess:
    """
    Manages a persistent stdio subprocess running one MCP server.

    The subprocess is started lazily on first use and kept alive for the
    lifetime of the application.  A threading lock serialises concurrent calls.
    A background thread reads stdout into a queue, so every request waits at
    most ``timeout`` seconds; a server that stops answering is killed and
    restarted on the next call instead of blocking the caller forever.

    Parameters
    ----------
    name          : Human-readable label used in log messages.
    command       : Subprocess command list, e.g. ``["npx", "--yes", "@acme/mcp"]``.
    cwd           : Working directory for the subprocess (None → inherit the parent's).
    timeout       : Seconds to wait for each reply.
    start_timeout : Seconds to wait for the server to start and finish the handshake.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        cwd: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        start_timeout: float = DEFAULT_START_TIMEOUT,
    ):
        self.name = name
        self._command = command
        self._cwd = cwd
        self._timeout = timeout
        self._start_timeout = start_timeout
        self._process: Optional[subprocess.Popen] = None
        self._messages: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._req_id = 1

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_running(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        logger.info("mcp[%s]: starting subprocess", self.name)
        process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._cwd,
            text=True,
            bufsize=1,
        )
        self._process = process
        self._messages = queue.Queue()
        threading.Thread(
            target=self._read_stdout, args=(process, self._messages),
            name=f"mcp-{self.name}-reader", daemon=True,
        ).start()
        try:
            self._req_id = 1
            self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            }})
            self._await_response(1, self._start_timeout)
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except TimeoutError:
            self._kill()
            raise ConnectionError(f"mcp[{self.name}]: did not start within {self._start_timeout:.0f}s") from None
        except Exception:
            self._kill()
            raise
        logger.info("mcp[%s]: ready", self.name)

    @staticmethod
    def _read_stdout(process: subprocess.Popen, messages: queue.Queue) -> None:
        """Reader thread: one JSON-RPC message per line → queue; None marks EOF."""
        try:
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.put(json.loads(line))
                except ValueError:
                    logger.debug("mcp: ignoring non-JSON stdout line %r", line[:200])
        except Exception:  # pipe closed under us
            pass
        finally:
            messages.put(None)

    def _send(self, message: dict) -> None:
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _await_response(self, req_id: int, timeout: float) -> dict:
        """Next reply with this id; notifications and stale replies are skipped."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"mcp[{self.name}]: no reply to request {req_id} within {timeout:.0f}s") from None
            if message is None:
                raise ConnectionError(f"mcp[{self.name}]: server exited")
            if isinstance(message, dict) and message.get("id") == req_id and (
                "result" in message or "error" in message
            ):
                return message

    def _kill(self) -> None:
        """Stop a dead or unresponsive server; the next call starts a fresh one."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        self._process = None

    def _rpc(self, method: str, params: dict) -> dict:
        self._req_id += 1
        req_id = self._req_id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        try:
            return self._await_response(req_id, self._timeout)
        except (TimeoutError, ConnectionError):
            self._kill()
            raise

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
                if "error" in response:
                    logger.error("mcp[%s]: tool '%s' error: %s", self.name, name, response["error"])
                    return f"Tool '{name}' failed: {response['error'].get('message', 'unknown error')}"
                content = response.get("result", {}).get("content", [])
                result = "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
                logger.info("mcp[%s]: tool '%s' returned %d chars", self.name, name, len(result))
                return result
            except TimeoutError:
                logger.error("mcp[%s]: tool '%s' timed out after %.0fs — server restarted on next call",
                             self.name, name, self._timeout)
                return f"Tool '{name}' timed out — please try again."
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
        # Servers excluded from the default (server_name=None) tool set, e.g. ledger writers.
        self._owner_only: set[str] = set()

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
    ) -> "MCPRegistry":
        """Register a stdio subprocess MCP server."""
        self._servers[name] = MCPProcess(name, command, cwd=cwd, timeout=timeout)
        self._set_owner_only(name, owner_only)
        logger.info("mcp-registry: registered stdio server '%s'", name)
        return self

    def register_http(
        self,
        name: str,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        owner_only: bool = False,
    ) -> "MCPRegistry":
        """Register an HTTP-based MCP server."""
        self._servers[name] = MCPHttpProcess(name, url, params=params, headers=headers)
        self._set_owner_only(name, owner_only)
        logger.info("mcp-registry: registered HTTP server '%s' → %s", name, url)
        return self

    def is_owner_only(self, name: str) -> bool:
        return name in self._owner_only

    def _set_owner_only(self, name: str, owner_only: bool) -> None:
        if owner_only:
            self._owner_only.add(name)
        else:
            self._owner_only.discard(name)

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
                      if None (default), return tools from all servers except
                      owner-only ones (those must be requested by name).
                      An unregistered name yields no tools (never all servers).
        """
        if server_name is None:
            servers = {n: s for n, s in self._servers.items() if n not in self._owner_only}
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