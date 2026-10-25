import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.src.model.mcp_tools import _mcp, close_mcp, _load_registry
from core.src.model.mcp_registry import MCPRegistry


class TestLoadRegistry(unittest.TestCase):
    """_load_registry() builds an MCPRegistry from an mcp.json file."""

    def _write_config(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps(data))
        return path

    def test_should_register_all_servers_from_config(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "mcp.json"
            cfg.write_text(json.dumps({
                "servers": [
                    {"name": "finance", "command": ["npx", "mcp-finance"]},
                    {"name": "maps",    "command": ["npx", "mcp-maps"]},
                ]
            }))
            registry = _load_registry(cfg)

        self.assertIn("finance", registry._servers)
        self.assertIn("maps", registry._servers)

    def test_should_return_empty_registry_when_file_missing(self):
        registry = _load_registry(Path("/nonexistent/mcp.json"))
        self.assertEqual(len(registry._servers), 0)

    def test_should_return_empty_registry_when_json_invalid(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("NOT JSON {{{")
            path = Path(f.name)
        try:
            registry = _load_registry(path)
        finally:
            path.unlink()
        self.assertEqual(len(registry._servers), 0)

    def test_should_skip_entries_missing_name_or_command(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "mcp.json"
            cfg.write_text(json.dumps({
                "servers": [
                    {"name": "ok", "command": ["npx", "mcp-ok"]},
                    {"name": "no-command"},
                    {"command": ["npx", "no-name"]},
                ]
            }))
            registry = _load_registry(cfg)

        self.assertIn("ok", registry._servers)
        self.assertEqual(len(registry._servers), 1)


class TestModuleSingleton(unittest.TestCase):
    """The module-level _mcp singleton is loaded from mcp.json."""

    def test_should_be_an_mcp_registry(self):
        self.assertIsInstance(_mcp, MCPRegistry)

    def test_should_have_finance_server_registered(self):
        """Finance server declared in mcp.json is registered."""
        self.assertIn("finance", _mcp._servers)

    def test_should_use_finance_mcp_command(self):
        server = _mcp._servers["finance"]
        self.assertIn("@easysolutions906/mcp-finance", server.command)

    @patch("core.src.model.mcp_tools._mcp")
    def test_close_mcp_delegates_to_registry(self, mock_registry):
        """close_mcp() calls close() on the shared registry."""
        close_mcp()
        mock_registry.close.assert_called_once()




def test_python_placeholder_resolves_to_sys_executable():
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "mcp.json"
        cfg.write_text(json.dumps({"servers": [{"name": "x", "command": ["{python}", "-m", "x"]}]}))
        registry = _load_registry(cfg)
    assert registry._servers["x"].command == [sys.executable, "-m", "x"]


def test_langchain_tools_unknown_server_returns_empty():
    registry = MCPRegistry().register_server("a", _server("t"))
    assert registry.langchain_tools(server_name="missing") == []
    assert registry._servers["a"]._client is None


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_python_placeholder_runs_from_project_root():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "mcp.json"
        cfg.write_text(json.dumps({"servers": [
            {"name": "py", "command": ["{python}", "-m", "x"]},
            {"name": "npx", "command": ["npx", "y"]},
        ]}))
        registry = _load_registry(cfg)
    assert registry._servers["py"].cwd == str(_PROJECT_ROOT)
    assert registry._servers["npx"].cwd is None


def _server(*tool_names):
    from mcp.server.mcpserver import MCPServer
    server = MCPServer("s")
    for name in tool_names:
        server.tool(name=name, description=name)(lambda: name)
    return server


def _registry_with(owner_only_name: str) -> MCPRegistry:
    registry = MCPRegistry()
    registry.register_server("public", _server("pub_tool"))
    registry.register_server(owner_only_name, _server("priv_tool"), owner_only=True)
    return registry


def test_default_tool_set_excludes_owner_only_servers():
    registry = _registry_with("expenses")
    try:
        assert [t.name for t in registry.langchain_tools()] == ["pub_tool"]
        assert registry._servers["expenses"]._client is None  # never even connected
    finally:
        registry.close()


def test_owner_only_server_still_available_by_name():
    registry = _registry_with("expenses")
    try:
        assert [t.name for t in registry.langchain_tools(server_name="expenses")] == ["priv_tool"]
    finally:
        registry.close()


def test_mcp_json_owner_only_flag_is_recorded():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "mcp.json"
        cfg.write_text(json.dumps({"servers": [
            {"name": "a", "command": ["npx", "a"]},
            {"name": "b", "command": ["npx", "b"], "owner_only": True},
        ]}))
        registry = _load_registry(cfg)
    assert registry.is_owner_only("b") and not registry.is_owner_only("a")


def test_optional_anyof_param_keeps_its_type():
    from core.src.model.mcp_registry import _build_args_model
    model = _build_args_model("t", {
        "properties": {"amount": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None}},
        "required": [],
    })
    assert model.model_fields["amount"].annotation == float | None
    assert model(amount=19.5).amount == 19.5


# ---------------------------------------------------------------------------
# Official SDK client against real MCP servers (in-process and stdio subprocess)
# ---------------------------------------------------------------------------

import asyncio
import os

import pytest

from core.tests.model import mcp_test_server

_SERVER_SCRIPT = str(Path(mcp_test_server.__file__))


@pytest.fixture
def inproc():
    registry = MCPRegistry().register_server("t", mcp_test_server.server, timeout=1)
    yield registry
    registry.close()


@pytest.fixture
def stdio(tmp_path):
    registry = MCPRegistry().register("t", [sys.executable, _SERVER_SCRIPT], cwd=str(tmp_path), timeout=1)
    yield registry
    registry.close()


def _tool(registry, name):
    return next(t for t in registry.langchain_tools("t") if t.name == name)


def test_tools_are_generated_from_the_server_schema(inproc):
    echo = _tool(inproc, "echo")
    assert echo.description == "Echo the text back."
    assert echo.invoke({"text": "hi", "times": 2}) == "hi hi"
    assert echo.invoke({"text": "hi"}) == "hi"  # optional param omitted


def test_tools_support_async_invoke(inproc):
    assert asyncio.run(_tool(inproc, "echo").ainvoke({"text": "async"})) == "async"


def test_undeclared_arguments_are_dropped(inproc):
    echo = _tool(inproc, "echo")
    assert echo.func(text="hi", function="legacy-extra") == "hi"


def test_tool_error_is_reported_to_the_llm(inproc):
    assert _tool(inproc, "boom").invoke({}).startswith("Tool 'boom' failed:")


def test_call_tool_routes_by_tool_name(inproc):
    assert inproc.call_tool("echo", {"text": "routed"}) == "routed"
    assert inproc.call_tool("missing", {}) == "Tool 'missing' is currently unavailable."


def test_timeout_reconnects_on_next_call(inproc):
    hang = _tool(inproc, "hang")
    assert hang.invoke({"seconds": 30}) == "Tool 'hang' timed out — please try again."
    assert _tool(inproc, "echo").invoke({"text": "back"}) == "back"


def test_stdio_server_runs_in_cwd_and_is_reused(stdio, tmp_path):
    assert os.path.realpath(_tool(stdio, "cwd").invoke({})) == os.path.realpath(tmp_path)
    pid = _tool(stdio, "pid")
    assert pid.invoke({}) == pid.invoke({})  # one persistent subprocess


def test_stdio_hung_server_is_restarted(stdio):
    first = _tool(stdio, "pid").invoke({})
    assert "timed out" in _tool(stdio, "hang").invoke({"seconds": 30})
    second = _tool(stdio, "pid").invoke({})
    assert second != first  # fresh subprocess
    with pytest.raises(ProcessLookupError):
        os.kill(int(first), 0)  # the hung one was stopped


def test_stdio_close_stops_the_subprocess(stdio):
    pid = int(_tool(stdio, "pid").invoke({}))
    stdio.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_server_that_never_starts_does_not_hang():
    registry = MCPRegistry().register("slow", [sys.executable, _SERVER_SCRIPT, "--slow-start"], start_timeout=0.5)
    try:
        assert registry.langchain_tools("slow") == []  # logged, not raised
        assert registry.call_tool("echo", {"text": "x"}) == "Tool 'echo' is currently unavailable."
    finally:
        registry.close()


def test_missing_command_is_unavailable_not_fatal():
    registry = MCPRegistry().register("nope", ["definitely-not-a-real-command-xyz"])
    try:
        assert registry.langchain_tools("nope") == []
    finally:
        registry.close()


def test_mcp_json_timeout_is_applied(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": [
        {"name": "a", "command": ["npx", "a"], "timeout": 5},
        {"name": "b", "command": ["npx", "b"]},
        {"name": "h", "url": "https://example.com/mcp", "timeout": 7},
    ]}))
    registry = _load_registry(cfg)
    assert registry._servers["a"].timeout == 5
    assert registry._servers["b"].timeout == 60
    assert registry._servers["h"].timeout == 7


def test_http_query_params_are_appended_to_url():
    from core.src.model.mcp_registry import _http_target
    assert _http_target("https://x/mcp", {"apikey": "k"}, None)() == "https://x/mcp?apikey=k"
    assert _http_target("https://x/mcp?a=1", {"apikey": "k"}, None)() == "https://x/mcp?a=1&apikey=k"


def test_third_party_servers_get_only_listed_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "s3cret")
    monkeypatch.setenv("NEEDED_KEY", "ok")
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": [
        {"name": "npx", "command": ["npx", "a"], "env": ["NEEDED_KEY", "UNSET_KEY"]},
        {"name": "own", "command": ["{python}", "-m", "x"]},
    ]}))
    with patch.object(MCPRegistry, "register", autospec=True, side_effect=lambda self, *a, **k: self) as reg:
        _load_registry(cfg)
    envs = {c.args[1]: c.kwargs["env"] for c in reg.call_args_list}
    assert envs["npx"] == {"NEEDED_KEY": "ok"}
    assert envs["own"]["SECRET_KEY"] == "s3cret"


def test_stdio_subprocess_does_not_inherit_unlisted_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "s3cret")
    script = tmp_path / "envserver.py"
    script.write_text(
        "import os\n"
        "from mcp.server.mcpserver import MCPServer\n"
        "s = MCPServer('e')\n"
        "@s.tool(name='getenv', description='env var')\n"
        "def getenv(key: str) -> str:\n"
        "    return os.environ.get(key, '<unset>')\n"
        "s.run()\n"
    )
    registry = MCPRegistry().register("e", [sys.executable, str(script)], env={"GIVEN": "yes"})
    try:
        assert registry.call_tool("getenv", {"key": "SECRET_KEY"}) == "<unset>"
        assert registry.call_tool("getenv", {"key": "GIVEN"}) == "yes"
    finally:
        registry.close()
