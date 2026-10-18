import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.src.model.mcp_tools import _mcp, close_mcp, _load_registry
from core.src.model.mcp_registry import MCPProcess, MCPRegistry


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
        self.assertIn("@easysolutions906/mcp-finance", server._command)

    @patch("core.src.model.mcp_tools._mcp")
    def test_close_mcp_delegates_to_registry(self, mock_registry):
        """close_mcp() calls close() on the shared registry."""
        close_mcp()
        mock_registry.close.assert_called_once()


class TestMCPProcess(unittest.TestCase):
    """Unit tests for the MCPProcess subprocess manager (no real npx calls)."""

    def setUp(self):
        self.mcp = MCPProcess("test", ["npx", "--yes", "@acme/test-mcp"])

    def _make_mock_process(self, responses: list[str]):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = responses
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.flush = MagicMock()
        return mock_proc

    @patch("core.src.model.mcp_registry.subprocess.Popen")
    def test_should_start_subprocess_on_first_call(self, mock_popen):
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        tool_resp = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "200 USD"}]}})
        mock_popen.return_value = self._make_mock_process([init_resp, tool_resp])

        result = self.mcp.call_tool("some_tool", {"arg": "value"})

        mock_popen.assert_called_once()
        self.assertEqual(result, "200 USD")

    @patch("core.src.model.mcp_registry.subprocess.Popen")
    def test_should_reuse_process_on_second_call(self, mock_popen):
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        resp1 = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "ok"}]}})
        resp2 = json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "ok2"}]}})
        mock_popen.return_value = self._make_mock_process([init_resp, resp1, resp2])

        self.mcp.call_tool("tool_a", {})
        self.mcp.call_tool("tool_b", {})

        self.assertEqual(mock_popen.call_count, 1)

    @patch("core.src.model.mcp_registry.subprocess.Popen")
    def test_should_join_multiple_text_content_items(self, mock_popen):
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        tool_resp = json.dumps({
            "jsonrpc": "2.0", "id": 2,
            "result": {"content": [
                {"type": "text", "text": "Line one"},
                {"type": "text", "text": "Line two"},
            ]},
        })
        mock_popen.return_value = self._make_mock_process([init_resp, tool_resp])

        result = self.mcp.call_tool("tool", {})

        self.assertIn("Line one", result)
        self.assertIn("Line two", result)

    @patch("core.src.model.mcp_registry.subprocess.Popen")
    def test_should_return_error_string_when_subprocess_raises(self, mock_popen):
        mock_popen.side_effect = FileNotFoundError("npx not found")

        result = self.mcp.call_tool("tool", {})

        self.assertIn("unavailable", result.lower())
        self.assertNotIn("FileNotFoundError", result)

    def test_should_terminate_process_when_close_is_called(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        self.mcp._process = mock_proc

        self.mcp.close()

        mock_proc.terminate.assert_called_once()
        self.assertIsNone(self.mcp._process)

    def test_should_not_raise_when_close_called_before_any_tool_call(self):
        try:
            self.mcp.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")

    def test_should_not_raise_when_close_called_twice(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        self.mcp._process = mock_proc

        self.mcp.close()
        try:
            self.mcp.close()
        except Exception as exc:
            self.fail(f"Second close() raised unexpectedly: {exc}")


class TestMCPRegistry(unittest.TestCase):
    """Unit tests for MCPRegistry routing and lifecycle."""

    def test_register_returns_self_for_chaining(self):
        registry = MCPRegistry()
        result = registry.register("svc", ["cmd"])
        self.assertIs(result, registry)

    def test_register_stores_server(self):
        registry = MCPRegistry()
        registry.register("svc", ["cmd"])
        self.assertIn("svc", registry._servers)
        self.assertIsInstance(registry._servers["svc"], MCPProcess)

    @patch.object(MCPProcess, "call_tool", return_value="ok result")
    def test_call_tool_routes_to_registered_server(self, mock_call):
        registry = MCPRegistry()
        registry.register("svc", ["cmd"])

        result = registry.call_tool("some_tool", {"x": 1})

        mock_call.assert_called_once_with("some_tool", {"x": 1})
        self.assertEqual(result, "ok result")

    @patch.object(MCPProcess, "close")
    def test_close_all_terminates_every_server(self, mock_close):
        registry = MCPRegistry()
        registry.register("a", ["cmd-a"])
        registry.register("b", ["cmd-b"])

        registry.close_all()

        self.assertEqual(mock_close.call_count, 2)

    @patch.object(MCPProcess, "close")
    def test_close_is_alias_for_close_all(self, mock_close):
        registry = MCPRegistry()
        registry.register("svc", ["cmd"])

        registry.close()

        mock_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()


def test_python_placeholder_resolves_to_sys_executable():
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "mcp.json"
        cfg.write_text(json.dumps({"servers": [{"name": "x", "command": ["{python}", "-m", "x"]}]}))
        registry = _load_registry(cfg)
    assert registry._servers["x"]._command == [sys.executable, "-m", "x"]


def test_langchain_tools_unknown_server_returns_empty():
    registry = MCPRegistry()
    server = MagicMock()
    server.list_tools.return_value = [{"name": "t"}]
    registry._servers["a"] = server
    assert registry.langchain_tools(server_name="missing") == []
    server.list_tools.assert_not_called()
