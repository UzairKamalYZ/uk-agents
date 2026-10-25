import unittest
from unittest.mock import patch

from claude_agent_sdk import ResultMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from core.src.model.claude_code_chat import ChatClaudeCode


def _result(text: str, is_error: bool = False) -> ResultMessage:
    return ResultMessage(
        subtype="error" if is_error else "success", duration_ms=1, duration_api_ms=1,
        is_error=is_error, num_turns=1, session_id="s", result=text,
    )


def _fake_query(*results):
    calls = []

    async def fake(prompt, options):
        calls.append((prompt, options))
        for r in results:
            yield r

    return fake, calls


def _convert(from_currency: str, amount: float) -> str:
    """Convert currency."""
    return f"{amount} {from_currency}"


class TestChatClaudeCode(unittest.TestCase):

    def test_should_return_result_text_as_ai_message(self):
        fake, _ = _fake_query(_result("Two flats match."))
        with patch("core.src.model.claude_code_chat.query", fake):
            response = ChatClaudeCode(model="claude-opus-5-5").invoke([HumanMessage(content="2 bed Warsaw")])

        self.assertIsInstance(response, AIMessage)
        self.assertEqual(response.text, "Two flats match.")
        self.assertEqual(response.tool_calls, [])

    def test_should_pass_system_messages_as_system_prompt_and_disable_builtin_tools(self):
        fake, calls = _fake_query(_result("ok"))
        with patch("core.src.model.claude_code_chat.query", fake):
            ChatClaudeCode(model="claude-opus-5-5").invoke([
                SystemMessage(content="You are helpful."),
                SystemMessage(content="Relevant listings: A"),
                HumanMessage(content="hi"),
            ])

        prompt, options = calls[0]
        self.assertEqual(prompt, "hi")
        self.assertEqual(options.system_prompt, "You are helpful.\n\nRelevant listings: A")
        self.assertEqual(options.tools, [])
        self.assertEqual(options.setting_sources, [])
        self.assertEqual(options.env["ANTHROPIC_API_KEY"], "")
        self.assertEqual(options.mcp_servers, {})

    def test_should_replay_history_as_transcript(self):
        fake, calls = _fake_query(_result("ok"))
        with patch("core.src.model.claude_code_chat.query", fake):
            ChatClaudeCode().invoke([
                HumanMessage(content="prior q"),
                AIMessage(content="prior a"),
                HumanMessage(content="new q"),
            ])

        prompt = calls[0][0]
        self.assertIn("User: prior q", prompt)
        self.assertIn("Assistant: prior a", prompt)
        self.assertTrue(prompt.endswith("new q"))

    def test_should_expose_bound_tools_as_allowed_mcp_tools(self):
        tool = StructuredTool.from_function(_convert, name="currency_convert")
        fake, calls = _fake_query(_result("ok"))
        with patch("core.src.model.claude_code_chat.query", fake):
            ChatClaudeCode().bind_tools([tool]).invoke([HumanMessage(content="convert")])

        options = calls[0][1]
        self.assertIn("agent_tools", options.mcp_servers)
        self.assertEqual(options.allowed_tools, ["mcp__agent_tools__currency_convert"])

    def test_should_raise_when_claude_code_returns_error(self):
        fake, _ = _fake_query(_result("Not logged in", is_error=True))
        with patch("core.src.model.claude_code_chat.query", fake):
            with self.assertRaises(RuntimeError):
                ChatClaudeCode().invoke([HumanMessage(content="hi")])

    def test_should_return_schema_instance_using_native_json_schema_output(self):
        from pydantic import BaseModel

        class Route(BaseModel):
            next: str

        result = _result("")
        result.structured_output = {"next": "finance_agent"}
        fake, calls = _fake_query(result)
        with patch("core.src.model.claude_code_chat.query", fake):
            out = ChatClaudeCode().with_structured_output(Route, method="json_schema").invoke(
                [SystemMessage(content="route"), HumanMessage(content="AAPL?")]
            )
        self.assertEqual(out, Route(next="finance_agent"))
        self.assertEqual(calls[0][1].output_format, {"type": "json_schema", "schema": Route.model_json_schema()})

    def test_should_raise_when_structured_output_missing(self):
        from pydantic import BaseModel

        class Route(BaseModel):
            next: str

        fake, _ = _fake_query(_result("plain text"))
        with patch("core.src.model.claude_code_chat.query", fake):
            with self.assertRaises(RuntimeError):
                ChatClaudeCode().with_structured_output(Route).invoke([HumanMessage(content="hi")])

    def test_plain_calls_send_no_output_format(self):
        fake, calls = _fake_query(_result("ok"))
        with patch("core.src.model.claude_code_chat.query", fake):
            ChatClaudeCode().invoke([HumanMessage(content="hi")])
        self.assertIsNone(calls[0][1].output_format)


    def test_sdk_tool_handler_awaits_async_tools(self):
        import asyncio
        calls = []

        async def convert(from_currency: str, amount: float) -> str:
            calls.append("async")
            return f"{amount} {from_currency}"

        tool = StructuredTool.from_function(func=_convert, coroutine=convert, name="convert")
        result = asyncio.run(ChatClaudeCode._to_sdk_tool(tool).handler({"from_currency": "PLN", "amount": 2}))
        self.assertEqual(result, {"content": [{"type": "text", "text": "2.0 PLN"}]})
        self.assertEqual(calls, ["async"])

    def test_sdk_tool_handler_reports_errors(self):
        import asyncio

        def broken(x: int) -> str:
            raise ValueError("bad")

        tool = StructuredTool.from_function(func=broken, name="broken", description="Always fails.")
        result = asyncio.run(ChatClaudeCode._to_sdk_tool(tool).handler({"x": 1}))
        self.assertTrue(result["is_error"])
        self.assertIn("bad", result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
