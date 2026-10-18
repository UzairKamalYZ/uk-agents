import unittest
from unittest.mock import patch

from core.src.model.llm_factory import create_llm


class TestCreateLlm(unittest.TestCase):

    @patch("core.src.model.llm_factory.ChatAnthropic")
    def test_should_create_chat_anthropic_for_anthropic_provider(self, mock_chat):
        llm = create_llm("anthropic", "claude-opus-5-5")

        self.assertIs(llm, mock_chat.return_value)
        kwargs = mock_chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5-5")
        self.assertIn("effort", kwargs["output_config"])
        self.assertEqual(kwargs["model_kwargs"], {"fallbacks": "default"})

    @patch("core.src.model.llm_factory.Config.ANTHROPIC_API", False)
    @patch("core.src.model.llm_factory.ChatClaudeCode")
    @patch("core.src.model.llm_factory.ChatAnthropic")
    def test_should_create_claude_code_model_when_anthropic_api_flag_is_false(self, mock_api, mock_cc):
        llm = create_llm("anthropic", "claude-opus-5-5")

        self.assertIs(llm, mock_cc.return_value)
        self.assertEqual(mock_cc.call_args.kwargs["model"], "claude-opus-5-5")
        mock_api.assert_not_called()

    def test_should_raise_for_unsupported_provider(self):
        with self.assertRaises(ValueError):
            create_llm("ollama", "llama3.2")


if __name__ == "__main__":
    unittest.main()
