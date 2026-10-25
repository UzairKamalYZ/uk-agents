import unittest
from unittest.mock import patch, MagicMock

from core.src.base_agent import BaseAgent

_FAKE_PROMPT = "You are a test agent."


class _MinimalAgent(BaseAgent):
    """Concrete subclass that satisfies all abstract hooks with test doubles."""
    def get_system_prompt(self) -> str:
        return _FAKE_PROMPT

    def get_rag_context_manager(self):
        return MagicMock()

    def get_mcp_tools(self) -> list | None:
        return None


class _CustomPromptAgent(BaseAgent):
    """Subclass that overrides the system prompt hook."""
    def get_system_prompt(self) -> str:
        return "custom system prompt"

    def get_rag_context_manager(self):
        return MagicMock()

    def get_mcp_tools(self) -> list | None:
        return None


class _CustomRagAgent(BaseAgent):
    """Subclass that overrides the RAG hook."""
    def get_system_prompt(self) -> str:
        return _FAKE_PROMPT

    def get_rag_context_manager(self):
        return self._custom_rag

    def get_mcp_tools(self) -> list | None:
        return None


class TestBaseAgent(unittest.TestCase):

    def setUp(self):
        self._llm_factory_patcher = patch("core.src.base_agent.create_llm")
        self._graph_patcher = patch("core.src.base_agent.LlmModelGraph")

        self.mock_create_llm = self._llm_factory_patcher.start()
        self.mock_graph_cls = self._graph_patcher.start()

        self.mock_llm = MagicMock()
        self.mock_create_llm.return_value = self.mock_llm
        self.mock_model = MagicMock()
        self.mock_graph_cls.return_value = self.mock_model

        self.agent = _MinimalAgent(session_id="test-session")

    def tearDown(self):
        self._llm_factory_patcher.stop()
        self._graph_patcher.stop()

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_should_create_llm_with_config_values(self):
        self.mock_create_llm.assert_called_once()

    def test_should_give_llm_model_graph_a_persistent_history_store(self):
        from core.src.model.session_manager import SessionManager
        store = self.mock_graph_cls.call_args.kwargs["history_store"]
        self.assertIsInstance(store, SessionManager)

    def test_should_pass_system_prompt_to_llm_model_graph(self):
        _, kwargs = self.mock_graph_cls.call_args
        self.assertEqual(kwargs["system_prompt"], _FAKE_PROMPT)

    def test_should_pass_rag_context_manager_to_llm_model_graph(self):
        _, kwargs = self.mock_graph_cls.call_args
        self.assertIn("rag_context_manager", kwargs)

    def test_should_assign_session_id_when_provided(self):
        self.assertEqual(self.agent.session_id, "test-session")

    def test_should_generate_session_id_when_none_provided(self):
        agent = _MinimalAgent()
        self.assertIsNotNone(agent.session_id)

    # ------------------------------------------------------------------
    # Hook overriding (Template Method)
    # ------------------------------------------------------------------

    def test_should_use_overridden_system_prompt(self):
        agent = _CustomPromptAgent()
        _, kwargs = self.mock_graph_cls.call_args
        self.assertEqual(kwargs["system_prompt"], "custom system prompt")

    def test_should_use_overridden_rag_context_manager(self):
        custom_rag = MagicMock()
        agent = _CustomRagAgent.__new__(_CustomRagAgent)
        agent._custom_rag = custom_rag
        _CustomRagAgent.__init__(agent)
        _, kwargs = self.mock_graph_cls.call_args
        self.assertIs(kwargs["rag_context_manager"], custom_rag)

    # ------------------------------------------------------------------
    # ask()
    # ------------------------------------------------------------------

    def test_should_delegate_ask_to_model(self):
        self.mock_model.ask.return_value = "answer"
        result = self.agent.ask("question")
        self.mock_model.ask.assert_called_once()
        self.assertEqual(result, "answer")

    def test_should_use_instance_session_id_when_none_passed(self):
        self.mock_model.ask.return_value = "ok"
        self.agent.ask("question")
        _, kwargs = self.mock_model.ask.call_args
        self.assertEqual(kwargs.get("session_id"), "test-session")

    def test_should_use_explicit_session_id_over_instance_id(self):
        self.mock_model.ask.return_value = "ok"
        self.agent.ask("question", session_id="override")
        _, kwargs = self.mock_model.ask.call_args
        self.assertEqual(kwargs.get("session_id"), "override")

    def test_should_return_stream_when_stream_true(self):
        self.mock_model.ask_stream.return_value = iter(["tok1", "tok2"])
        result = self.agent.ask("question", stream=True)
        self.mock_model.ask_stream.assert_called_once()
        self.assertEqual(list(result), ["tok1", "tok2"])

    def test_should_delegate_aask_to_model_aask(self):
        import asyncio
        from unittest.mock import AsyncMock
        self.mock_model.aask = AsyncMock(return_value="async answer")
        self.assertEqual(asyncio.run(self.agent.aask("question")), "async answer")
        self.mock_model.aask.assert_awaited_once_with("question", session_id="test-session")
        self.assertEqual(asyncio.run(self.agent.aask("q", session_id="override")), "async answer")
        self.assertEqual(self.mock_model.aask.call_args.kwargs["session_id"], "override")

    # ------------------------------------------------------------------
    # close() / context manager
    # ------------------------------------------------------------------

    def test_should_delegate_close_to_model(self):
        self.agent.close()
        self.mock_model.close.assert_called_once()

    def test_should_close_on_context_manager_exit(self):
        with self.agent:
            pass
        self.mock_model.close.assert_called_once()

    def test_should_not_raise_when_close_called(self):
        try:
            self.agent.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()


class TestPendingHooksDefault(unittest.TestCase):

    def test_pending_hooks_default_noop(self):
        with patch("core.src.base_agent.create_llm"), patch("core.src.base_agent.LlmModelGraph"):
            agent = _MinimalAgent(session_id="s")
        self.assertIsNone(agent.pending_summary("s"))
        self.assertEqual(agent.confirm_pending("s"), "")
        self.assertIsNone(agent.discard_pending("s"))
