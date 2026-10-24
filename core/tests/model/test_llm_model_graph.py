import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from core.src.model.llm_model_graph import LlmModelGraph, State

_FAKE_SYSTEM_PROMPT = "You are a helpful property agent."


class TestLlmModelGraph(unittest.TestCase):

    def setUp(self):
        self._load_file_patcher = patch("core.src.model.llm_model_graph.load_prompt")
        self._embedder_patcher = patch("core.src.model.llm_model_graph.Embedder")
        self._rag_patcher = patch("core.src.model.llm_model_graph.RagContextManager")
        self._mcp_patcher = patch("core.src.model.llm_model_graph._mcp")

        mock_load_file = self._load_file_patcher.start()
        self._embedder_patcher.start()
        self._rag_patcher.start()
        self.mock_mcp = self._mcp_patcher.start()

        mock_load_file.return_value = _FAKE_SYSTEM_PROMPT
        self.mock_mcp.langchain_tools.return_value = []

        self.mock_llm = MagicMock()
        self.model = LlmModelGraph(self.mock_llm)
        self.model.graph = MagicMock(name="CompiledGraphMock")

    def tearDown(self):
        self._load_file_patcher.stop()
        self._embedder_patcher.stop()
        self._rag_patcher.stop()
        self._mcp_patcher.stop()

    def _hist(self, session_id: str) -> list:
        return self.model._history.get(self.model.agent_name, session_id)

    def _make_state(self, **kwargs) -> State:
        defaults: State = {
            "user_prompt": "find a flat",
            "context": "",
            "answer": "",
            "session_history": [],
        }
        defaults.update(kwargs)
        return defaults

    def _stream_chunk(self, content: str, node: str):
        chunk = AIMessageChunk(content=content)
        return (chunk, {"langgraph_node": node})

    # ------------------------------------------------------------------
    # ask()
    # ------------------------------------------------------------------

    def test_should_return_answer_string_when_ask_is_called(self):
        self.model.graph.invoke.return_value = {"answer": "3 listings found in Warsaw."}
        result = self.model.ask("3-bed apartment in Warsaw")
        self.assertEqual(result, "3 listings found in Warsaw.")

    def test_should_invoke_graph_once_when_ask_is_called(self):
        self.model.graph.invoke.return_value = {"answer": "ok"}
        self.model.ask("3-bed apartment in Warsaw")
        self.model.graph.invoke.assert_called_once()

    def test_should_build_history_from_query_and_answer_when_ask_is_called(self):
        SESSION = "sess-history-ask"
        self.model.graph.invoke.return_value = {"answer": "3 listings found."}
        self.model.ask("find me a flat", session_id=SESSION)
        hist = self._hist(SESSION)
        self.assertEqual(len(hist), 2)
        self.assertIsInstance(hist[0], HumanMessage)
        self.assertIsInstance(hist[1], AIMessage)
        self.assertEqual(hist[0].content, "find me a flat")
        self.assertEqual(hist[1].content, "3 listings found.")

    def test_should_accumulate_history_across_turns_when_asking(self):
        SESSION = "sess-accum-ask"
        self.model.graph.invoke.side_effect = [{"answer": "answer 1"}, {"answer": "answer 2"}]
        self.model.ask("query 1", session_id=SESSION)
        self.model.ask("query 2", session_id=SESSION)
        self.assertEqual(len(self._hist(SESSION)), 4)

    def test_should_pass_correct_initial_state_when_ask_is_called(self):
        self.model.graph.invoke.return_value = {"answer": "ok"}
        self.model.ask("find a studio", session_id="s1")
        state = self.model.graph.invoke.call_args[0][0]
        self.assertEqual(state["user_prompt"], "find a studio")
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")

    def test_should_isolate_history_between_different_sessions(self):
        self.model.graph.invoke.return_value = {"answer": "ok"}
        self.model.ask("q1", session_id="session-A")
        self.model.ask("q2", session_id="session-B")
        self.assertEqual(self._hist("session-A")[0].content, "q1")
        self.assertEqual(self._hist("session-B")[0].content, "q2")

    def test_should_share_history_within_same_session_across_turns(self):
        SESSION = "sess-share"
        self.model.graph.invoke.side_effect = [{"answer": "answer 1"}, {"answer": "answer 2"}]
        self.model.ask("turn 1", session_id=SESSION)
        self.model.ask("turn 2", session_id=SESSION)
        second_state = self.model.graph.invoke.call_args_list[1][0][0]
        history = second_state["session_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].content, "turn 1")
        self.assertEqual(history[1].content, "answer 1")

    def test_should_generate_session_id_when_none_provided_to_ask(self):
        self.model.graph.invoke.return_value = {"answer": "ok"}
        self.model.ask("any query")
        self.assertEqual(len(self.model._history.histories), 1)

    # ------------------------------------------------------------------
    # ask_stream()
    # ------------------------------------------------------------------

    def test_should_yield_full_answer_when_streaming(self):
        """ask_stream() delegates to ask() and yields the complete answer as one token."""
        self.model.graph.invoke.return_value = {"answer": "Here are 3 flats in Warsaw."}
        tokens = list(self.model.ask_stream("find a flat"))
        self.assertEqual(tokens, ["Here are 3 flats in Warsaw."])

    def test_should_store_ai_response_in_history_when_streaming(self):
        """ask_stream() records both the user query and the full AI answer in history."""
        SESSION = "sess-stream-hist"
        self.model.graph.invoke.return_value = {"answer": "Found 2 listings."}
        list(self.model.ask_stream("my query", session_id=SESSION))
        hist = self._hist(SESSION)
        self.assertEqual(len(hist), 2)
        self.assertIsInstance(hist[0], HumanMessage)
        self.assertIsInstance(hist[1], AIMessage)
        self.assertEqual(hist[0].content, "my query")
        self.assertEqual(hist[1].content, "Found 2 listings.")

    # ------------------------------------------------------------------
    # _retrieve_node()
    # ------------------------------------------------------------------

    def test_should_query_rag_with_user_prompt(self):
        """_retrieve_node() passes user_prompt directly to the vector store."""
        self.model.rag_context_manager.get_context.return_value = "listing A\nlisting B"
        state = self._make_state(user_prompt="2 bed Warsaw")
        self.model._retrieve_node(state)
        self.model.rag_context_manager.get_context.assert_called_once_with("2 bed Warsaw")

    def test_should_return_context_from_rag(self):
        self.model.rag_context_manager.get_context.return_value = "listing A"
        result = self.model._retrieve_node(self._make_state())
        self.assertEqual(result["context"], "listing A")

    def test_should_return_empty_context_when_no_matches(self):
        self.model.rag_context_manager.get_context.return_value = ""
        result = self.model._retrieve_node(self._make_state())
        self.assertEqual(result["context"], "")

    # ------------------------------------------------------------------
    # _generate_node()
    # ------------------------------------------------------------------

    def test_should_return_answer_from_llm(self):
        self.mock_llm.invoke.return_value = AIMessage(content="Here are listings in Warsaw.")
        result = self.model._generate_node(self._make_state(context="some context"))
        self.assertEqual(result["answer"], "Here are listings in Warsaw.")

    def test_should_return_only_text_blocks_when_response_has_thinking(self):
        self.mock_llm.invoke.return_value = AIMessage(content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Two flats match."},
        ])
        result = self.model._generate_node(self._make_state(context="some context"))
        self.assertEqual(result["answer"], "Two flats match.")

    def test_should_answer_without_context(self):
        self.mock_llm.invoke.return_value = AIMessage(content="General answer.")
        result = self.model._generate_node(self._make_state(context=""))
        self.assertEqual(result["answer"], "General answer.")

    def test_should_use_user_prompt_as_question(self):
        """_generate_node() sends the original user_prompt to the LLM, not a reformulation."""
        self.mock_llm.invoke.return_value = AIMessage(content="answer")
        state = self._make_state(user_prompt="cheap 2 bed Warsaw")
        self.model._generate_node(state)
        call_messages = self.mock_llm.invoke.call_args[0][0]
        human_messages = [m for m in call_messages if isinstance(m, HumanMessage)]
        self.assertEqual(human_messages[-1].content, "cheap 2 bed Warsaw")

    def test_should_include_session_history_in_prompt(self):
        history = [HumanMessage(content="prior q"), AIMessage(content="prior a")]
        self.mock_llm.invoke.return_value = AIMessage(content="contextual answer")
        result = self.model._generate_node(self._make_state(session_history=history))
        self.assertEqual(result["answer"], "contextual answer")

    def test_should_execute_tool_call_loop_when_llm_requests_tool(self):
        tool_call = {"id": "call_1", "name": "currency_convert", "args": {"from_currency": "PLN", "to_currency": "USD", "amount": 1.0}}
        self.mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="1 PLN = 0.25 USD"),
        ]
        mock_tool = MagicMock()
        mock_tool.name = "currency_convert"
        mock_tool.invoke.return_value = "0.25 USD"
        self.model._tools = [mock_tool]

        result = self.model._generate_node(self._make_state(context="Flat 3000 PLN/month"))

        mock_tool.invoke.assert_called_once()
        self.assertEqual(self.mock_llm.invoke.call_count, 2)
        self.assertEqual(result["answer"], "1 PLN = 0.25 USD")

    # ------------------------------------------------------------------
    # _initial_state()
    # ------------------------------------------------------------------

    def test_should_set_user_prompt_in_initial_state(self):
        state = self.model._initial_state("find me a studio")
        self.assertEqual(state["user_prompt"], "find me a studio")

    def test_should_have_empty_derived_fields_in_initial_state(self):
        state = self.model._initial_state("anything")
        self.assertEqual(state["context"], "")
        self.assertEqual(state["answer"], "")
        self.assertEqual(state["session_history"], [])
        self.assertNotIn("reformulated_question", state)
        self.assertNotIn("needs_search", state)

    def test_should_copy_history_so_mutations_do_not_affect_original(self):
        history = [HumanMessage(content="h1")]
        state = self.model._initial_state("q", history=history)
        state["session_history"].append(AIMessage(content="extra"))
        self.assertEqual(len(history), 1)

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    @patch("core.src.model.llm_model_graph.close_mcp")
    def test_should_delegate_to_close_mcp_when_close_is_called(self, mock_close_mcp):
        self.model.close()
        mock_close_mcp.assert_called_once()

    def test_should_not_raise_when_close_is_called(self):
        try:
            self.model.close()
        except Exception as exc:
            self.fail(f"close() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
