import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from clients.streamlit.main import StreamlitClient, _get_agent_response, AGENT_API_URL


class TestStreamlitClient(unittest.TestCase):

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    @patch("clients.streamlit.main.subprocess.run")
    def test_should_call_subprocess_run_when_start_is_called(self, mock_run):
        """start() launches the Streamlit server via subprocess.run."""
        StreamlitClient().start()
        mock_run.assert_called_once()

    @patch("clients.streamlit.main.subprocess.run")
    def test_should_use_current_python_executable_when_launching_streamlit(self, mock_run):
        """start() uses sys.executable so the active venv is respected."""
        StreamlitClient().start()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], sys.executable)

    @patch("clients.streamlit.main.subprocess.run")
    def test_should_include_streamlit_run_subcommand_in_command(self, mock_run):
        """start() constructs a 'python -m streamlit run <file>' command."""
        StreamlitClient().start()
        cmd = mock_run.call_args[0][0]
        self.assertIn("streamlit", cmd)
        self.assertIn("run", cmd)

    @patch("clients.streamlit.main.subprocess.run")
    def test_should_point_to_main_py_when_launching_streamlit(self, mock_run):
        """start() passes the path of this main.py file to streamlit run."""
        StreamlitClient().start()
        cmd = mock_run.call_args[0][0]
        self.assertTrue(str(cmd[-1]).endswith("main.py"))

    @patch("clients.streamlit.main.subprocess.run")
    def test_should_pass_check_true_to_subprocess_when_starting(self, mock_run):
        """start() uses check=True so non-zero exit codes raise CalledProcessError."""
        StreamlitClient().start()
        kwargs = mock_run.call_args[1]
        self.assertTrue(kwargs.get("check"))

    # ------------------------------------------------------------------
    # BaseClient conformance
    # ------------------------------------------------------------------

    def test_should_implement_base_client_interface(self):
        """StreamlitClient is a concrete implementation of BaseClient."""
        from clients.base import BaseClient
        self.assertIsInstance(StreamlitClient(), BaseClient)

    @patch("clients.streamlit.main.subprocess.run")
    def test_should_call_stop_via_context_manager_when_block_exits(self, _mock_run):
        """Context-manager exit triggers stop() without raising."""
        with StreamlitClient():
            pass


class TestGetAgentResponse(unittest.TestCase):
    """Unit tests for the _get_agent_response helper."""

    @patch("clients.streamlit.main.st")
    @patch("clients.streamlit.main.requests.get")
    def test_should_return_agent_response_when_api_call_succeeds(self, mock_get, mock_st):
        """_get_agent_response returns the 'response' field from the API JSON."""
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        mock_get.return_value.json.return_value = {"response": "Found 3 flats."}
        mock_get.return_value.raise_for_status = MagicMock()

        result = _get_agent_response("2-bed Warsaw")

        self.assertEqual(result, "Found 3 flats.")

    @patch("clients.streamlit.main.st")
    @patch("clients.streamlit.main.requests.get")
    def test_should_call_api_with_prompt_param_when_querying(self, mock_get, mock_st):
        """_get_agent_response passes the user query as the 'prompt' param."""
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        mock_get.return_value.json.return_value = {"response": "ok"}
        mock_get.return_value.raise_for_status = MagicMock()

        _get_agent_response("studio Kraków")

        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args.args, (AGENT_API_URL,))
        self.assertEqual(mock_get.call_args.kwargs["params"], {"prompt": "studio Kraków"})

    @patch("clients.streamlit.main.Config")
    @patch("clients.streamlit.main.st")
    @patch("clients.streamlit.main.requests.get")
    def test_should_send_the_api_key_only_when_one_is_configured(self, mock_get, mock_st, mock_config):
        """With API_KEY set the REST API rejects requests without the X-API-Key header."""
        mock_st.session_state = {}
        mock_get.return_value.json.return_value = {"response": "ok"}

        mock_config.API_KEY = "secret"
        _get_agent_response("hi")
        self.assertEqual(mock_get.call_args.kwargs["headers"], {"X-API-Key": "secret"})

        mock_config.API_KEY = ""
        _get_agent_response("hi")
        self.assertIsNone(mock_get.call_args.kwargs["headers"])

    @patch("clients.streamlit.main.st")
    @patch("clients.streamlit.main.requests.get")
    def test_should_return_error_message_when_connection_fails(self, mock_get, mock_st):
        """_get_agent_response returns a human-readable error on ConnectionError."""
        import requests as _requests
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        mock_get.side_effect = _requests.exceptions.ConnectionError

        result = _get_agent_response("anything")

        self.assertIn("Error", result)
        self.assertIn("connect", result.lower())

    @patch("clients.streamlit.main.st")
    @patch("clients.streamlit.main.requests.get")
    def test_should_return_fallback_when_response_has_no_response_key(self, mock_get, mock_st):
        """_get_agent_response falls back to a default string when 'response' is absent."""
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        mock_get.return_value.json.return_value = {}
        mock_get.return_value.raise_for_status = MagicMock()

        result = _get_agent_response("anything")

        self.assertEqual(result, "No response from agent.")

    @patch("clients.streamlit.main.st")
    @patch("clients.streamlit.main.requests.get")
    def test_should_continue_the_conversation_when_a_session_id_was_returned(self, mock_get, mock_st):
        """The session id from the first reply is sent with every later message."""
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.session_state = {}
        mock_get.return_value.json.return_value = {"response": "ok", "session_id": "abc"}
        mock_get.return_value.raise_for_status = MagicMock()

        _get_agent_response("2-bed Warsaw")
        _get_agent_response("cheaper ones?")

        self.assertEqual(mock_get.call_args_list[0].kwargs["params"], {"prompt": "2-bed Warsaw"})
        self.assertEqual(mock_get.call_args_list[1].kwargs["params"],
                         {"prompt": "cheaper ones?", "session_id": "abc"})
        self.assertEqual(mock_st.session_state["session_id"], "abc")


if __name__ == "__main__":
    unittest.main()


class TestChatPage(unittest.TestCase):
    """The page itself, run headless with Streamlit's AppTest (no server, no REST API)."""

    def _page(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).parents[1] / "streamlit" / "main.py"))

    def _reply(self, text):
        response = MagicMock()
        response.json.return_value = {"response": text, "session_id": "abc"}
        return response

    def test_welcome_screen_offers_a_starter_prompt_for_each_agent_with_an_example(self):
        import json
        agents = json.loads((Path(__file__).parents[2] / "agents.json").read_text())["agents"]
        examples = [a["example"] for a in agents if a.get("enabled", True) and a.get("example")]
        at = self._page().run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual([b.label for b in at.main.button], examples)

    def test_usage_disclaimer_is_shown_on_the_welcome_screen_and_during_a_chat(self):
        from clients.streamlit.main import _DISCLAIMER
        with patch("requests.get", return_value=self._reply("Hello.")):
            at = self._page().run(timeout=30)
            self.assertIn(_DISCLAIMER, [c.value for c in at.caption])
            at.chat_input[0].set_value("hi").run(timeout=30)
        self.assertIn(_DISCLAIMER, [c.value for c in at.caption])
        self.assertIn("Uzair Kamal", _DISCLAIMER)

    def test_reloaded_page_resumes_the_chat_until_new_chat(self):
        with patch("requests.get", return_value=self._reply("Hello again.")):
            at = self._page().run(timeout=30)
            at.chat_input[0].set_value("hi").run(timeout=30)
        reloaded = self._page().run(timeout=30)   # a new browser session
        self.assertEqual([m.markdown[0].value for m in reloaded.chat_message], ["hi", "Hello again."])
        self.assertEqual(reloaded.session_state["session_id"], "abc")
        reloaded.sidebar.button[0].click().run(timeout=30)   # "New chat"
        self.assertEqual(list(self._page().run(timeout=30).chat_message), [])

    def test_starter_prompt_is_asked_and_new_chat_clears_the_conversation(self):
        with patch("requests.get", return_value=self._reply("I can do many things.")) as mock_get:
            at = self._page().run(timeout=30)
            starter = at.main.button[0]
            starter.click().run(timeout=30)
            self.assertEqual(mock_get.call_args.kwargs["params"], {"prompt": starter.label})
            self.assertEqual([m.markdown[0].value for m in at.chat_message],
                             [starter.label, "I can do many things."])
            self.assertEqual(len(at.main.button), 0)   # the welcome screen is gone

            at.sidebar.button[0].click().run(timeout=30)
        self.assertEqual(len(at.chat_message), 0)
        self.assertNotIn("session_id", at.session_state)

    def test_connection_error_is_shown_as_an_error_message(self):
        import requests as _requests
        with patch("requests.get", side_effect=_requests.exceptions.ConnectionError):
            at = self._page().run(timeout=30)
            at.chat_input[0].set_value("hi").run(timeout=30)
        self.assertIn("Could not connect", at.chat_message[1].error[0].value)
