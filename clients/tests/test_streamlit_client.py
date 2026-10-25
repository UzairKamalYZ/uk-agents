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

        mock_get.assert_called_once_with(AGENT_API_URL, params={"prompt": "studio Kraków"})

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
