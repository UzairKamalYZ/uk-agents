from unittest.mock import MagicMock, patch

from orchestrator.tracing import tag_current_trace


def test_tags_the_enclosing_run():
    run = MagicMock()
    with patch("orchestrator.tracing.get_current_run_tree", return_value=run):
        tag_current_trace(["agent:finance_agent"], {"agents": ["finance_agent"]})
    run.add_tags.assert_called_once_with(["agent:finance_agent"])
    run.add_metadata.assert_called_once_with({"agents": ["finance_agent"]})


def test_no_op_without_tracing():
    with patch("orchestrator.tracing.get_current_run_tree", return_value=None):
        tag_current_trace(["agent:x"], {})  # must not raise
