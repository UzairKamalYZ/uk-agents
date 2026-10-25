from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agents.fbr_tax import runner as runner_mod
from agents.fbr_tax.agent import BUSY, FAILED, FbrTaxAgent
from agents.fbr_tax.runner import SESSION_TTL, SKILL, FilingRunner, FilingSessions
from agents.fbr_tax.tracker import SERVER_NAME, FilingTracker

NOW = datetime(2026, 10, 19, 20, 0)


@pytest.fixture
def project(tmp_path):
    skill = tmp_path / "fbr" / ".claude" / "skills" / SKILL
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: fbr-iris-filing\n---\n")
    return tmp_path / "fbr"


@pytest.fixture
def agent(tmp_path, project):
    with patch("core.src.base_agent.create_llm"), \
         patch("core.src.base_agent.LlmModelGraph"), \
         patch("agents.fbr_tax.agent.Config") as cfg:
        cfg.FBR_TAX_DIR = str(project)
        cfg.SESSION_DB_FILE = str(tmp_path / "sessions.db")
        cfg.FBR_TAX_DATA_DIR = str(tmp_path / "data")
        cfg.LLM_MODEL_NAME = "claude-opus-5-5"
        cfg.LLM_EFFORT = "medium"
        cfg.FBR_TAX_MAX_TURNS = 50
        a = FbrTaxAgent(session_id="s")
        a._now = lambda: NOW
        a._runner.run = MagicMock(return_value=("Please log in to IRIS.", "cc-1"))
        yield a


def test_first_message_starts_a_session_and_next_resumes_it(agent):
    assert agent.ask("file my FBR return", session_id="chat") == "Please log in to IRIS."
    agent._runner.run.assert_called_with("file my FBR return", None)
    agent._runner.run.return_value = ("Which mode?", "cc-1")
    assert agent.ask("I'm logged in", session_id="chat") == "Which mode?"
    agent._runner.run.assert_called_with("I'm logged in", "cc-1")


def test_sessions_are_per_chat(agent):
    agent.ask("file my return", session_id="a")
    agent.ask("file my return", session_id="b")
    assert agent._runner.run.call_args_list[1].args == ("file my return", None)


def test_stale_session_is_not_resumed(agent):
    agent.ask("file my return", session_id="chat")
    agent._now = lambda: NOW + SESSION_TTL + timedelta(minutes=1)
    agent.ask("file my return", session_id="chat")
    agent._runner.run.assert_called_with("file my return", None)


def test_failure_clears_the_session_without_retrying(agent):
    agent.ask("file my return", session_id="chat")
    agent._runner.run.side_effect = RuntimeError("boom")
    assert agent.ask("yes", session_id="chat") == FAILED
    assert agent._runner.run.call_count == 2           # no replay of "yes" in a fresh session
    agent._runner.run.side_effect = None
    agent.ask("resume the filing", session_id="chat")
    agent._runner.run.assert_called_with("resume the filing", None)


def test_busy_while_another_turn_runs(agent):
    agent._busy.acquire()
    try:
        assert agent.ask("status?", session_id="chat") == BUSY
    finally:
        agent._busy.release()
    agent._runner.run.assert_not_called()


def test_missing_skill_is_reported(agent, project):
    (project / ".claude" / "skills" / SKILL / "SKILL.md").unlink()
    assert "not found" in agent.ask("file my return", session_id="chat")
    agent._runner.run.assert_not_called()


def test_stream_yields_the_whole_reply(agent):
    assert list(agent.ask("hi", stream=True, session_id="chat")) == ["Please log in to IRIS."]


def test_aask_runs_the_claude_code_session_not_the_plain_llm(agent):
    # The orchestrator calls aask(); it must drive the browser session like ask() does
    import asyncio
    assert asyncio.run(agent.aask("is my return filed?", session_id="chat")) == "Please log in to IRIS."
    agent._runner.run.assert_called_once_with("is my return filed?", None)
    agent.model.aask.assert_not_called()


def test_session_store_round_trip(tmp_path):
    store = FilingSessions(tmp_path / "s.db")
    assert store.get("chat", NOW) is None
    store.set("chat", "cc-1", NOW)
    store.set("chat", "cc-2", NOW)
    assert store.get("chat", NOW) == "cc-2"
    store.clear("chat")
    assert store.get("chat", NOW) is None


def test_options_lock_down_tools_and_enable_skill_and_chrome(project):
    opts = FilingRunner(project, "extra rules", FilingTracker(project / "f.db"), model="m", max_turns=7)._options("cc-1")
    assert opts.cwd == str(project)
    assert opts.resume == "cc-1"
    assert opts.skills == [SKILL, "claude-in-chrome"]
    assert opts.extra_args == {"chrome": None}
    assert "Bash" not in opts.tools and "WebFetch" not in opts.tools
    # skills= only grants Skill(name); without the Skill tool itself the model writes the call as text
    assert "Skill" in opts.tools
    assert opts.permission_mode == "dontAsk"
    assert "Write(/snapshots/**)" in opts.allowed_tools
    assert not any(t in ("Write", "Edit") for t in opts.allowed_tools)   # writes are path-scoped only
    assert opts.system_prompt["append"] == "extra rules"
    assert opts.setting_sources == ["project"]
    assert SERVER_NAME in opts.mcp_servers and f"mcp__{SERVER_NAME}" in opts.allowed_tools


def test_run_returns_result_and_session(project):
    result = runner_mod.ResultMessage.__new__(runner_mod.ResultMessage)
    result.__dict__.update(result="done", session_id="cc-9", is_error=False, num_turns=3, duration_ms=10)

    async def fake_query(prompt, options):
        yield result

    with patch.object(runner_mod, "query", fake_query):
        assert FilingRunner(project, "x", FilingTracker(project / "f.db")).run("hi") == ("done", "cc-9")


def test_run_raises_on_error_result(project):
    result = runner_mod.ResultMessage.__new__(runner_mod.ResultMessage)
    result.__dict__.update(result="bad", session_id="cc-9", is_error=True, errors=["x"], num_turns=1, duration_ms=1)

    async def fake_query(prompt, options):
        yield result

    with patch.object(runner_mod, "query", fake_query), pytest.raises(RuntimeError):
        FilingRunner(project, "x", FilingTracker(project / "f.db")).run("hi")
