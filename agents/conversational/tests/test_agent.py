import json
from unittest.mock import patch

from agents.conversational.agent import ConversationalAgent, agent_catalog


def _write(tmp_path, agents):
    f = tmp_path / "agents.json"
    f.write_text(json.dumps({"agents": agents}))
    return f


def test_catalog_lists_enabled_agents_with_summary(tmp_path):
    f = _write(tmp_path, [
        {"name": "a", "display_name": "Agent A", "summary": "does A", "description": "route A"},
        {"name": "b", "summary": "", "description": "route B"},
        {"name": "c", "display_name": "Off", "summary": "x", "enabled": False},
    ])
    assert agent_catalog(f) == "- Agent A: does A\n- b: route B"


def test_catalog_survives_a_broken_file(tmp_path):
    f = tmp_path / "agents.json"
    f.write_text("{not json")
    assert agent_catalog(f) == "(the agent list is unavailable right now)"


def test_real_registry_lists_every_enabled_agent():
    from pathlib import Path
    root = Path(__file__).parents[3]
    specs = [s for s in json.loads((root / "agents.json").read_text())["agents"] if s.get("enabled", True)]
    catalog = agent_catalog(root / "agents.json")
    for s in specs:
        assert s["summary"], f"{s['name']} has no summary"
        assert f"- {s['display_name']}: {s['summary']}" in catalog


def test_system_prompt_contains_the_catalog(tmp_path):
    f = _write(tmp_path, [{"name": "x", "display_name": "FBR-PK Tax Filing", "summary": "files tax"}])
    with patch("core.src.base_agent.create_llm"), \
         patch("core.src.base_agent.LlmModelGraph") as graph, \
         patch("agents.conversational.agent.Config") as cfg:
        cfg.AGENTS_FILE = str(f)
        ConversationalAgent(session_id="s")
    prompt = graph.call_args.kwargs["system_prompt"]
    assert "- FBR-PK Tax Filing: files tax" in prompt
    assert "{agents}" not in prompt
