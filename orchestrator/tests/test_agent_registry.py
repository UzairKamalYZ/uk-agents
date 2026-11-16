import json

from orchestrator.agent_registry import load_agents


class FakeAgent:
    def __init__(self, rag_enabled=False, llm_provider="anthropic", llm_model="m"):
        self.rag_enabled = rag_enabled


def _write(tmp_path, specs):
    f = tmp_path / "agents.json"
    f.write_text(json.dumps({"agents": specs}))
    return f


def _spec(name, **extra):
    return {"name": name, "description": "d", "class": f"{__name__}.FakeAgent", **extra}


def test_agents_have_no_owner_only_flag(tmp_path):
    entry = load_agents(_write(tmp_path, [_spec("a", owner_only=True)]))[0]  # an old key is ignored
    assert not hasattr(entry, "owner_only")


def test_display_name_and_remarks_read_with_fallback_label(tmp_path):
    f = _write(tmp_path, [
        _spec("expenditure_tracker", display_name="Expenditure Tracker", remarks=["Counting coins…"]),
        _spec("finance_agent"),
    ])
    entries = {e.name: e for e in load_agents(f)}
    assert entries["expenditure_tracker"].label == "Expenditure Tracker"
    assert entries["expenditure_tracker"].remarks == ("Counting coins…",)
    assert entries["finance_agent"].label == "Finance Agent"
    assert entries["finance_agent"].remarks == ()


def test_an_agent_that_cannot_start_is_left_out(tmp_path):
    f = _write(tmp_path, [
        {"name": "missing", "description": "d", "class": "no.such.module.Agent"},
        {"name": "broken", "description": "d", "class": f"{__name__}.BrokenAgent"},
        _spec("good"),
    ])
    assert [e.name for e in load_agents(f)] == ["good"]


class BrokenAgent:
    def __init__(self, **kwargs):
        raise ValueError("PINECONE_API_KEY is not set")


def test_every_agent_in_the_real_registry_starts(tmp_path):
    """agents.json as shipped: every enabled agent's class imports and constructs."""
    from pathlib import Path
    from unittest.mock import patch

    from core.src.config.config import Config
    from core.src.model.mcp_tools import _mcp

    registry = Path(__file__).parents[2] / "agents.json"
    enabled = [s["name"] for s in json.loads(registry.read_text())["agents"] if s.get("enabled", True)]
    with patch("core.src.base_agent.create_llm"), \
         patch("core.src.base_agent.Embedder"), \
         patch.object(_mcp, "langchain_tools", return_value=[]), \
         patch.object(Config, "EXPENSES_DIR", str(tmp_path / "expenses")), \
         patch.object(Config, "FBR_TAX_DATA_DIR", str(tmp_path / "fbr")), \
         patch.object(Config, "SESSION_DB_FILE", str(tmp_path / "sessions.db")):
        entries = load_agents(registry)
    assert [e.name for e in entries] == enabled
    assert all(e.instance.model.system_prompt.strip() for e in entries)
