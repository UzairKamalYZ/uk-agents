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


def test_owner_only_read_and_defaults_false(tmp_path):
    f = _write(tmp_path, [_spec("a", owner_only=True), _spec("b")])
    entries = {e.name: e for e in load_agents(f)}
    assert entries["a"].owner_only is True
    assert entries["b"].owner_only is False


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
