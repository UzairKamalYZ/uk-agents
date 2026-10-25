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
