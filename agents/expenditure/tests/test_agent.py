from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest

from agents.expenditure.agent import ExpenditureAgent
from agents.expenditure.ledger import Expense, ExpenseLedger


@pytest.fixture
def agent(tmp_path):
    with patch("core.src.base_agent.create_llm"), \
         patch("core.src.base_agent.LlmModelGraph"), \
         patch("agents.expenditure.agent.Config") as cfg, \
         patch("agents.expenditure.agent._mcp"):
        cfg.EXPENSES_DIR = str(tmp_path)
        a = ExpenditureAgent(session_id="s")
        a._today = lambda: date(2026, 10, 18)
        yield a


def _pending(tmp_path, amount="19"):
    ExpenseLedger(tmp_path).set_pending(
        Expense(date(2026, 10, 18), "Colruyt", None if amount is None else Decimal(amount)), "amount unclear"
    )


def test_confirm_pending_logs_and_returns_totals(agent, tmp_path):
    _pending(tmp_path)
    reply = agent.confirm_pending("s")
    assert reply.startswith("✅ Expense saved\n📍 Colruyt\n💶 €19.00")
    assert "2026-10-18 | Colruyt | 19.00" in (tmp_path / "2026-10.txt").read_text()


def test_confirm_without_pending(agent):
    assert agent.confirm_pending("s") == "ℹ️ Nothing to confirm.\n\n— Expenditure Tracker"


def test_pending_summary_and_discard(agent, tmp_path):
    assert agent.pending_summary("s") is None
    _pending(tmp_path)
    assert agent.pending_summary("s") == "Colruyt, €19.00, 2026-10-18 (problem: amount unclear)"
    agent.discard_pending("s")
    assert agent.pending_summary("s") is None


def test_uses_expenses_mcp_server(agent):
    with patch("agents.expenditure.agent._mcp") as mcp:
        agent.get_mcp_tools()
    mcp.langchain_tools.assert_called_once_with(server_name="expenses")


def test_pending_summary_with_unknown_amount(agent, tmp_path):
    _pending(tmp_path, amount=None)
    assert agent.pending_summary("s") == "Colruyt, amount unknown, 2026-10-18 (problem: amount unclear)"


def test_confirm_with_unknown_amount_keeps_pending_and_saves_nothing(agent, tmp_path):
    _pending(tmp_path, amount=None)
    assert agent.confirm_pending("s").startswith("⚠️ I still need the amount\n📍 Colruyt\n📅 Sun 18 Oct 2026")
    assert not (tmp_path / "2026-10.txt").exists()
    assert agent.pending_summary("s") is not None


def test_ask_prefixes_today(agent):
    agent.model.ask.return_value = "ok"
    assert agent.ask("Colruyt 90", session_id="x") == "ok"
    agent.model.ask.assert_called_once_with("Today is 2026-10-18 (Sunday).\n\nColruyt 90", session_id="x")


def test_aask_prefixes_today_and_logs_input(agent, tmp_path):
    import asyncio
    from unittest.mock import AsyncMock
    agent.model.aask = AsyncMock(return_value="ok")
    assert asyncio.run(agent.aask("Colruyt 90", session_id="x")) == "ok"
    agent.model.aask.assert_awaited_once_with("Today is 2026-10-18 (Sunday).\n\nColruyt 90", session_id="x")
    assert (tmp_path / "inputs.log").read_text().endswith("| Colruyt 90\n")


def test_every_input_is_logged(agent, tmp_path):
    agent.model.ask.return_value = "ok"
    agent.ask("Colruyt 90", session_id="x")
    _pending(tmp_path)
    agent.confirm_pending("s")
    _pending(tmp_path)
    agent.discard_pending("s")
    entries = [line.split(" | ", 1)[1] for line in (tmp_path / "inputs.log").read_text().splitlines()]
    assert entries == ["Colruyt 90", "[confirmed pending expense]", "[discarded pending expense]"]


def test_input_log_failure_does_not_block_reply(agent):
    agent.model.ask.return_value = "ok"
    with patch.object(agent._ledger, "log_input", side_effect=OSError("disk full")):
        assert agent.ask("Colruyt 90") == "ok"


def test_ask_without_tools_refuses_without_calling_model(tmp_path, caplog):
    with patch("core.src.base_agent.create_llm"), \
         patch("core.src.base_agent.LlmModelGraph"), \
         patch("agents.expenditure.agent.Config") as cfg, \
         patch("agents.expenditure.agent._mcp") as mcp:
        cfg.EXPENSES_DIR = str(tmp_path)
        mcp.langchain_tools.return_value = []
        with caplog.at_level("ERROR"):
            a = ExpenditureAgent(session_id="s")
    assert "expenses" in caplog.text
    assert a.ask("Colruyt 90").startswith("⚠️ Expense tracking is unavailable right now — nothing was saved.")
    a.model.ask.assert_not_called()
    import asyncio
    assert asyncio.run(a.aask("Colruyt 90")).startswith("⚠️ Expense tracking is unavailable")
    a.model.aask.assert_not_called()
    assert "Colruyt 90" in (tmp_path / "inputs.log").read_text()
