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
        Expense(date(2026, 10, 18), "Colruyt", Decimal(amount)), "amount unclear"
    )


def test_confirm_pending_logs_and_returns_totals(agent, tmp_path):
    _pending(tmp_path)
    reply = agent.confirm_pending("s")
    assert reply.startswith("Logged €19 at Colruyt")
    assert "2026-10-18 | Colruyt | 19.00" in (tmp_path / "2026-10.txt").read_text()


def test_confirm_without_pending(agent):
    assert agent.confirm_pending("s") == "Nothing to confirm."


def test_pending_summary_and_discard(agent, tmp_path):
    assert agent.pending_summary("s") is None
    _pending(tmp_path)
    assert agent.pending_summary("s") == "Colruyt, €19, 2026-10-18 (problem: amount unclear)"
    agent.discard_pending("s")
    assert agent.pending_summary("s") is None


def test_uses_expenses_mcp_server(agent):
    with patch("agents.expenditure.agent._mcp") as mcp:
        agent.get_mcp_tools()
    mcp.langchain_tools.assert_called_once_with(server_name="expenses")
