import os
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.expenditure import mcp_server
from agents.expenditure.ledger import Expense, ExpenseLedger
from core.src.model.mcp_registry import MCPProcess

TODAY = date(2026, 10, 18)


@pytest.fixture
def ledger(tmp_path):
    with patch.object(mcp_server, "_today", return_value=TODAY):
        yield ExpenseLedger(tmp_path)


def test_add_expense_returns_logged_reply(ledger):
    reply = mcp_server.add_expense("2026-10-18", "Colruyt", 90, ledger=ledger)
    assert reply.startswith("Logged €90 at Colruyt (2026-10-18).")
    assert "prev_month: " in reply


def test_propose_expense_stores_pending_and_writes_nothing(ledger, tmp_path):
    reply = mcp_server.propose_expense("2026-10-18", "Colruyt", 900, "amount looks high", ledger=ledger)
    assert reply.startswith("I understood: Colruyt")
    assert ledger.get_pending() is not None
    assert not (tmp_path / "2026-10.txt").exists()


def test_add_expense_clears_pending(ledger):
    ledger.set_pending(Expense(TODAY, "Colruyt", Decimal("900")), "x")
    mcp_server.add_expense("2026-10-18", "Colruyt", 90, ledger=ledger)
    assert ledger.get_pending() is None


def test_undo_last_expense_empty(ledger):
    assert mcp_server.undo_last_expense(ledger=ledger) == "Nothing to undo."


def test_undo_last_expense_removes(ledger):
    mcp_server.add_expense("2026-10-18", "Colruyt", 90, ledger=ledger)
    assert mcp_server.undo_last_expense(ledger=ledger).startswith("Removed €90 at Colruyt")


def test_get_expense_totals(ledger):
    mcp_server.add_expense("2026-10-18", "Colruyt", 90, ledger=ledger)
    assert mcp_server.get_expense_totals(ledger=ledger) == "prev_month: €0, current_month: €90"


def test_invalid_date(ledger):
    assert mcp_server.add_expense("yesterday", "X", 5, ledger=ledger) == "Invalid date 'yesterday', expected YYYY-MM-DD."
    assert mcp_server.propose_expense("yesterday", "X", 5, "p", ledger=ledger) == "Invalid date 'yesterday', expected YYYY-MM-DD."


@pytest.mark.parametrize("amount", ["abc", 0, -5, "nan", "inf"])
def test_invalid_amount(ledger, amount):
    expected = f"Invalid amount '{amount}', expected a positive number."
    assert mcp_server.add_expense("2026-10-18", "X", amount, ledger=ledger) == expected
    assert mcp_server.propose_expense("2026-10-18", "X", amount, "p", ledger=ledger) == expected
    assert ledger.get_pending() is None


def test_server_lists_four_tools():
    with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"EXPENSES_DIR": d}):
        proc = MCPProcess("expenses", [sys.executable, "-m", "agents.expenditure.mcp_server"])
        try:
            tools = proc.list_tools()
        finally:
            proc.close()
    assert {t["name"] for t in tools} == {"add_expense", "propose_expense", "undo_last_expense", "get_expense_totals"}
    assert all("ledger" not in str(t.get("inputSchema")) for t in tools)
