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
    reply = mcp_server.add_expense("Colruyt", 90, "2026-10-18", ledger=ledger)
    assert reply.startswith("✅ Expense saved\n📍 Colruyt\n💶 €90.00\n📅 Sun 18 Oct 2026")
    assert "This month: €90.00" in reply


def test_propose_expense_stores_pending_and_writes_nothing(ledger, tmp_path):
    reply = mcp_server.propose_expense("Colruyt", "amount looks high", 900, "2026-10-18", ledger=ledger)
    assert reply.startswith("⚠️ Please check this expense\n📍 Colruyt")
    assert ledger.get_pending() is not None
    assert ledger.expenses() == []


def test_add_expense_clears_pending(ledger):
    ledger.set_pending(Expense(TODAY, "Colruyt", Decimal("900")), "x")
    mcp_server.add_expense("Colruyt", 90, "2026-10-18", ledger=ledger)
    assert ledger.get_pending() is None


def test_undo_last_expense_empty(ledger):
    assert mcp_server.undo_last_expense(ledger=ledger) == "ℹ️ Nothing to undo.\n\n— Expenditure Tracker"


def test_undo_last_expense_removes(ledger):
    mcp_server.add_expense("Colruyt", 90, "2026-10-18", ledger=ledger)
    assert mcp_server.undo_last_expense(ledger=ledger).startswith("↩️ Expense removed\n📍 Colruyt\n💶 €90.00")


def test_get_expense_totals(ledger):
    mcp_server.add_expense("Colruyt", 90, "2026-10-18", ledger=ledger)
    assert mcp_server.get_expense_totals(ledger=ledger) == (
        "📊 Totals\nLast month: €0.00\nThis month: €90.00\n\n— Expenditure Tracker"
    )


def test_get_expense_breakdown(ledger):
    mcp_server.add_expense("Colruyt", 90, "2026-10-18", ledger=ledger)
    mcp_server.add_expense("Delhaize", 20, "2026-09-02", ledger=ledger)
    assert "• Colruyt: €90.00" in mcp_server.get_expense_breakdown(ledger=ledger)
    september = mcp_server.get_expense_breakdown("2026-09", ledger=ledger)
    assert september.startswith("🧾 Spending by place — September 2026\n• Delhaize: €20.00")
    assert mcp_server.get_expense_breakdown("Sept", ledger=ledger) == (
        "Invalid month 'Sept', expected current, previous or YYYY-MM."
    )


@pytest.mark.parametrize("month", ["previous", "Last"])
def test_breakdown_previous_month_largest_first(ledger, month):
    mcp_server.add_expense("Delhaize", 20, "2026-09-02", ledger=ledger)
    mcp_server.add_expense("Colruyt", 70, "2026-09-10", ledger=ledger)
    mcp_server.add_expense("delhaize", 15, "2026-09-20", ledger=ledger)
    mcp_server.add_expense("Aldi", 500, "2026-10-01", ledger=ledger)
    assert mcp_server.get_expense_breakdown(month, ledger=ledger) == (
        "🧾 Spending by place — September 2026\n• Colruyt: €70.00\n• Delhaize: €35.00\n\n"
        "Total: €105.00\n\n— Expenditure Tracker"
    )


def test_breakdown_previous_month_across_year_boundary(ledger):
    mcp_server.add_expense("Colruyt", 10, "2025-12-31", ledger=ledger)
    with patch.object(mcp_server, "_today", return_value=date(2026, 1, 15)):
        assert "December 2025\n• Colruyt: €10.00" in mcp_server.get_expense_breakdown("previous", ledger=ledger)


def test_invalid_date(ledger):
    assert mcp_server.add_expense("X", 5, "yesterday", ledger=ledger) == "Invalid date 'yesterday', expected YYYY-MM-DD."
    assert mcp_server.propose_expense("X", "p", 5, "yesterday", ledger=ledger) == "Invalid date 'yesterday', expected YYYY-MM-DD."


@pytest.mark.parametrize("amount", ["abc", 0, -5, "nan", "inf", 0.001, "0,004"])
def test_invalid_amount(ledger, amount):
    expected = f"Invalid amount '{amount}', expected a positive number."
    assert mcp_server.add_expense("X", amount, "2026-10-18", ledger=ledger) == expected
    assert mcp_server.propose_expense("X", "p", amount, "2026-10-18", ledger=ledger) == expected
    assert ledger.get_pending() is None


@pytest.mark.parametrize("amount", [None, ""])
def test_add_expense_requires_amount(ledger, amount):
    assert mcp_server.add_expense("X", amount, "2026-10-18", ledger=ledger) == (
        f"Invalid amount '{amount}', expected a positive number."
    )


def test_smallest_valid_amount_after_rounding(ledger):
    assert "💶 €0.01" in mcp_server.add_expense("X", 0.005, "2026-10-18", ledger=ledger)


@pytest.mark.parametrize("dateless", [None, ""])
def test_dateless_add_expense_uses_today(ledger, tmp_path, dateless):
    reply = mcp_server.add_expense("Colruyt", 90, dateless, ledger=ledger)
    assert "📅 Sun 18 Oct 2026" in reply
    assert ledger.expenses() == [Expense(TODAY, "Colruyt", Decimal("90.00"))]


def test_dateless_propose_expense_uses_today(ledger):
    mcp_server.propose_expense("Colruyt", "looks high", 900, ledger=ledger)
    assert ledger.get_pending()[0].date == TODAY


@pytest.mark.parametrize("amount", [None, ""])
def test_propose_without_amount_stores_unknown(ledger, tmp_path, amount):
    reply = mcp_server.propose_expense("Colruyt", "amount missing", amount, "2026-10-18", ledger=ledger)
    assert "💶 amount unknown\n" in reply
    assert ledger.get_pending() == (Expense(TODAY, "Colruyt", None), "amount missing")
    assert ledger.expenses() == []


def test_server_lists_six_tools():
    with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"EXPENSES_DIR": d}):
        proc = MCPProcess("expenses", [sys.executable, "-m", "agents.expenditure.mcp_server"])
        try:
            tools = proc.list_tools()
        finally:
            proc.close()
    assert {t["name"] for t in tools} == {
        "add_expense", "propose_expense", "undo_last_expense", "get_expense_totals", "get_expense_breakdown",
        "get_expense_summary",
    }
    assert all("ledger" not in str(t.get("inputSchema")) for t in tools)
    required = {t["name"]: set(t["inputSchema"].get("required", [])) for t in tools}
    assert required["add_expense"] == {"place", "amount"}
    assert required["propose_expense"] == {"place", "problem"}


def test_expense_summary_totals_and_top_five(ledger):
    ledger.add(date(2026, 9, 3), "Rent", 900)
    for place, amount in [("Aldi", 10), ("Colruyt", 150), ("Lidl", 7.5), ("IKEA", 230),
                          ("Bakery", 3), ("Shell", 60), ("Colruyt", 42)]:
        ledger.add(date(2026, 10, 2), place, amount)
    assert mcp_server.get_expense_summary(ledger=ledger) == (
        "📊 Totals\nLast month: €900.00\nThis month: €502.50\n\n"
        "🏆 Top 5 expenses — October 2026\n"
        "1. IKEA — €230.00 (02 Oct)\n2. Colruyt — €150.00 (02 Oct)\n3. Shell — €60.00 (02 Oct)\n"
        "4. Colruyt — €42.00 (02 Oct)\n5. Aldi — €10.00 (02 Oct)\n\n— Expenditure Tracker"
    )


def test_expense_summary_with_no_expenses_this_month(ledger):
    ledger.add(date(2026, 9, 3), "Rent", 900)
    reply = mcp_server.get_expense_summary(ledger=ledger)
    assert "Last month: €900.00\nThis month: €0.00" in reply
    assert "No expenses logged this month yet." in reply
