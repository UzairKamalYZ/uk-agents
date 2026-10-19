import sqlite3
from datetime import date, datetime
from decimal import Decimal

from agents.expenditure.ledger import DB_FILENAME, Expense, ExpenseLedger
from agents.expenditure.replies import format_breakdown, format_logged, format_question, format_totals


def _e(d, place, amount):
    return Expense(d, place, Decimal(amount))


def test_add_stores_row_in_database(tmp_path):
    ExpenseLedger(tmp_path).add(date(2026, 10, 18), "Colruyt", 90)
    assert (tmp_path / DB_FILENAME).exists()
    assert ExpenseLedger(tmp_path).expenses() == [_e(date(2026, 10, 18), "Colruyt", "90.00")]


def test_records_survive_a_restart(tmp_path):
    ExpenseLedger(tmp_path).add(date(2026, 10, 18), "Colruyt", 90)
    ExpenseLedger(tmp_path).set_pending(_e(date(2026, 10, 18), "Aldi", "5"), "check")
    ExpenseLedger(tmp_path).log_input("Colruyt 90")
    fresh = ExpenseLedger(tmp_path)  # a new process would build a new ledger on the same folder
    assert fresh.expenses() == [_e(date(2026, 10, 18), "Colruyt", "90.00")]
    assert fresh.get_pending()[1] == "check"
    assert [t for _, t in fresh.inputs()] == ["Colruyt 90"]


def test_totals_prev_and_current(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    assert ledger.totals(date(2026, 10, 18)) == (Decimal("0"), Decimal("0"))
    ledger.add(date(2026, 9, 5), "Rent", 1000)
    ledger.add(date(2026, 10, 2), "Shop", 30)
    assert ledger.totals(date(2026, 10, 18)) == (Decimal("1000.00"), Decimal("30.00"))


def test_totals_across_year_boundary(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 12, 31), "Shop", 50)
    assert ledger.totals(date(2027, 1, 1))[0] == Decimal("50.00")


def test_monthly_totals_view(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 9, 5), "Rent", 1000)
    ledger.add(date(2026, 10, 2), "Shop", 30)
    ledger.add(date(2026, 10, 3), "Shop", 12.5)
    with sqlite3.connect(tmp_path / DB_FILENAME) as conn:
        rows = conn.execute("SELECT month, entries, total_eur FROM monthly_totals ORDER BY month").fetchall()
    assert rows == [("2026-09", 1, 1000.0), ("2026-10", 2, 42.5)]


def test_backdated_expense_and_undo(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "A", 10)
    ledger.add(date(2026, 9, 30), "B", 20)
    undone = ledger.undo_last()
    assert undone == _e(date(2026, 9, 30), "B", "20.00")
    assert ledger.expenses() == [_e(date(2026, 10, 1), "A", "10.00")]
    assert ledger.totals(date(2026, 10, 18)) == (Decimal("0"), Decimal("10.00"))


def test_undo_keeps_the_row_for_the_record(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "A", 5)
    ledger.undo_last()
    with sqlite3.connect(tmp_path / DB_FILENAME) as conn:
        assert conn.execute("SELECT place, deleted_at IS NOT NULL FROM expenses").fetchall() == [("A", 1)]


def test_undo_on_empty_returns_none(tmp_path):
    assert ExpenseLedger(tmp_path).undo_last() is None


def test_repeated_undo_walks_back(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "A", 5)
    ledger.add(date(2026, 10, 1), "A", 5)
    ledger.add(date(2026, 10, 2), "B", 7)
    assert ledger.undo_last().place == "B"
    assert ledger.undo_last().place == "A"
    assert ledger.expenses() == [_e(date(2026, 10, 1), "A", "5.00")]


def test_amount_rounding_and_comma(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    assert ledger.add(date(2026, 10, 1), "A", 12.499999).amount == Decimal("12.50")
    assert ledger.add(date(2026, 10, 1), "A", "12,5").amount == Decimal("12.50")
    assert ledger.totals(date(2026, 10, 1))[1] == Decimal("25.00")


def test_place_whitespace_is_collapsed(tmp_path):
    assert ExpenseLedger(tmp_path).add(date(2026, 10, 1), "  Albert\n Heijn ", 1).place == "Albert Heijn"


def test_pending_roundtrip_and_confirm(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    assert ledger.confirm_pending() is None
    e = _e(date(2026, 10, 18), "Colruyt", "19.00")
    ledger.set_pending(e, "amount unclear")
    assert ledger.get_pending() == (e, "amount unclear")
    assert ledger.confirm_pending() == e
    assert ledger.get_pending() is None
    assert ledger.expenses() == [e]


def test_set_pending_replaces_previous(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.set_pending(_e(date(2026, 10, 18), "A", "1"), "x")
    ledger.set_pending(_e(date(2026, 10, 18), "B", "2"), "y")
    assert ledger.get_pending() == (_e(date(2026, 10, 18), "B", "2.00"), "y")


def test_replies_format():
    assert format_totals(Decimal("1000"), Decimal("12.5")) == (
        "📊 Totals\nLast month: €1,000.00\nThis month: €12.50\n\n— Expenditure Tracker"
    )
    e = Expense(date(2026, 10, 18), "Colruyt", Decimal("19.00"))
    assert format_question(e, "amount unclear") == (
        "⚠️ Please check this expense\n📍 Colruyt\n💶 €19.00\n📅 Sun 18 Oct 2026\n❓ amount unclear\n\n"
        "Reply yes to save, no to discard, or send the corrected version.\n\n— Expenditure Tracker"
    )


def test_logged_reply_format():
    e = Expense(date(2026, 10, 18), "Colruyt", Decimal("90"))
    assert format_logged(e, Decimal("1000"), Decimal("120")) == (
        "✅ Expense saved\n📍 Colruyt\n💶 €90.00\n📅 Sun 18 Oct 2026\n\n"
        "📊 Totals\nLast month: €1,000.00\nThis month: €120.00\n\n— Expenditure Tracker"
    )


def test_question_with_unknown_amount():
    e = Expense(date(2026, 10, 18), "Colruyt", None)
    assert "💶 amount unknown\n" in format_question(e, "amount missing")


def test_breakdown_reply_format():
    rows = [("Colruyt", Decimal("150")), ("Albert Heijn", Decimal("50"))]
    assert format_breakdown(date(2026, 10, 1), rows) == (
        "🧾 Spending by place — October 2026\n• Colruyt: €150.00\n• Albert Heijn: €50.00\n\n"
        "Total: €200.00\n\n— Expenditure Tracker"
    )
    assert "No expenses logged." in format_breakdown(date(2026, 10, 1), [])


def test_breakdown_groups_places_case_insensitively_largest_first(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "Albert Heijn", 50)
    ledger.add(date(2026, 10, 2), "Colruyt", 90)
    ledger.add(date(2026, 10, 3), "colruyt", 60)
    ledger.add(date(2026, 9, 30), "Colruyt", 999)
    assert ledger.breakdown(2026, 10) == [("Colruyt", Decimal("150.00")), ("Albert Heijn", Decimal("50.00"))]
    assert ledger.breakdown(2026, 11) == []


def test_log_input_records_one_row_per_message(tmp_path):
    ledger = ExpenseLedger(tmp_path / "new")
    ledger.log_input("Colruyt 90\neuros", datetime(2026, 10, 18, 9, 5, 1))
    ledger.log_input("undo last", datetime(2026, 10, 18, 9, 6, 0))
    assert ledger.inputs() == [
        ("2026-10-18 09:05:01", "Colruyt 90 euros"),
        ("2026-10-18 09:06:00", "undo last"),
    ]


def test_pending_roundtrip_with_unknown_amount(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    e = Expense(date(2026, 10, 18), "Colruyt", None)
    ledger.set_pending(e, "amount missing")
    assert ledger.get_pending() == (e, "amount missing")


def test_corrupt_database_reads_no_pending(tmp_path):
    (tmp_path / DB_FILENAME).write_text("this is not a database")
    assert ExpenseLedger(tmp_path).get_pending() is None
