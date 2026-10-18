from datetime import date
from decimal import Decimal

from agents.expenditure.ledger import Expense, ExpenseLedger
from agents.expenditure.replies import format_question, format_totals


def test_add_writes_month_file_line(tmp_path):
    ExpenseLedger(tmp_path).add(date(2026, 10, 18), "Colruyt", 90)
    assert (tmp_path / "2026-10.txt").read_text() == "2026-10-18 | Colruyt | 90.00\n"


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


def test_backdated_expense_and_undo_hit_previous_month_file(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "A", 10)
    ledger.add(date(2026, 9, 30), "B", 20)
    undone = ledger.undo_last()
    assert undone.amount == Decimal("20.00") and undone.place == "B"
    assert "20.00" not in (tmp_path / "2026-09.txt").read_text()
    assert (tmp_path / "2026-10.txt").read_text() == "2026-10-01 | A | 10.00\n"


def test_undo_on_empty_returns_none(tmp_path):
    assert ExpenseLedger(tmp_path).undo_last() is None


def test_undo_removes_only_last_duplicate(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "A", 5)
    ledger.add(date(2026, 10, 1), "A", 5)
    ledger.undo_last()
    assert (tmp_path / "2026-10.txt").read_text().count("A | 5.00") == 1


def test_amount_rounding_and_comma(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    assert ledger.add(date(2026, 10, 1), "A", 12.499999).amount == Decimal("12.50")
    assert ledger.add(date(2026, 10, 1), "A", "12,5").amount == Decimal("12.50")


def test_pipe_in_place_is_sanitised(tmp_path):
    ExpenseLedger(tmp_path).add(date(2026, 10, 1), "A|B", 1)
    line = (tmp_path / "2026-10.txt").read_text().strip()
    assert "A/B" in line
    assert len(line.split(" | ")) == 3


def test_pending_roundtrip_and_confirm(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    assert ledger.confirm_pending() is None
    e = Expense(date(2026, 10, 18), "Colruyt", Decimal("19.00"))
    ledger.set_pending(e, "amount unclear")
    assert ledger.get_pending() == (e, "amount unclear")
    assert ledger.confirm_pending() == e
    assert ledger.get_pending() is None
    assert "2026-10-18 | Colruyt | 19.00" in (tmp_path / "2026-10.txt").read_text()


def test_replies_format():
    assert format_totals(Decimal("1000"), Decimal("12.5")) == "prev_month: €1000, current_month: €12.50"
    e = Expense(date(2026, 10, 18), "Colruyt", Decimal("19.00"))
    assert format_question(e) == (
        "I understood: Colruyt, €19, 2026-10-18. Reply yes to save, or send the corrected version."
    )
