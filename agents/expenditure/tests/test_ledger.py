import json
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from agents.expenditure.ledger import Expense, ExpenseLedger
from agents.expenditure.replies import format_breakdown, format_logged, format_question, format_totals


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


def test_log_input_appends_one_line_per_message(tmp_path):
    ledger = ExpenseLedger(tmp_path / "new")
    ledger.log_input("Colruyt 90\neuros", datetime(2026, 10, 18, 9, 5, 1))
    ledger.log_input("undo last", datetime(2026, 10, 18, 9, 6, 0))
    assert (tmp_path / "new" / "inputs.log").read_text() == (
        "2026-10-18 09:05:01 | Colruyt 90 euros\n2026-10-18 09:06:00 | undo last\n"
    )


def test_pending_roundtrip_with_unknown_amount(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    e = Expense(date(2026, 10, 18), "Colruyt", None)
    ledger.set_pending(e, "amount missing")
    assert ledger.get_pending() == (e, "amount missing")


@pytest.mark.parametrize("content", ["", "{not json", '{"expense": {}}', "[]"])
def test_corrupt_pending_treated_as_none(tmp_path, content):
    (tmp_path / "pending.json").write_text(content)
    assert ExpenseLedger(tmp_path).get_pending() is None


@pytest.mark.parametrize("content", ["", "[{trunc", '{"a": 1}'])
def test_corrupt_history_treated_as_empty(tmp_path, content):
    (tmp_path / "history.json").write_text(content)
    ledger = ExpenseLedger(tmp_path)
    assert ledger.undo_last() is None
    ledger.add(date(2026, 10, 1), "A", 5)
    assert ledger.undo_last().place == "A"


def test_json_writes_are_atomic(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    ledger.add(date(2026, 10, 1), "A", 5)
    with patch("agents.expenditure.ledger.os.replace", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            ledger.add(date(2026, 10, 2), "B", 6)
    # The old history file is intact and no temp files are left behind
    assert len(json.loads((tmp_path / "history.json").read_text())) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [".lock", "2026-10.txt", "history.json"]


def _add_many(root, place, n):
    ledger = ExpenseLedger(root)
    for _ in range(n):
        ledger.add(date(2026, 10, 18), place, 1)


def test_concurrent_adds_from_threads_and_processes_lose_nothing(tmp_path):
    import multiprocessing
    import threading

    procs = [multiprocessing.get_context("spawn").Process(target=_add_many, args=(tmp_path, f"P{i}", 25))
             for i in range(2)]
    threads = [threading.Thread(target=_add_many, args=(tmp_path, f"T{i}", 25)) for i in range(2)]
    for w in procs + threads:
        w.start()
    for w in procs + threads:
        w.join()
    assert len((tmp_path / "2026-10.txt").read_text().splitlines()) == 100
    assert len(json.loads((tmp_path / "history.json").read_text())) == 100


def test_lock_is_reentrant(tmp_path):
    ledger = ExpenseLedger(tmp_path)
    with ledger.locked(), ledger.locked():
        ledger.add(date(2026, 10, 18), "X", 1)
    ledger.add(date(2026, 10, 18), "Y", 1)  # fully released afterwards
    assert len((tmp_path / "2026-10.txt").read_text().splitlines()) == 2
