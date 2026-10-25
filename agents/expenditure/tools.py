"""The expense ledger as LangChain tools, called in-process by ExpenditureAgent."""

import logging
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from core.src.config.config import Config

from .ledger import Expense, ExpenseLedger
from .replies import (
    format_breakdown,
    format_logged,
    format_nothing,
    format_question,
    format_summary,
    format_totals,
    format_undone,
)

logger = logging.getLogger(__name__)


def _today() -> date:
    return date.today()


def _ledger(ledger: ExpenseLedger | None) -> ExpenseLedger:
    return ledger or ExpenseLedger(Path(Config.EXPENSES_DIR))


def _parse_date(date_str: str | None) -> date | str:
    """Return the date (empty/None → today) or an error string."""
    if date_str is None or not str(date_str).strip():
        return _today()
    try:
        return date.fromisoformat(str(date_str).strip())
    except ValueError:
        return f"Invalid date '{date_str}', expected YYYY-MM-DD."


def _parse_amount(amount) -> Decimal | str:
    """Return the amount rounded to cents, or an error string. Validated after rounding."""
    try:
        parsed = Decimal(str(amount).replace(",", "."))
        valid = parsed.is_finite()
        if valid:
            parsed = ExpenseLedger._to_amount(parsed)
            valid = parsed > 0
    except (InvalidOperation, ValueError):
        valid = False
    if not valid:
        return f"Invalid amount '{amount}', expected a positive number."
    return parsed


def add_expense(place: str, amount: float, date: str | None = None,
                *, ledger: ExpenseLedger | None = None) -> str:
    """Save a clearly understood expense.

    amount is required and must be positive. date is YYYY-MM-DD; omit it (or pass empty) for today.
    """
    parsed_date = _parse_date(date)
    if isinstance(parsed_date, str):
        return parsed_date
    parsed_amount = _parse_amount(amount)
    if isinstance(parsed_amount, str):
        return parsed_amount
    led = _ledger(ledger)
    expense = led.add(parsed_date, place, parsed_amount)
    led.clear_pending()
    prev, cur = led.totals(_today())
    return format_logged(expense, prev, cur)


def propose_expense(place: str, problem: str, amount: float | None = None, date: str | None = None,
                    *, ledger: ExpenseLedger | None = None) -> str:
    """Use instead of saving when anything about the input looks wrong, uncertain, or missing.

    Stores the expense as pending and returns a confirmation question. problem says what looks wrong.
    amount may be omitted when it is missing or unintelligible. date is YYYY-MM-DD; omit it
    (or pass empty) for today.
    """
    parsed_date = _parse_date(date)
    if isinstance(parsed_date, str):
        return parsed_date
    parsed_amount = None
    if amount is not None and str(amount).strip():
        parsed_amount = _parse_amount(amount)
        if isinstance(parsed_amount, str):
            return parsed_amount
    led = _ledger(ledger)
    expense = Expense(parsed_date, ExpenseLedger._clean_place(place), parsed_amount)
    led.set_pending(expense, problem)
    return format_question(expense, problem)


def undo_last_expense(*, ledger: ExpenseLedger | None = None) -> str:
    """Remove the most recently saved expense."""
    led = _ledger(ledger)
    expense = led.undo_last()
    if expense is None:
        return format_nothing("undo")
    prev, cur = led.totals(_today())
    return format_undone(expense, prev, cur)


def get_expense_totals(*, ledger: ExpenseLedger | None = None) -> str:
    """Return previous-month and current-month expense totals."""
    prev, cur = _ledger(ledger).totals(_today())
    return format_totals(prev, cur)


def get_expense_summary(*, ledger: ExpenseLedger | None = None) -> str:
    """Spending overview: this month's and last month's totals, plus this month's 5 largest expenses.

    Use for any general "how much have I spent / what are my expenses" question.
    """
    led = _ledger(ledger)
    today = _today()
    prev, cur = led.totals(today)
    return format_summary(today, prev, cur, led.top_expenses(today.year, today.month, limit=5))


def get_expense_breakdown(month: str | None = None, *, ledger: ExpenseLedger | None = None) -> str:
    """Summarise one month's spending: total per place (shop, merchant), largest first.

    month is "current" (this month), "previous" (last month) or YYYY-MM; omit it (or pass empty)
    for the current month.
    """
    first = _parse_month(month)
    if isinstance(first, str):
        return first
    return format_breakdown(first, _ledger(ledger).breakdown(first.year, first.month))


def _parse_month(month: str | None) -> date | str:
    """Return the first day of the month, or an error string."""
    key = "" if month is None else str(month).strip().lower()
    this_month = _today().replace(day=1)
    if key in ("", "current", "this"):
        return this_month
    if key in ("previous", "last"):
        return (this_month - timedelta(days=1)).replace(day=1)
    try:
        return date.fromisoformat(f"{key}-01")
    except ValueError:
        return f"Invalid month '{month}', expected current, previous or YYYY-MM."


def _call(tool: str, fn, *args) -> str:
    """Run one tool, logging its arguments and the first line of its reply (errors included)."""
    logger.info("tool %s called with %s", tool, args)
    try:
        reply = fn(*args)
    except Exception:
        logger.exception("tool %s failed", tool)
        raise
    logger.info("tool %s → %s", tool, reply.splitlines()[0] if reply else "")
    return reply


# Thin wrappers keep the `ledger` kwarg out of the tool schemas Claude sees.
def _add_expense(place: str, amount: float, date: str | None = None) -> str:
    return _call("add_expense", add_expense, place, amount, date)


def _propose_expense(place: str, problem: str, amount: float | None = None, date: str | None = None) -> str:
    return _call("propose_expense", propose_expense, place, problem, amount, date)


def _undo_last_expense() -> str:
    return _call("undo_last_expense", undo_last_expense)


def _get_expense_totals() -> str:
    return _call("get_expense_totals", get_expense_totals)


def _get_expense_summary() -> str:
    return _call("get_expense_summary", get_expense_summary)


def _get_expense_breakdown(month: str | None = None) -> str:
    return _call("get_expense_breakdown", get_expense_breakdown, month)


_TOOLS = (
    ("add_expense", _add_expense, add_expense),
    ("propose_expense", _propose_expense, propose_expense),
    ("undo_last_expense", _undo_last_expense, undo_last_expense),
    ("get_expense_totals", _get_expense_totals, get_expense_totals),
    ("get_expense_summary", _get_expense_summary, get_expense_summary),
    ("get_expense_breakdown", _get_expense_breakdown, get_expense_breakdown),
)


def build_tools() -> list[BaseTool]:
    """The expense tools for ExpenditureAgent; each tool's description is its function's docstring."""
    return [StructuredTool.from_function(func=wrapper, name=name, description=impl.__doc__)
            for name, wrapper, impl in _TOOLS]
