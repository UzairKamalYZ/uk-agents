"""Local stdio MCP server exposing the expense ledger as tools."""

from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from core.src.config.config import Config

from .ledger import Expense, ExpenseLedger
from .replies import (
    format_breakdown,
    format_logged,
    format_nothing,
    format_question,
    format_totals,
    format_undone,
)

server = MCPServer("expenses")


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


def get_expense_breakdown(month: str | None = None, *, ledger: ExpenseLedger | None = None) -> str:
    """Return spending per place (shop, merchant) for one month, largest first.

    month is YYYY-MM; omit it (or pass empty) for the current month.
    """
    if month is None or not str(month).strip():
        first = _today().replace(day=1)
    else:
        try:
            first = date.fromisoformat(f"{str(month).strip()}-01")
        except ValueError:
            return f"Invalid month '{month}', expected YYYY-MM."
    return format_breakdown(first, _ledger(ledger).breakdown(first.year, first.month))


# Thin wrappers keep the `ledger` kwarg out of the tool schemas.
@server.tool(name="add_expense", description=add_expense.__doc__)
def _add_expense(place: str, amount: float, date: str | None = None) -> str:
    return add_expense(place, amount, date)


@server.tool(name="propose_expense", description=propose_expense.__doc__)
def _propose_expense(place: str, problem: str, amount: float | None = None, date: str | None = None) -> str:
    return propose_expense(place, problem, amount, date)


@server.tool(name="undo_last_expense", description=undo_last_expense.__doc__)
def _undo_last_expense() -> str:
    return undo_last_expense()


@server.tool(name="get_expense_totals", description=get_expense_totals.__doc__)
def _get_expense_totals() -> str:
    return get_expense_totals()


@server.tool(name="get_expense_breakdown", description=get_expense_breakdown.__doc__)
def _get_expense_breakdown(month: str | None = None) -> str:
    return get_expense_breakdown(month)


if __name__ == "__main__":
    server.run()
