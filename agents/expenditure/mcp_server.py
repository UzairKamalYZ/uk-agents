"""Local stdio MCP server exposing the expense ledger as tools."""

from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from core.src.config.config import Config

from .ledger import Expense, ExpenseLedger
from .replies import format_logged, format_question, format_totals, format_undone

server = MCPServer("expenses")


def _today() -> date:
    return date.today()


def _ledger(ledger: ExpenseLedger | None) -> ExpenseLedger:
    return ledger or ExpenseLedger(Path(Config.EXPENSES_DIR))


def _parse(date_str: str, amount) -> tuple[date, Decimal] | str:
    """Return (date, amount) or an error string."""
    try:
        parsed_date = date.fromisoformat(str(date_str))
    except ValueError:
        return f"Invalid date '{date_str}', expected YYYY-MM-DD."
    try:
        parsed_amount = Decimal(str(amount).replace(",", "."))
        valid = parsed_amount.is_finite() and parsed_amount > 0
    except InvalidOperation:
        valid = False
    if not valid:
        return f"Invalid amount '{amount}', expected a positive number."
    return parsed_date, parsed_amount


def add_expense(date: str, place: str, amount: float, *, ledger: ExpenseLedger | None = None) -> str:
    """Save a clearly understood expense. date is YYYY-MM-DD."""
    parsed = _parse(date, amount)
    if isinstance(parsed, str):
        return parsed
    led = _ledger(ledger)
    expense = led.add(parsed[0], place, parsed[1])
    led.clear_pending()
    prev, cur = led.totals(_today())
    return format_logged(expense, prev, cur)


def propose_expense(date: str, place: str, amount: float, problem: str,
                    *, ledger: ExpenseLedger | None = None) -> str:
    """Use instead of saving when anything about the input looks wrong or uncertain.

    Stores the expense as pending and returns a confirmation question. date is YYYY-MM-DD.
    """
    parsed = _parse(date, amount)
    if isinstance(parsed, str):
        return parsed
    led = _ledger(ledger)
    expense = Expense(parsed[0], ExpenseLedger._clean_place(place), ExpenseLedger._to_amount(parsed[1]))
    led.set_pending(expense, problem)
    return format_question(expense)


def undo_last_expense(*, ledger: ExpenseLedger | None = None) -> str:
    """Remove the most recently saved expense."""
    led = _ledger(ledger)
    expense = led.undo_last()
    if expense is None:
        return "Nothing to undo."
    prev, cur = led.totals(_today())
    return format_undone(expense, prev, cur)


def get_expense_totals(*, ledger: ExpenseLedger | None = None) -> str:
    """Return previous-month and current-month expense totals."""
    prev, cur = _ledger(ledger).totals(_today())
    return format_totals(prev, cur)


# Thin wrappers keep the `ledger` kwarg out of the tool schemas.
@server.tool(name="add_expense", description=add_expense.__doc__)
def _add_expense(date: str, place: str, amount: float) -> str:
    return add_expense(date, place, amount)


@server.tool(name="propose_expense", description=propose_expense.__doc__)
def _propose_expense(date: str, place: str, amount: float, problem: str) -> str:
    return propose_expense(date, place, amount, problem)


@server.tool(name="undo_last_expense", description=undo_last_expense.__doc__)
def _undo_last_expense() -> str:
    return undo_last_expense()


@server.tool(name="get_expense_totals", description=get_expense_totals.__doc__)
def _get_expense_totals() -> str:
    return get_expense_totals()


if __name__ == "__main__":
    server.run()
