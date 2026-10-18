from decimal import Decimal

from .ledger import Expense


def format_eur(amount: Decimal) -> str:
    if amount == amount.to_integral_value():
        return f"€{int(amount)}"
    return f"€{amount:.2f}"


def format_totals(prev: Decimal, cur: Decimal) -> str:
    return f"prev_month: {format_eur(prev)}, current_month: {format_eur(cur)}"


def format_logged(e: Expense, prev: Decimal, cur: Decimal) -> str:
    return f"Logged {format_eur(e.amount)} at {e.place} ({e.date.isoformat()}).\n{format_totals(prev, cur)}"


def format_undone(e: Expense, prev: Decimal, cur: Decimal) -> str:
    return f"Removed {format_eur(e.amount)} at {e.place} ({e.date.isoformat()}).\n{format_totals(prev, cur)}"


def format_question(e: Expense) -> str:
    return (
        f"I understood: {e.place}, {format_eur(e.amount)}, {e.date.isoformat()}. "
        "Reply yes to save, or send the corrected version."
    )
