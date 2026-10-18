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


def format_amount(amount: Decimal | None) -> str:
    return "amount unknown" if amount is None else format_eur(amount)


def format_question(e: Expense, problem: str) -> str:
    return (
        f"I understood: {e.place}, {format_amount(e.amount)}, {e.date.isoformat()} (problem: {problem}). "
        "Reply yes to save, no to discard, or send the corrected version."
    )


def format_need_amount(e: Expense) -> str:
    return (
        f"I still need the amount for {e.place} ({e.date.isoformat()}). "
        "Send the corrected version, or no to discard."
    )
