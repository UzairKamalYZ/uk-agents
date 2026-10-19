from datetime import date
from decimal import Decimal

from .ledger import Expense

AGENT_NAME = "Expenditure Tracker"


def format_eur(amount: Decimal) -> str:
    return f"€{amount:,.2f}"


def format_amount(amount: Decimal | None) -> str:
    return "amount unknown" if amount is None else format_eur(amount)


def format_day(d: date) -> str:
    return d.strftime("%a %d %b %Y")


def sign(text: str) -> str:
    return f"{text}\n\n— {AGENT_NAME}"


def _details(e: Expense) -> str:
    return f"📍 {e.place}\n💶 {format_amount(e.amount)}\n📅 {format_day(e.date)}"


def _totals_block(prev: Decimal, cur: Decimal) -> str:
    return f"📊 Totals\nLast month: {format_eur(prev)}\nThis month: {format_eur(cur)}"


def format_totals(prev: Decimal, cur: Decimal) -> str:
    return sign(_totals_block(prev, cur))


def format_logged(e: Expense, prev: Decimal, cur: Decimal) -> str:
    return sign(f"✅ Expense saved\n{_details(e)}\n\n{_totals_block(prev, cur)}")


def format_undone(e: Expense, prev: Decimal, cur: Decimal) -> str:
    return sign(f"↩️ Expense removed\n{_details(e)}\n\n{_totals_block(prev, cur)}")


def format_summary(today: date, prev: Decimal, cur: Decimal, top: list[Expense]) -> str:
    """Spending overview: both monthly totals, then this month's largest expenses."""
    title = f"🏆 Top {len(top)} expenses — {today.strftime('%B %Y')}" if top else ""
    lines = "\n".join(
        f"{i}. {e.place} — {format_eur(e.amount)} ({e.date.strftime('%d %b')})" for i, e in enumerate(top, 1)
    )
    body = _totals_block(prev, cur)
    body += f"\n\n{title}\n{lines}" if top else "\n\nNo expenses logged this month yet."
    return sign(body)


def format_breakdown(month: date, rows: list[tuple[str, Decimal]]) -> str:
    title = f"🧾 Spending by place — {month.strftime('%B %Y')}"
    if not rows:
        return sign(f"{title}\nNo expenses logged.")
    lines = "\n".join(f"• {place}: {format_eur(amount)}" for place, amount in rows)
    total = sum((amount for _, amount in rows), Decimal("0"))
    return sign(f"{title}\n{lines}\n\nTotal: {format_eur(total)}")


def format_question(e: Expense, problem: str) -> str:
    return sign(
        f"⚠️ Please check this expense\n{_details(e)}\n❓ {problem}\n\n"
        "Reply yes to save, no to discard, or send the corrected version."
    )


def format_need_amount(e: Expense) -> str:
    return sign(
        f"⚠️ I still need the amount\n📍 {e.place}\n📅 {format_day(e.date)}\n\n"
        "Send the corrected version, or no to discard."
    )


def format_nothing(action: str) -> str:
    return sign(f"ℹ️ Nothing to {action}.")
