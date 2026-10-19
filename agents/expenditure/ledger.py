import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

DB_FILENAME = "expenses.db"

# Expenses are never deleted: undo stamps deleted_at so the long-run record stays complete.
# monthly_totals keeps one row per month for anyone browsing the database directly.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    spent_on     TEXT    NOT NULL,
    month        TEXT    NOT NULL,
    place        TEXT    NOT NULL,
    amount_cents INTEGER NOT NULL,
    created_at   TEXT    NOT NULL,
    deleted_at   TEXT
);
CREATE INDEX IF NOT EXISTS expenses_by_month ON expenses (month, deleted_at);
CREATE TABLE IF NOT EXISTS inputs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    text        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    spent_on     TEXT    NOT NULL,
    place        TEXT    NOT NULL,
    amount_cents INTEGER,
    problem      TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
);
CREATE VIEW IF NOT EXISTS monthly_totals AS
    SELECT month, COUNT(*) AS entries, SUM(amount_cents) / 100.0 AS total_eur
    FROM expenses WHERE deleted_at IS NULL GROUP BY month;
"""


@dataclass(frozen=True)
class Expense:
    date: date
    place: str
    amount: Decimal | None  # None only for a pending item whose amount is still unknown


class ExpenseLedger:
    """SQLite-backed expense store: every expense, every received input, and one pending slot.

    Shared by the bot process and the expenses MCP subprocess, so each operation opens its own
    short-lived connection; WAL mode plus a busy timeout lets both write safely.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._db_path = self._root / DB_FILENAME
        self._schema_ready = False

    def add(self, date: date, place: str, amount: Decimal | float | str) -> Expense:
        expense = Expense(date, self._clean_place(place), self._to_amount(amount))
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO expenses (spent_on, month, place, amount_cents, created_at) VALUES (?, ?, ?, ?, ?)",
                (expense.date.isoformat(), _month_key(expense.date), expense.place,
                 _to_cents(expense.amount), _now()),
            )
        logger.info("expenses: saved #%d %s | %s | %.2f", cur.lastrowid, expense.date, expense.place, expense.amount)
        return expense

    def log_input(self, text: str, when: datetime | None = None) -> None:
        """Record one received message, whatever happens to it afterwards."""
        when = when or datetime.now()
        with self._connect() as conn:
            conn.execute("INSERT INTO inputs (received_at, text) VALUES (?, ?)",
                         (when.isoformat(sep=" ", timespec="seconds"), " ".join(text.split())))

    def inputs(self) -> list[tuple[str, str]]:
        """All recorded inputs as (received_at, text), oldest first."""
        with self._connect() as conn:
            return conn.execute("SELECT received_at, text FROM inputs ORDER BY id").fetchall()

    def expenses(self) -> list[Expense]:
        """All live (not undone) expenses, in the order they were saved."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT spent_on, place, amount_cents FROM expenses WHERE deleted_at IS NULL ORDER BY id"
            ).fetchall()
        return [_row_to_expense(r) for r in rows]

    def undo_last(self) -> Expense | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, spent_on, place, amount_cents FROM expenses WHERE deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                logger.info("expenses: undo requested but nothing to undo")
                return None
            conn.execute("UPDATE expenses SET deleted_at = ? WHERE id = ?", (_now(), row[0]))
        expense = _row_to_expense(row[1:])
        logger.info("expenses: undone #%d %s | %s | %.2f", row[0], expense.date, expense.place, expense.amount)
        return expense

    def totals(self, today: date) -> tuple[Decimal, Decimal]:
        prev_year, prev_month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return self._month_total(date(prev_year, prev_month, 1)), self._month_total(today)

    def breakdown(self, year: int, month: int) -> list[tuple[str, Decimal]]:
        """Per-place totals for one month, largest first. Places match case-insensitively."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT place, amount_cents FROM expenses WHERE month = ? AND deleted_at IS NULL ORDER BY id",
                (_month_key(date(year, month, 1)),),
            ).fetchall()
        totals: dict[str, tuple[str, Decimal]] = {}
        for place, cents in rows:
            name, total = totals.get(place.casefold(), (place, Decimal("0")))
            totals[place.casefold()] = (name, total + _from_cents(cents))
        return sorted(totals.values(), key=lambda t: (-t[1], t[0].casefold()))

    def top_expenses(self, year: int, month: int, limit: int = 5) -> list[Expense]:
        """The largest single (not undone) expenses of one month, biggest first; ties → earliest saved."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT spent_on, place, amount_cents FROM expenses WHERE month = ? AND deleted_at IS NULL "
                "ORDER BY amount_cents DESC, id LIMIT ?",
                (_month_key(date(year, month, 1)), limit),
            ).fetchall()
        return [_row_to_expense(r) for r in rows]

    def set_pending(self, expense: Expense, problem: str) -> None:
        cents = None if expense.amount is None else _to_cents(expense.amount)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending (id, spent_on, place, amount_cents, problem, created_at) "
                "VALUES (1, ?, ?, ?, ?, ?)",
                (expense.date.isoformat(), expense.place, cents, problem, _now()),
            )
        logger.info("expenses: pending %s | %s | %s (problem: %s)",
                    expense.date, expense.place, expense.amount, problem)

    def get_pending(self) -> tuple[Expense, str] | None:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT spent_on, place, amount_cents, problem FROM pending").fetchone()
        except sqlite3.DatabaseError:
            logger.exception("expenses: could not read the pending item from %s — treating as none", self._db_path)
            return None
        if row is None:
            return None
        return _row_to_expense(row[:3]), row[3]

    def clear_pending(self) -> None:
        with self._connect() as conn:
            removed = conn.execute("DELETE FROM pending").rowcount
        if removed:
            logger.info("expenses: pending item cleared")

    def confirm_pending(self) -> Expense | None:
        pending = self.get_pending()
        if pending is None:
            return None
        expense = pending[0]
        added = self.add(expense.date, expense.place, expense.amount)
        self.clear_pending()
        return added

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation; commits on success, rolls back on error."""
        self._root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            if not self._schema_ready:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                self._schema_ready = True
                logger.info("expenses: using database %s", self._db_path)
            with conn:
                yield conn
        finally:
            conn.close()

    def _month_total(self, d: date) -> Decimal:
        with self._connect() as conn:
            (cents,) = conn.execute(
                "SELECT COALESCE(SUM(amount_cents), 0) FROM expenses WHERE month = ? AND deleted_at IS NULL",
                (_month_key(d),),
            ).fetchone()
        return _from_cents(cents)

    @staticmethod
    def _clean_place(place: str) -> str:
        return " ".join(place.split())

    @staticmethod
    def _to_amount(amount: Decimal | float | str) -> Decimal:
        return Decimal(str(amount).replace(",", ".")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _to_cents(amount: Decimal) -> int:
    return int((amount * 100).to_integral_value(rounding=ROUND_HALF_UP))


def _from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(Decimal("0.01"))


def _row_to_expense(row) -> Expense:
    spent_on, place, cents = row
    return Expense(date.fromisoformat(spent_on), place, None if cents is None else _from_cents(cents))
