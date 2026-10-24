import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

logger = logging.getLogger(__name__)

# Anything a corrupt/truncated/hand-edited JSON file can raise while being decoded.
_CORRUPT = (OSError, ValueError, KeyError, TypeError, AttributeError, InvalidOperation)

try:
    import fcntl

    def _lock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
except ImportError:  # Windows
    import msvcrt

    def _lock_file(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

# Per-directory reentrant locks (threads in this process) + nesting depth, so only the
# outermost acquisition takes the file lock that also excludes the expenses MCP server process.
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_DEPTH: dict[Path, int] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class Expense:
    date: date
    place: str
    amount: Decimal | None  # None only for a pending item whose amount is still unknown

    def line(self) -> str:
        return f"{self.date.isoformat()} | {self.place} | {self.amount:.2f}"

    def to_dict(self) -> dict:
        amount = None if self.amount is None else str(self.amount)
        return {"date": self.date.isoformat(), "place": self.place, "amount": amount}

    @classmethod
    def from_dict(cls, d: dict) -> "Expense":
        amount = None if d["amount"] is None else Decimal(d["amount"])
        return cls(date.fromisoformat(d["date"]), d["place"], amount)


class ExpenseLedger:
    """Plain-text monthly expense files, an input log, a history stack and a pending slot."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive access to the ledger across threads and processes; reentrant."""
        key = self._root.resolve()
        with _THREAD_LOCKS_GUARD:
            rlock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with rlock:
            if _DEPTH.get(key, 0):
                _DEPTH[key] += 1
                try:
                    yield
                finally:
                    _DEPTH[key] -= 1
                return
            self._root.mkdir(parents=True, exist_ok=True)
            with (self._root / ".lock").open("a+b") as f:
                _lock_file(f)  # released when the file is closed
                _DEPTH[key] = 1
                try:
                    yield
                finally:
                    _DEPTH[key] = 0

    def add(self, date: date, place: str, amount: Decimal | float | str) -> Expense:
        expense = Expense(date, self._clean_place(place), self._to_amount(amount))
        with self.locked():
            with self._month_file(date).open("a", encoding="utf-8") as f:
                f.write(expense.line() + "\n")
            history = self._read_list(self._history_path)
            history.append(expense.to_dict())
            self._write_json(self._history_path, history)
        return expense

    def log_input(self, text: str, when: datetime | None = None) -> None:
        """Append one received message to inputs.log, whatever happens to it afterwards."""
        when = when or datetime.now()
        with self.locked(), self._inputs_path.open("a", encoding="utf-8") as f:
            f.write(f"{when.isoformat(sep=' ', timespec='seconds')} | {' '.join(text.split())}\n")

    def undo_last(self) -> Expense | None:
        with self.locked():
            history = self._read_list(self._history_path)
            if not history:
                return None
            expense = Expense.from_dict(history.pop())
            path = self._month_file(expense.date)
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                target = expense.line()
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i] == target:
                        del lines[i]
                        break
                self._write_text(path, "".join(f"{ln}\n" for ln in lines))
            self._write_json(self._history_path, history)
        return expense

    def totals(self, today: date) -> tuple[Decimal, Decimal]:
        prev_year, prev_month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return self._month_total(prev_year, prev_month), self._month_total(today.year, today.month)

    def breakdown(self, year: int, month: int) -> list[tuple[str, Decimal]]:
        """Per-place totals for one month, largest first. Places match case-insensitively."""
        totals: dict[str, tuple[str, Decimal]] = {}
        for place, amount in self._month_entries(year, month):
            name, total = totals.get(place.casefold(), (place, Decimal("0")))
            totals[place.casefold()] = (name, total + amount)
        return sorted(totals.values(), key=lambda t: (-t[1], t[0].casefold()))

    def set_pending(self, expense: Expense, problem: str) -> None:
        with self.locked():
            self._write_json(self._pending_path, {"expense": expense.to_dict(), "problem": problem})

    def get_pending(self) -> tuple[Expense, str] | None:
        if not self._pending_path.exists():
            return None
        try:
            data = json.loads(self._pending_path.read_text(encoding="utf-8"))
            return Expense.from_dict(data["expense"]), str(data["problem"])
        except _CORRUPT:
            logger.exception("expenses: corrupt %s — treating as no pending item", self._pending_path)
            return None

    def clear_pending(self) -> None:
        with self.locked():
            self._pending_path.unlink(missing_ok=True)

    def confirm_pending(self) -> Expense | None:
        # Held across read → add → clear so a concurrent change can't be saved twice or lost
        with self.locked():
            pending = self.get_pending()
            if pending is None:
                return None
            expense = pending[0]
            added = self.add(expense.date, expense.place, expense.amount)
            self.clear_pending()
        return added

    @property
    def _history_path(self) -> Path:
        return self._root / "history.json"

    @property
    def _inputs_path(self) -> Path:
        return self._root / "inputs.log"

    @property
    def _pending_path(self) -> Path:
        return self._root / "pending.json"

    def _month_file(self, d: date) -> Path:
        return self._root / f"{d.year:04d}-{d.month:02d}.txt"

    def _month_entries(self, year: int, month: int) -> list[tuple[str, Decimal]]:
        path = self._month_file(date(year, month, 1))
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(" | ")
            if len(parts) == 3:
                entries.append((parts[1], Decimal(parts[2])))
        return entries

    def _month_total(self, year: int, month: int) -> Decimal:
        return sum((amount for _, amount in self._month_entries(year, month)), Decimal("0"))

    @staticmethod
    def _clean_place(place: str) -> str:
        return place.strip().replace("|", "/")

    @staticmethod
    def _to_amount(amount: Decimal | float | str) -> Decimal:
        return Decimal(str(amount).replace(",", ".")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _read_list(self, path: Path) -> list:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except _CORRUPT:
            data = None
        if not isinstance(data, list):
            logger.error("expenses: corrupt %s — treating as empty", path)
            return []
        return data

    def _write_json(self, path: Path, data) -> None:
        self._write_text(path, json.dumps(data))

    def _write_text(self, path: Path, text: str) -> None:
        """Atomic write: temp file in the same directory, then os.replace."""
        self._root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._root, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
