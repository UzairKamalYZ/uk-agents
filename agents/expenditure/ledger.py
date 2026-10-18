import json
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


@dataclass(frozen=True)
class Expense:
    date: date
    place: str
    amount: Decimal

    def line(self) -> str:
        return f"{self.date.isoformat()} | {self.place} | {self.amount:.2f}"

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "place": self.place, "amount": str(self.amount)}

    @classmethod
    def from_dict(cls, d: dict) -> "Expense":
        return cls(date.fromisoformat(d["date"]), d["place"], Decimal(d["amount"]))


class ExpenseLedger:
    """Plain-text monthly expense files plus a history stack and a pending slot."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def add(self, date: date, place: str, amount: Decimal | float | str) -> Expense:
        expense = Expense(date, self._clean_place(place), self._to_amount(amount))
        self._root.mkdir(parents=True, exist_ok=True)
        with self._month_file(date).open("a", encoding="utf-8") as f:
            f.write(expense.line() + "\n")
        history = self._read_list(self._history_path)
        history.append(expense.to_dict())
        self._write_json(self._history_path, history)
        return expense

    def undo_last(self) -> Expense | None:
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
            path.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
        self._write_json(self._history_path, history)
        return expense

    def totals(self, today: date) -> tuple[Decimal, Decimal]:
        prev_year, prev_month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return self._month_total(prev_year, prev_month), self._month_total(today.year, today.month)

    def set_pending(self, expense: Expense, problem: str) -> None:
        self._write_json(self._pending_path, {"expense": expense.to_dict(), "problem": problem})

    def get_pending(self) -> tuple[Expense, str] | None:
        if not self._pending_path.exists():
            return None
        data = json.loads(self._pending_path.read_text(encoding="utf-8"))
        return Expense.from_dict(data["expense"]), data["problem"]

    def clear_pending(self) -> None:
        self._pending_path.unlink(missing_ok=True)

    def confirm_pending(self) -> Expense | None:
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
    def _pending_path(self) -> Path:
        return self._root / "pending.json"

    def _month_file(self, d: date) -> Path:
        return self._root / f"{d.year:04d}-{d.month:02d}.txt"

    def _month_total(self, year: int, month: int) -> Decimal:
        path = self._month_file(date(year, month, 1))
        total = Decimal("0")
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split(" | ")
                if len(parts) == 3:
                    total += Decimal(parts[2])
        return total

    @staticmethod
    def _clean_place(place: str) -> str:
        return place.strip().replace("|", "/")

    @staticmethod
    def _to_amount(amount: Decimal | float | str) -> Decimal:
        return Decimal(str(amount).replace(",", ".")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _read_list(self, path: Path) -> list:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
