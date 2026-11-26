"""
When crawl passes run: at fixed times of day (CRAWLER_RUN_AT, e.g. "04:30,13:00") in
CRAWLER_TIMEZONE, or, with CRAWLER_RUN_AT empty, every CRAWLER_INTERVAL_MINUTES.
"""
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse_times(text: str) -> list[time]:
    """"04:30, 13:00" → [04:30, 13:00], sorted and without repeats. Raises ValueError on a bad entry."""
    times = set()
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        match = _TIME.match(part)
        if not match:
            raise ValueError(f"CRAWLER_RUN_AT: {part!r} is not a time like 04:30")
        times.add(time(int(match.group(1)), int(match.group(2))))
    return sorted(times)


@dataclass(frozen=True)
class Schedule:
    times: tuple[time, ...] = ()          # empty → every interval_minutes
    tz: str = "UTC"
    interval_minutes: int = 60

    def __post_init__(self):
        try:
            ZoneInfo(self.tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"CRAWLER_TIMEZONE: unknown time zone {self.tz!r}") from exc

    @property
    def fixed(self) -> bool:
        return bool(self.times)

    def _slots_around(self, now: datetime) -> list[datetime]:
        zone = ZoneInfo(self.tz)
        local = now.astimezone(zone)
        days = [local.date() + timedelta(days=d) for d in (-1, 0, 1)]
        return sorted(datetime.combine(day, t, tzinfo=zone) for day in days for t in self.times)

    def next_run(self, now: datetime, last_pass: datetime | None = None) -> datetime:
        """When the next pass starts (now, if it is due)."""
        if not self.fixed:
            if last_pass is None:
                return now
            return max(now, last_pass + timedelta(minutes=self.interval_minutes))
        if last_pass is None or last_pass < self.previous_slot(now):
            return now                    # never ran, or a slot was missed (restart, deploy): catch up
        return next(s for s in self._slots_around(now) if s > now).astimezone(timezone.utc)

    def previous_slot(self, now: datetime) -> datetime:
        """The latest scheduled time at or before now."""
        return [s for s in self._slots_around(now) if s <= now][-1].astimezone(timezone.utc)

    def describe(self) -> str:
        if not self.fixed:
            return f"a pass every {self.interval_minutes} minutes"
        return "passes at " + ", ".join(t.strftime("%H:%M") for t in self.times) + f" ({self.tz})"
