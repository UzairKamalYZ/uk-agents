"""
agents/fbr_tax/tracker.py — the agent's own record of each tax year's filing status.

Before any filing, the Claude Code session checks IRIS (Completed Tasks → DECLARATION
for submitted returns, the Draft tab for drafts), records what it saw here, and asks
decide() what to do.  The decision rules live in code, not in the prompt:

  tax year already submitted          → stop, nothing to file
  previous year not submitted         → stop and ask the owner (its return is the template)
  a draft exists for the tax year     → ask the owner: continue that draft or start fresh
  no draft, not submitted             → create the draft

A status older than CHECK_TTL counts as unknown, so the session has to look at IRIS
again rather than act on a stale record.  The tools are exposed to the session as an
in-process MCP server ("fbr_tracker").
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

logger = logging.getLogger(__name__)

SERVER_NAME = "fbr_tracker"
CHECK_TTL = timedelta(hours=12)
STATUSES = ("not_started", "draft_saved", "submitted")

CHECK_FIRST = "check_iris"
ALREADY_SUBMITTED = "already_submitted"
PREVIOUS_NOT_SUBMITTED = "previous_year_not_submitted"
ASK_CONTINUE_OR_FRESH = "ask_continue_or_fresh"
CREATE_DRAFT = "create_draft"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    tax_year    INTEGER PRIMARY KEY,
    status      TEXT NOT NULL,
    draft_count INTEGER NOT NULL DEFAULT 0,
    note        TEXT NOT NULL DEFAULT '',
    checked_at  TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Filing:
    tax_year: int
    status: str
    draft_count: int
    note: str
    checked_at: datetime


class FilingTracker:
    """SQLite record of what IRIS showed for each tax year."""

    def __init__(self, db_file: str | Path, now: Callable[[], datetime] = datetime.now) -> None:
        self._db_path = Path(db_file)
        self._now = now

    def record(self, tax_year: int, status: str, draft_count: int = 0, note: str = "") -> Filing:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        if draft_count < 0:
            raise ValueError("draft_count cannot be negative")
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO filings VALUES (?, ?, ?, ?, ?) ON CONFLICT(tax_year) DO UPDATE SET "
                "status = excluded.status, draft_count = excluded.draft_count, "
                "note = excluded.note, checked_at = excluded.checked_at",
                (tax_year, status, draft_count, note, now.isoformat()),
            )
        logger.info("[fbr_tracker] TY%d recorded: %s, drafts=%d", tax_year, status, draft_count)
        return Filing(tax_year, status, draft_count, note, now)

    def get(self, tax_year: int) -> Filing | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tax_year, status, draft_count, note, checked_at FROM filings WHERE tax_year = ?",
                (tax_year,),
            ).fetchone()
        if row is None:
            return None
        return Filing(row[0], row[1], row[2], row[3], datetime.fromisoformat(row[4]))

    def all(self) -> list[Filing]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tax_year, status, draft_count, note, checked_at FROM filings ORDER BY tax_year"
            ).fetchall()
        return [Filing(r[0], r[1], r[2], r[3], datetime.fromisoformat(r[4])) for r in rows]

    def decide(self, tax_year: int) -> tuple[str, str]:
        """(action, explanation) for filing *tax_year*, from fresh records only."""
        now = self._now()
        current, previous = self.get(tax_year), self.get(tax_year - 1)
        # A submitted return stays submitted, so an old "submitted" record is still good.
        if current is not None and current.status == "submitted":
            return ALREADY_SUBMITTED, f"TY{tax_year} is already submitted. Do not create or edit a draft."
        stale = [y for y, f in ((tax_year, current), (tax_year - 1, previous))
                 if f is None or (f.status != "submitted" and now - f.checked_at > CHECK_TTL)]
        if stale:
            years = " and ".join(f"TY{y}" for y in stale)
            return CHECK_FIRST, (f"No recent IRIS check for {years}. Look in IRIS (Completed Tasks → "
                                 "DECLARATION, and the Draft tab), record each year, then decide again.")
        if previous.status != "submitted":
            return PREVIOUS_NOT_SUBMITTED, (f"TY{tax_year - 1} is not submitted ({previous.status}). "
                                            "Stop and ask the owner how to proceed; do not create a draft.")
        if current.draft_count > 0 or current.status == "draft_saved":
            n = max(current.draft_count, 1)
            return ASK_CONTINUE_OR_FRESH, (f"TY{tax_year} is not submitted and has {n} draft(s) in IRIS. "
                                           "Ask the owner: continue the existing draft, or start a fresh attempt? "
                                           "Do nothing in IRIS until they answer.")
        return CREATE_DRAFT, f"TY{tax_year} is not submitted and has no draft. Create the draft."

    @contextmanager
    def _connect(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            conn.executescript(_SCHEMA)
            with conn:
                yield conn
        finally:
            conn.close()


def _text(s: str, error: bool = False) -> dict:
    out = {"content": [{"type": "text", "text": s}]}
    if error:
        out["is_error"] = True
    return out


def _describe(f: Filing) -> str:
    note = f" — {f.note}" if f.note else ""
    return f"TY{f.tax_year}: {f.status}, drafts={f.draft_count}, checked {f.checked_at:%Y-%m-%d %H:%M}{note}"


def build_server(tracker: FilingTracker):
    """The tracker as an in-process MCP server for the Claude Code session."""

    async def record(args: dict) -> dict:
        try:
            f = tracker.record(int(args["tax_year"]), args["status"],
                               int(args.get("draft_count", 0)), args.get("note", ""))
        except (KeyError, ValueError) as e:
            return _text(f"Not recorded: {e}", error=True)
        return _text("Recorded " + _describe(f))

    async def decide(args: dict) -> dict:
        action, why = tracker.decide(int(args["tax_year"]))
        logger.info("[fbr_tracker] decide TY%s → %s", args["tax_year"], action)
        return _text(f"{action}: {why}")

    async def show(args: dict) -> dict:
        filings = tracker.all()
        return _text("\n".join(_describe(f) for f in filings) or "Nothing recorded yet.")

    year = {"type": "integer", "description": "Tax year N (1 July N-1 to 30 June N), e.g. 2027"}
    tools = [
        SdkMcpTool(
            name="record_filing_status",
            description=("Record what IRIS shows right now for one tax year: 'submitted' (in Completed "
                         "Tasks → DECLARATION), 'draft_saved' (a draft exists, not submitted) or "
                         "'not_started'. Give draft_count = number of drafts for that year in the Draft tab. "
                         "Only record what you saw in IRIS or what the owner just told you."),
            input_schema={"type": "object", "properties": {
                "tax_year": year,
                "status": {"type": "string", "enum": list(STATUSES)},
                "draft_count": {"type": "integer", "minimum": 0},
                "note": {"type": "string", "description": "Short, no amounts or personal identifiers"},
            }, "required": ["tax_year", "status"]},
            handler=record,
        ),
        SdkMcpTool(
            name="decide_filing_action",
            description=("Before creating, opening or deleting any draft: returns the required next action "
                         "for filing this tax year. Follow it exactly."),
            input_schema={"type": "object", "properties": {"tax_year": year}, "required": ["tax_year"]},
            handler=decide,
        ),
        SdkMcpTool(
            name="show_filing_records",
            description="Every tax year's recorded filing status and when it was last checked in IRIS.",
            input_schema={"type": "object", "properties": {}},
            handler=show,
        ),
    ]
    return create_sdk_mcp_server(SERVER_NAME, tools=tools)
