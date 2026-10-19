"""
agents/fbr_tax/runner.py — runs the fbr-iris-filing skill in a Claude Code session.

The skill drives Chrome (Claude in Chrome) on iris.fbr.gov.pk, so it cannot run
through the agents' usual LangChain model: it needs a real Claude Code session
started in the fbr-tax-filing project, with the skill and Chrome turned on.

Every chat message is one turn of that session.  When the skill needs the user
(log in, pick a mode, give a bank balance, approve the draft) the turn ends with
a question; the next message resumes the same session.  FilingSessions remembers
which Claude Code session belongs to which chat.

Permissions are fixed here, not left to the prompt: no Bash, no web tools, and
file writes only to snapshots/ and the skill's iris-notes.md.  The session also gets the
filing tracker (tracker.py), which decides whether a draft may be created.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from .tracker import SERVER_NAME as TRACKER, FilingTracker, build_server

logger = logging.getLogger(__name__)

SKILL = "fbr-iris-filing"
SESSION_TTL = timedelta(hours=12)   # an older session is not resumed: the next message starts fresh

_NOTES = f"/.claude/skills/{SKILL}/iris-notes.md"
# Permission rules; a leading "/" is relative to the project directory.
_ALLOWED_TOOLS = [
    "Read", "Glob", "Grep",
    "Write(/snapshots/**)", "Edit(/snapshots/**)",
    f"Write({_NOTES})", f"Edit({_NOTES})",
    "mcp__claude-in-chrome",
    f"mcp__{TRACKER}",
]
_BUILTIN_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fbr_tax_sessions (
    session_id        TEXT PRIMARY KEY,
    claude_session_id TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
"""


class FilingSessions:
    """Chat session id → Claude Code session id, in the shared sessions.db."""

    def __init__(self, db_file: str | Path) -> None:
        self._db_path = Path(db_file)

    def get(self, session_id: str, now: datetime) -> str | None:
        """The Claude Code session to resume, or None when there is none or it is too old."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT claude_session_id, updated_at FROM fbr_tax_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None or now - datetime.fromisoformat(row[1]) > SESSION_TTL:
            return None
        return row[0]

    def set(self, session_id: str, claude_session_id: str, now: datetime) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fbr_tax_sessions VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE "
                "SET claude_session_id = excluded.claude_session_id, updated_at = excluded.updated_at",
                (session_id, claude_session_id, now.isoformat()),
            )

    def clear(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM fbr_tax_sessions WHERE session_id = ?", (session_id,))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            conn.executescript(_SCHEMA)
            with conn:
                yield conn
        finally:
            conn.close()


class FilingRunner:
    """One turn of a Claude Code session in the fbr-tax-filing project."""

    def __init__(
        self, project_dir: Path, instructions: str, tracker: FilingTracker,
        model: str | None = None, effort: str | None = None, max_turns: int = 200,
    ) -> None:
        self._project_dir = project_dir
        self._tracker = tracker
        self._instructions = instructions
        self._model = model
        self._effort = effort
        self._max_turns = max_turns

    def skill_installed(self) -> bool:
        return (self._project_dir / ".claude" / "skills" / SKILL / "SKILL.md").is_file()

    def run(self, prompt: str, resume: str | None = None) -> tuple[str, str]:
        """Send *prompt*; return (reply text, Claude Code session id to resume next time)."""
        return asyncio.run(self._run(prompt, resume))

    async def _run(self, prompt: str, resume: str | None) -> tuple[str, str]:
        logger.info("[fbr_tax] claude-code turn (resume=%s)", resume or "new session")
        result: ResultMessage | None = None
        async for message in query(prompt=prompt, options=self._options(resume)):
            if isinstance(message, ResultMessage):
                result = message
        if result is None or result.is_error:
            errors = (result.errors or result.result) if result else "no result returned"
            raise RuntimeError(f"Claude Code turn failed: {errors}")
        logger.info("[fbr_tax] turn done (session=%s, turns=%d, %d ms)",
                    result.session_id, result.num_turns, result.duration_ms)
        return result.result or "", result.session_id

    def _options(self, resume: str | None) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            cwd=str(self._project_dir),
            system_prompt={"type": "preset", "preset": "claude_code", "append": self._instructions},
            model=self._model,
            effort=self._effort,
            max_turns=self._max_turns,
            resume=resume,
            skills=[SKILL, "claude-in-chrome"],
            tools=_BUILTIN_TOOLS,             # no Bash, no web fetch/search
            mcp_servers={TRACKER: build_server(self._tracker, self._project_dir / "snapshots")},
            allowed_tools=_ALLOWED_TOOLS,
            permission_mode="dontAsk",        # anything not allowed above is denied
            setting_sources=["project"],      # the filing project's skill; not the user's hooks/plugins
            extra_args={"chrome": None},      # Claude in Chrome
            # .env is loaded into os.environ; blank the key so the CLI uses its own login
            env={"ANTHROPIC_API_KEY": ""},
        )
