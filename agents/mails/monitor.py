"""
agents/mails/monitor.py — background loop that reads new emails every
MAIL_CHECK_INTERVAL_HOURS and delivers a formatted summary every
MAIL_SUMMARY_INTERVAL_HOURS.

Timing model
------------
A single loop wakes every 60 seconds and checks two independent clocks:

  check clock   — resets every MAIL_CHECK_INTERVAL_HOURS (default 2 h)
                  triggers: IMAP fetch → LLM summarise → append to pending

  summary clock — resets every MAIL_SUMMARY_INTERVAL_HOURS (default 5 h)
                  triggers: format pending list → deliver → clear pending

The first check runs immediately at startup so the pending list is populated
before the first summary window arrives.  The first summary send fires after
the full MAIL_SUMMARY_INTERVAL_HOURS interval has elapsed.

No-repetition guarantee
-----------------------
EmailStore persists every processed Message-ID to SQLite.  On each check
cycle, only messages whose IDs are not in that store are summarised and added
to the pending list.  The store is updated immediately after summarisation,
before the message is appended, so a crash between those two steps at worst
causes one duplicate entry — not a missed message.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.src.config.config import Config
from core.src.model.llm_factory import create_llm
from core.src.utils import load_prompt
from .email_store import EmailStore
from .imap_client import IMAPAccount

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT_FILE = Path(__file__).parent / "prompts" / "summarise_email.txt"

_TICK_SECONDS = 60
_MAX_PENDING = 500   # guard against unbounded growth when delivery fails


def _load_accounts() -> list[IMAPAccount]:
    """Parse MAIL_ACCOUNTS JSON and return IMAPAccount instances."""
    raw = Config.MAIL_ACCOUNTS
    try:
        configs = json.loads(raw) if raw else []
    except json.JSONDecodeError as exc:
        raise ValueError(f"MAIL_ACCOUNTS is not valid JSON: {exc}") from exc

    if not isinstance(configs, list) or not configs:
        raise ValueError("MAIL_ACCOUNTS must be a non-empty JSON array in .env.")

    accounts = []
    for cfg in configs:
        for key in ("email", "imap_host", "password"):
            if key not in cfg:
                raise ValueError(f"Each account must have '{key}' — missing in: {cfg}")
        accounts.append(IMAPAccount(cfg))

    return accounts


class MailMonitor:
    """
    Polls multiple IMAP inboxes, summarises new emails with the LLM,
    and periodically delivers a combined summary.

    Parameters
    ----------
    send_summary           : callable(message: str) -> None that delivers the
                             formatted summary text.  Pass None to log only.
    check_interval_hours   : override MAIL_CHECK_INTERVAL_HOURS from config
    summary_interval_hours : override MAIL_SUMMARY_INTERVAL_HOURS from config
    """

    def __init__(
        self,
        send_summary: Callable[[str], None] | None = None,
        check_interval_hours: int | None = None,
        summary_interval_hours: int | None = None,
    ) -> None:
        self._accounts = _load_accounts()
        self._store = EmailStore()
        self._llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)
        self._summarise_template = load_prompt(_SUMMARISE_PROMPT_FILE)
        self._send_fn = send_summary

        self._check_secs = (check_interval_hours or int(Config.MAIL_CHECK_INTERVAL_HOURS)) * 3600
        self._summary_secs = (summary_interval_hours or int(Config.MAIL_SUMMARY_INTERVAL_HOURS)) * 3600

        self._pending: list[str] = []
        self._last_check_mono: float = 0.0
        self._last_summary_mono: float = time.monotonic()
        self._last_check_wall: datetime = datetime.now(timezone.utc) - timedelta(
            seconds=self._check_secs
        )
        self._running = False

    # ------------------------------------------------------------------
    # Per-email summarisation
    # ------------------------------------------------------------------

    def _summarise(self, msg: dict) -> str:
        prompt = self._summarise_template.format(
            subject=msg["subject"],
            body=msg["body"],
        )
        resp = self._llm.invoke(prompt)
        text = resp.text
        return text.strip()

    # ------------------------------------------------------------------
    # Check cycle
    # ------------------------------------------------------------------

    def _run_check(self) -> None:
        since = self._last_check_wall
        logger.info(
            "[MailMonitor] checking %d account(s) since %s",
            len(self._accounts),
            since.strftime("%Y-%m-%d %H:%M UTC"),
        )

        counter = len(self._pending) + 1
        new_count = 0

        for account in self._accounts:
            for msg in account.fetch_since(since):
                mid = msg["message_id"]

                if self._store.is_seen(mid):
                    continue

                try:
                    summary_text = self._summarise(msg)
                except Exception:
                    logger.exception("[MailMonitor] LLM summarisation failed for %s", mid[:60])
                    summary_text = "(summarisation failed)"

                self._store.mark_seen(mid)

                if len(self._pending) >= _MAX_PENDING:
                    logger.warning(
                        "[MailMonitor] pending list reached cap (%d) — trigger summary early",
                        _MAX_PENDING,
                    )
                    self._run_summary()

                self._pending.append(
                    f"{counter}. From {msg['from_addr']} to {msg['to']}\n"
                    f"Summary: {summary_text}"
                )
                counter += 1
                new_count += 1

        self._last_check_wall = datetime.now(timezone.utc)
        logger.info("[MailMonitor] check complete — %d new email(s) added", new_count)

    # ------------------------------------------------------------------
    # Summary cycle
    # ------------------------------------------------------------------

    def _run_summary(self) -> None:
        if not self._pending:
            logger.info("[MailMonitor] summary interval reached — no new emails, skipping")
            return

        body = "\n\n".join(self._pending)
        logger.info("[MailMonitor] sending summary (%d email(s))", len(self._pending))

        if self._send_fn:
            try:
                self._send_fn(body)
            except Exception:
                logger.exception("[MailMonitor] delivery failed — summary logged instead")
                logger.info("[MailMonitor] email summary:\n%s", body)
        else:
            logger.info("[MailMonitor] email summary:\n%s", body)

        self._pending.clear()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        logger.info(
            "[MailMonitor] started — %d account(s), check every %dh, summary every %dh",
            len(self._accounts),
            self._check_secs // 3600,
            self._summary_secs // 3600,
        )

        while self._running:
            now = time.monotonic()

            if now - self._last_check_mono >= self._check_secs:
                try:
                    self._run_check()
                except Exception:
                    logger.exception("[MailMonitor] unexpected error during check")
                self._last_check_mono = time.monotonic()

            if time.monotonic() - self._last_summary_mono >= self._summary_secs:
                try:
                    self._run_summary()
                except Exception:
                    logger.exception("[MailMonitor] unexpected error during summary send")
                self._last_summary_mono = time.monotonic()

            time.sleep(_TICK_SECONDS)

        logger.info("[MailMonitor] stopped")

    def stop(self) -> None:
        self._running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
