import logging
import sqlite3
from datetime import date
from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.config.config import Config
from core.src.model.mcp_tools import _mcp
from core.src.utils import load_prompt

from .ledger import ExpenseLedger
from .replies import format_amount, format_logged, format_need_amount, format_nothing, sign

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"
UNAVAILABLE_REPLY = sign("⚠️ Expense tracking is unavailable right now — nothing was saved.")


class ExpenditureAgent(BaseAgent):
    """Owner's personal expense tracker — uses the expenses MCP server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ledger = ExpenseLedger(Path(Config.EXPENSES_DIR))

    def get_system_prompt(self) -> str:
        return load_prompt(str(_PROMPT_FILE))

    def get_rag_context_manager(self):
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        tools = _mcp.langchain_tools(server_name="expenses")
        # Without tools the LLM could claim a save that never happened — fail closed in ask().
        self._tools_available = bool(tools)
        if not self._tools_available:
            logger.error("ExpenditureAgent: no tools from the 'expenses' MCP server — expense tracking disabled")
        return tools

    def _today(self) -> date:
        return date.today()

    def ask(self, prompt: str, stream: bool = False, session_id: str = None):
        logger.info("ExpenditureAgent: received %r (session=%s)", prompt, session_id)
        self._log_input(prompt)
        if not self._tools_available:
            return iter([UNAVAILABLE_REPLY]) if stream else UNAVAILABLE_REPLY
        today = self._today()
        dated = f"Today is {today.isoformat()} ({today.strftime('%A')}).\n\n{prompt}"
        return super().ask(dated, stream=stream, session_id=session_id)

    # Single owner: session_id is ignored for pending state.

    def pending_summary(self, session_id: str) -> str | None:
        pending = self._ledger.get_pending()
        if pending is None:
            return None
        e, problem = pending
        return f"{e.place}, {format_amount(e.amount)}, {e.date.isoformat()} (problem: {problem})"

    def confirm_pending(self, session_id: str) -> str:
        self._log_input("[confirmed pending expense]")
        pending = self._ledger.get_pending()
        if pending is not None and pending[0].amount is None:
            return format_need_amount(pending[0])
        e = self._ledger.confirm_pending()
        if e is None:
            return format_nothing("confirm")
        return format_logged(e, *self._ledger.totals(self._today()))

    def discard_pending(self, session_id: str) -> None:
        self._log_input("[discarded pending expense]")
        self._ledger.clear_pending()

    def _log_input(self, text: str) -> None:
        # The input log is a record, not a gate — a write failure must not block the reply.
        try:
            self._ledger.log_input(text)
        except (OSError, sqlite3.Error):
            logger.exception("ExpenditureAgent: could not record the input")
