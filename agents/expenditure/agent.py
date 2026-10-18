from datetime import date
from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.config.config import Config
from core.src.model.mcp_tools import _mcp
from core.src.utils import load_prompt

from .ledger import ExpenseLedger
from .replies import format_eur, format_logged

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"


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
        return _mcp.langchain_tools(server_name="expenses")

    def _today(self) -> date:
        return date.today()

    # Single owner: session_id is ignored for pending state.

    def pending_summary(self, session_id: str) -> str | None:
        pending = self._ledger.get_pending()
        if pending is None:
            return None
        e, problem = pending
        return f"{e.place}, {format_eur(e.amount)}, {e.date.isoformat()} (problem: {problem})"

    def confirm_pending(self, session_id: str) -> str:
        e = self._ledger.confirm_pending()
        if e is None:
            return "Nothing to confirm."
        return format_logged(e, *self._ledger.totals(self._today()))

    def discard_pending(self, session_id: str) -> None:
        self._ledger.clear_pending()
