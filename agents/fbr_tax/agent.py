import logging
import threading
from datetime import datetime
from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.config.config import Config
from core.src.utils import load_prompt

from .runner import SKILL, FilingRunner, FilingSessions

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"
_SIGNATURE = "\n\n— FBR-PK Tax Filing"

BUSY = "⏳ Still working on your previous message — I'll answer it first." + _SIGNATURE
FAILED = ("⚠️ The filing session hit an error. Your next message starts a new session — "
          "ask me to resume the filing and I'll pick up from the saved checkpoint." + _SIGNATURE)
NO_SKILL = f"⚠️ The {SKILL} skill was not found in {{dir}}. Set FBR_TAX_DIR in .env." + _SIGNATURE


class FbrTaxAgent(BaseAgent):
    """FBR IRIS filing through the fbr-iris-filing skill, in a Claude Code session with Chrome.

    ask() bypasses the usual LLM graph: each message is one resumed turn of that session
    (see runner.py).  Only one turn runs at a time — there is one browser.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._runner = FilingRunner(
            Path(Config.FBR_TAX_DIR), load_prompt(_PROMPT_FILE),
            model=kwargs.get("llm_model") or Config.LLM_MODEL_NAME,
            effort=Config.LLM_EFFORT, max_turns=Config.FBR_TAX_MAX_TURNS,
        )
        self._sessions = FilingSessions(Config.SESSION_DB_FILE)
        self._busy = threading.Lock()

    def get_system_prompt(self) -> str:
        return load_prompt(_PROMPT_FILE)

    def get_rag_context_manager(self):
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return []

    def _now(self) -> datetime:
        return datetime.now()

    def ask(self, prompt: str, stream: bool = False, session_id: str = None):
        reply = self._ask(prompt, session_id or self.session_id)
        return iter([reply]) if stream else reply

    def _ask(self, prompt: str, session_id: str) -> str:
        if not self._runner.skill_installed():
            logger.error("[fbr_tax] skill missing under %s", Config.FBR_TAX_DIR)
            return NO_SKILL.format(dir=Config.FBR_TAX_DIR)
        if not self._busy.acquire(blocking=False):
            logger.warning("[fbr_tax] turn refused: another turn is running (session=%s)", session_id)
            return BUSY
        try:
            resume = self._sessions.get(session_id, self._now())
            reply, claude_session = self._runner.run(prompt, resume)
            self._sessions.set(session_id, claude_session, self._now())
            return reply
        except Exception:
            # Not retried: replaying a mid-filing answer ("yes", an amount) in a fresh session
            # would lose its meaning.  The next message starts a new session instead.
            logger.exception("[fbr_tax] turn failed (session=%s)", session_id)
            self._sessions.clear(session_id)
            return FAILED
        finally:
            self._busy.release()
