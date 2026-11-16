import asyncio
import inspect
import logging
import uuid
from pathlib import Path

from .config.config import Config
from .model.embedder import Embedder
from .model.llm_factory import create_llm
from .model.llm_model_graph import LlmModelGraph
from .model.rag_context_manager import RagContextManager
from .model.session_manager import shared_session_store
from .utils import load_prompt

logger = logging.getLogger(__name__)


class NullRagContextManager:
    """No-op RAG context manager — returns empty string, skips all vector search."""

    def get_context(self, user_prompt: str, k: int | None = None) -> str:
        return ""


class BaseAgent:
    """
    Shared base for all agents: LLM, LangGraph pipeline, session history, ask / aask / close.

    A new agent is a subclass plus `prompts/System_Prompt.txt` next to its module.
    Override a hook only when the default doesn't fit:

        get_system_prompt()       — default: prompts/System_Prompt.txt beside the agent's module
        get_rag_context_manager() — default: vector search when "rag" is true in agents.json
        get_mcp_tools()           — default: [] (no tools); None = every MCP tool, or a list

    Hooks are called inside __init__, so they must not read instance attributes that a
    subclass sets after calling super().__init__().
    """

    def __init__(
        self,
        session_id: str = None,
        rag_enabled: bool = False,
        llm_provider: str = None,
        llm_model: str = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self._rag_enabled = rag_enabled
        self.model = LlmModelGraph(
            create_llm(llm_provider or Config.LLM_PROVIDER, llm_model or Config.LLM_MODEL_NAME),
            system_prompt=self.get_system_prompt(),
            rag_context_manager=self.get_rag_context_manager(),
            tools=self.get_mcp_tools(),
            agent_name=self.__class__.__name__,
            rag_enabled=rag_enabled,
            history_store=shared_session_store(Config.SESSION_DB_FILE),
        )
        logger.debug("%s initialised (session=%s)", self.__class__.__name__, self.session_id)

    # Hooks

    def get_system_prompt(self) -> str:
        return load_prompt(Path(inspect.getfile(type(self))).parent / "prompts" / "System_Prompt.txt")

    def get_rag_context_manager(self):
        return RagContextManager(Embedder()) if self._rag_enabled else NullRagContextManager()

    def get_mcp_tools(self) -> list | None:
        return []

    # Public interface

    def ask(self, prompt: str, stream: bool = False, session_id: str = None):
        sid = session_id or self.session_id
        if stream:
            return self.model.ask_stream(prompt, session_id=sid)
        return self.model.ask(prompt, session_id=sid)

    async def aask(self, prompt: str, session_id: str = None) -> str:
        """Async ask() — what the orchestrator calls. Blocking answer, no streaming.

        An agent that answers in its own ask() but has no aask() of its own gets that
        ask() run in a worker thread — never the plain LLM pipeline it bypasses.
        """
        sid = session_id or self.session_id
        if type(self).ask is not BaseAgent.ask and type(self).aask is BaseAgent.aask:
            return await asyncio.to_thread(self.ask, prompt, session_id=sid)
        return await self.model.aask(prompt, session_id=sid)

    # Pending-confirmation hooks — no-ops unless an agent keeps pending state.

    def pending_summary(self, session_id: str) -> str | None:
        return None

    def confirm_pending(self, session_id: str) -> str:
        return ""

    def discard_pending(self, session_id: str) -> None:
        return None

    def close(self) -> None:
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
