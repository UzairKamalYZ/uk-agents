import logging
import uuid
from abc import ABC, abstractmethod

from .config.config import Config
from .model.llm_factory import create_llm
from .model.llm_model_graph import LlmModelGraph
from .model.session_manager import shared_session_store

logger = logging.getLogger(__name__)


class NullRagContextManager:
    """No-op RAG context manager — returns empty string, skips all vector search."""

    def get_context(self, user_prompt: str, k: int = 5) -> str:
        return ""


class BaseAgent(ABC):
    """
    Shared base for all domain agents.

    Owns LLM creation, LangGraph pipeline wiring, session identity, and the
    ask / stream / close interface.  Subclasses must implement all three hooks:

        get_system_prompt()       — return the fully-loaded prompt string
        get_rag_context_manager() — return a RAG manager or NullRagContextManager
        get_mcp_tools()           — return None (all), [] (none), or a filtered list

    Note: hooks are called inside __init__ via Python's MRO.  Do not read
    subclass instance attributes inside a hook unless they were set before
    calling super().__init__().
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
        system_prompt = self.get_system_prompt()
        rag_context_manager = self.get_rag_context_manager()
        mcp_tools = self.get_mcp_tools()
        llm = create_llm(
            llm_provider or Config.LLM_PROVIDER,
            llm_model or Config.LLM_MODEL_NAME,
        )
        self.model = LlmModelGraph(
            llm,
            system_prompt=system_prompt,
            rag_context_manager=rag_context_manager,
            tools=mcp_tools,
            agent_name=self.__class__.__name__,
            rag_enabled=self._rag_enabled,
            history_store=shared_session_store(Config.SESSION_DB_FILE),
        )
        logger.debug("%s initialised (session=%s)", self.__class__.__name__, self.session_id)

    # ------------------------------------------------------------------
    # Hook points — override in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_rag_context_manager(self): ...

    @abstractmethod
    def get_mcp_tools(self) -> list | None: ...

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, prompt: str, stream: bool = False, session_id: str = None):
        logger.info("[%s] BaseAgent.ask() entered", self.__class__.__name__)
        sid = session_id or self.session_id
        if stream:
            return self.model.ask_stream(prompt, session_id=sid)
        return self.model.ask(prompt, session_id=sid)

    async def aask(self, prompt: str, session_id: str = None) -> str:
        """Async ask() — what the orchestrator calls. Blocking answer, no streaming."""
        logger.info("[%s] BaseAgent.aask() entered", self.__class__.__name__)
        return await self.model.aask(prompt, session_id=session_id or self.session_id)

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
