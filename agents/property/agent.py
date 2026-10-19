import json
import logging

from rich.console import Console

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.config.config import Config
from core.src.model.embedder import Embedder
from core.src.model.rag_context_manager import RagContextManager
from core.src.model.mcp_tools import _mcp
from core.src.utils import load_prompt
from .scraping.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class PropertySearchAgent(BaseAgent):
    """Property-search agent."""

    def get_system_prompt(self) -> str:
        return load_prompt(Config.PROMPT_FILE)

    def get_rag_context_manager(self):
        if self._rag_enabled:
            return RagContextManager(Embedder())
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return _mcp.langchain_tools(server_name="finance")

    def __init__(self, session_id: str = None, rag_enabled: bool = False,
                 llm_provider: str = None, llm_model: str = None):
        super().__init__(session_id, rag_enabled=rag_enabled,
                         llm_provider=llm_provider, llm_model=llm_model)
