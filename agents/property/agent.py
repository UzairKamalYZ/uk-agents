from core.src.base_agent import BaseAgent
from core.src.config.config import Config
from core.src.model.mcp_tools import _mcp
from core.src.utils import load_prompt


class PropertySearchAgent(BaseAgent):
    """Property search over the indexed listings (RAG), with the finance MCP server's tools."""

    def get_system_prompt(self) -> str:
        return load_prompt(Config.PROMPT_FILE)

    def get_mcp_tools(self) -> list:
        return _mcp.langchain_tools(server_name="finance")
