from core.src.base_agent import BaseAgent
from core.src.model.mcp_tools import _mcp


class FinanceAgent(BaseAgent):
    """Stock analysis and investment advice agent — uses the stocks MCP server."""

    def get_mcp_tools(self) -> list:
        return _mcp.langchain_tools(server_name="stocks")
