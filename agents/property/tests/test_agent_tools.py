from unittest.mock import MagicMock, patch

from agents.property.agent import PropertySearchAgent


def test_get_mcp_tools_is_scoped_to_finance_server():
    with patch("agents.property.agent._mcp") as mcp:
        mcp.langchain_tools.return_value = ["t"]
        result = PropertySearchAgent.get_mcp_tools(MagicMock())
    mcp.langchain_tools.assert_called_once_with(server_name="finance")
    assert result == ["t"]
