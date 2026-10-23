"""
Pytest configuration for clients/tests/.

Responsibilities:
1. Stub core.src.config.config so clients get predictable settings.
2. Stub orchestrator.agent_interface so that importing clients does not
   trigger the full orchestrator → graph → llm_factory → langchain_anthropic
   import chain.
3. Stub third-party packages that are absent from the minimal test environment:
   uvicorn, fastapi, and telegram.
"""

import sys
import importlib.machinery
from pathlib import Path
from unittest.mock import MagicMock

_PROJECT_ROOT = Path(__file__).parents[2]  # clients/tests/ → clients/ → project root


def _stub(name: str) -> MagicMock:
    """Register a MagicMock in sys.modules and return it.

    Sets __spec__ to a real ModuleSpec so that importlib.util.find_spec()
    does not raise ValueError (Python 3.13 + transformers compatibility).
    """
    mock = MagicMock()
    mock.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = mock
    return mock


# ---------------------------------------------------------------------------
# langchain_anthropic — keeps the LLM client out of client-layer tests.
# Must be stubbed before any orchestrator module is imported.
# ---------------------------------------------------------------------------
_stub("langchain_anthropic")

# ---------------------------------------------------------------------------
# orchestrator.agent_interface — clients import OrchestratorAgent from here.
# Stubbing the module stops Python following the deep import chain:
#   agent_interface → graph → llm_factory → langchain_anthropic → …
# ---------------------------------------------------------------------------
_orchestrator_interface = _stub("orchestrator.agent_interface")
_orchestrator_interface.OrchestratorAgent = MagicMock

# core.src.config.config — clients now import Config from here
_core_config_mod = _stub("core.src.config.config")
_Config = MagicMock()
_Config.API_KEY = ""
_Config.TELEGRAM_BOT_TOKEN = "test-token"
_Config.TELEGRAM_OWNER_ID = ""
_Config.CRON_SEARCH_PROMPT = "find properties"
# llm_model_graph.py reads these three at module level via os.environ.setdefault;
# they must be real strings or the call raises TypeError when both test suites run together.
_Config.LANGCHAIN_TRACING_V2 = "false"
_Config.LANGCHAIN_PROJECT = "uk-agents"
_Config.LANGCHAIN_API_KEY = ""
# MCP_FILE must point to the real mcp.json so that mcp_tools.py can load the
# registry correctly even when core.src.config.config is stubbed.
_Config.MCP_FILE = str(_PROJECT_ROOT / "mcp.json")
_core_config_mod.Config = _Config

# ---------------------------------------------------------------------------
# Third-party packages not installed in the minimal test environment
# ---------------------------------------------------------------------------

# uvicorn / fastapi
_stub("uvicorn")
_fastapi = _stub("fastapi")
_fastapi.FastAPI       = MagicMock
_fastapi.Depends       = MagicMock()
_fastapi.HTTPException = MagicMock
_fastapi.Security      = MagicMock()
_stub("fastapi.responses").StreamingResponse = MagicMock
_stub("fastapi.security").APIKeyHeader       = MagicMock

# streamlit — not installed in the minimal test environment
_stub("streamlit")

# telegram (python-telegram-bot)
_tg            = _stub("telegram")
_tg.Update     = MagicMock()
_tg_ext        = _stub("telegram.ext")
_tg_ext.ApplicationBuilder = MagicMock()
_tg_ext.CommandHandler     = MagicMock()
_tg_ext.MessageHandler     = MagicMock()
_ctx_types                 = MagicMock()
_ctx_types.DEFAULT_TYPE    = MagicMock()
_tg_ext.ContextTypes       = _ctx_types
_tg_ext.filters            = MagicMock()
