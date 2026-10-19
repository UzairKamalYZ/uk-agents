import json
import logging
from pathlib import Path

from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.config.config import Config
from core.src.utils import load_prompt

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "System_Prompt.txt"


def agent_catalog(agents_file: str | Path) -> str:
    """'- <display name>: <summary>' for every enabled agent in agents.json."""
    try:
        specs = json.loads(Path(agents_file).read_text(encoding="utf-8"))["agents"]
    except (OSError, ValueError, KeyError):
        logger.exception("ConversationalAgent: could not read the agent list from %s", agents_file)
        return "(the agent list is unavailable right now)"
    return "\n".join(
        f"- {s.get('display_name', s['name'])}: {s.get('summary') or s.get('description', '')}"
        for s in specs if s.get("enabled", True)
    )


class ConversationalAgent(BaseAgent):
    """
    Lightweight agent for greetings and general small talk.

    - No RAG: no vector search.
    - No MCP tools: no finance/currency tools bound to the LLM.
    - Knows the other agents: the list comes from agents.json, so it never goes stale.
    """

    def get_system_prompt(self) -> str:
        return load_prompt(_PROMPT_FILE).replace("{agents}", agent_catalog(Config.AGENTS_FILE))

    def get_rag_context_manager(self):
        return NullRagContextManager()

    def get_mcp_tools(self) -> list:
        return []
