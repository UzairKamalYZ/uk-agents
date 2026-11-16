import json
import logging
from pathlib import Path

from core.src.base_agent import BaseAgent
from core.src.config.config import Config

logger = logging.getLogger(__name__)


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

    No RAG, no tools.  It knows the other agents: the list in its prompt comes from
    agents.json, so it never goes stale.
    """

    def get_system_prompt(self) -> str:
        return super().get_system_prompt().replace("{agents}", agent_catalog(Config.AGENTS_FILE))
