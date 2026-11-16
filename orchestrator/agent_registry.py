from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core.src.config.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentEntry:
    """Everything the graph needs to know about one agent."""
    name: str          # routing token, e.g. "property_agent"
    description: str   # shown to the router LLM so it knows when to route here
    instance: object   # the live agent (BaseAgent)
    llm_provider: str
    llm_model: str
    display_name: str = ""          # shown to users; defaults to a title-cased name
    remarks: tuple[str, ...] = ()   # one-liners shown while the agent works

    @property
    def label(self) -> str:
        return self.display_name or self.name.replace("_", " ").title()


def load_agents(agents_file: str | Path = Config.AGENTS_FILE) -> list[AgentEntry]:
    """Instantiate every enabled agent in agents.json ("enabled" defaults to true).

    An agent that cannot start (bad class path, missing key, unreachable service) is
    logged and left out, so one broken agent never takes the others down.
    """
    specs = json.loads(Path(agents_file).read_text(encoding="utf-8"))["agents"]
    entries: list[AgentEntry] = []
    for spec in specs:
        name = spec["name"]
        if not spec.get("enabled", True):
            logger.info("[agent_registry] skipping disabled agent: %s", name)
            continue
        llm_provider = spec.get("llm_provider", Config.LLM_PROVIDER)
        llm_model = spec.get("llm_model", Config.LLM_MODEL_NAME)
        try:
            module_path, class_name = spec["class"].rsplit(".", 1)
            agent_class = getattr(importlib.import_module(module_path), class_name)
            instance = agent_class(
                rag_enabled=spec.get("rag", False), llm_provider=llm_provider, llm_model=llm_model
            )
        except Exception:
            logger.exception("[agent_registry] %s could not start — left out", name)
            continue
        logger.info("[agent_registry] loaded %s → %s (provider=%s model=%s)",
                    name, spec["class"], llm_provider, llm_model)
        entries.append(AgentEntry(
            name=name,
            description=spec["description"],
            instance=instance,
            llm_provider=llm_provider,
            llm_model=llm_model,
            display_name=spec.get("display_name", ""),
            remarks=tuple(spec.get("remarks", ())),
        ))
    return entries
