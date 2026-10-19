from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.src.config.config import Config

logger = logging.getLogger(__name__)

_AGENTS_FILE = Path(Config.AGENTS_FILE)


@dataclass(frozen=True)
class AgentEntry:
    """Holds everything the graph needs to know about one agent."""
    name: str          # routing token, e.g. "property_agent"
    description: str   # shown to the supervisor LLM so it knows when to route here
    instance: object   # live BaseAgent-compatible object (must have .ask())
    llm_provider: str  # provider used by this agent (e.g. "anthropic")
    llm_model: str     # model name used by this agent (e.g. "claude-opus-5-5")
    owner_only: bool = False  # True → only the bot owner may use this agent
    display_name: str = ""    # shown to users, e.g. "Expenditure Tracker" (defaults to a title-cased name)
    remarks: tuple[str, ...] = field(default_factory=tuple)  # one-liners shown while the agent works

    @property
    def label(self) -> str:
        return self.display_name or self.name.replace("_", " ").title()


def load_agents(agents_file: Path = _AGENTS_FILE) -> list[AgentEntry]:
    """
    Parse agents.json, skip disabled entries, and return instantiated AgentEntry list.

    Steps
    -----
    1. Read and parse agents.json.
    2. Skip any entry where "enabled" is false (defaults to true if omitted).
    3. For each enabled entry, split the class path into module + class name.
    4. Import the module dynamically with importlib.
    5. Instantiate the class with rag_enabled, llm_provider, and llm_model from the spec.
    6. Wrap in an AgentEntry and collect.
    """
    raw = json.loads(agents_file.read_text(encoding="utf-8"))
    entries: list[AgentEntry] = []

    for spec in raw["agents"]:

        # Step 2 — honour the enabled flag; missing flag means enabled
        if not spec.get("enabled", True):
            logger.info("[agent_registry] skipping disabled agent: %s", spec["name"])
            continue

        # Step 3 — split "agents.property.agent.LocalAgent" into:
        #           module_path = "agents.property.agent"
        #           class_name  = "LocalAgent"
        module_path, class_name = spec["class"].rsplit(".", 1)

        # Step 4 — dynamic import; equivalent to: from <module_path> import <class_name>
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Step 5 — instantiate; pass rag_enabled and per-agent LLM config
        rag_enabled = spec.get("rag", False)
        llm_provider = spec.get("llm_provider", Config.LLM_PROVIDER)
        llm_model = spec.get("llm_model", Config.LLM_MODEL_NAME)
        instance = cls(rag_enabled=rag_enabled, llm_provider=llm_provider, llm_model=llm_model)
        logger.info(
            "[agent_registry] loaded %s → %s (provider=%s model=%s)",
            spec["name"], spec["class"], llm_provider, llm_model,
        )

        # Step 6 — collect
        entries.append(AgentEntry(
            name=spec["name"],
            description=spec["description"],
            instance=instance,
            llm_provider=llm_provider,
            llm_model=llm_model,
            owner_only=spec.get("owner_only", False),
            display_name=spec.get("display_name", ""),
            remarks=tuple(spec.get("remarks", ())),
        ))

    return entries
