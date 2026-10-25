"""
orchestrator/tracing.py — makes the routed agents visible on the LangSmith trace.

OrchestratorAgent.ask()/aask() run as the "orchestrator.request" trace.  As each
planned agent is picked (on_route), tag_current_trace() adds the tag
"agent:<name>" and the running "agents" list to that trace, so filtering by tag
"agent:expenditure_tracker" in LangSmith shows every request that agent handled.
With LANGCHAIN_TRACING_V2=false there is no current run and this is a no-op.
"""
from __future__ import annotations

from typing import Any

from langsmith.run_helpers import get_current_run_tree


def tag_current_trace(tags: list[str], metadata: dict[str, Any]) -> None:
    """Add tags/metadata to the enclosing LangSmith run (the orchestrator.request trace)."""
    run = get_current_run_tree()
    if run is None:
        return
    run.add_tags(tags)
    run.add_metadata(metadata)
