"""
orchestrator/tracing.py — makes every routing decision visible in the logs and in LangSmith.

Each user turn gets a short request_id (see MultiAgentGraph._initial_state).  It is
written into every routing log line and attached to the LangSmith trace as metadata,
so a log line can be matched to its trace and vice versa.

record_route() emits one small LangSmith run per decision, named "route → <agent>"
and tagged "agent:<agent>" / "routed_by:<node>", nested under the node that made
the decision.  In LangSmith, filter by tag "agent:expenditure_tracker" to see every
request that agent handled.  With LANGCHAIN_TRACING_V2=false, traceable is a no-op
and only the log line remains.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@traceable(name="route", run_type="chain", tags=["routing"])
def _route_run(decision: dict[str, Any]) -> dict[str, Any]:
    """The decision is the run's input and output — that is all the trace needs."""
    return decision


def record_route(source: str, agent: str, reason: str, request_id: str | None) -> None:
    """Log and trace one routing decision: <source> node sent the turn to <agent>."""
    decision = {"agent": agent, "routed_by": source, "reason": reason, "request_id": request_id}
    logger.info("[route] request=%s %s → %s (%s)", request_id, source, agent, reason or "no reason given")
    try:
        _route_run(
            decision,
            langsmith_extra={
                "name": f"route → {agent}",
                "tags": [f"agent:{agent}", f"routed_by:{source}"],
                "metadata": decision,
            },
        )
    except Exception:  # tracing must never break a request
        logger.exception("[route] could not record LangSmith run")


def tag_current_trace(tags: list[str], metadata: dict[str, Any]) -> None:
    """Add tags/metadata to the enclosing LangSmith run (the orchestrator.request trace)."""
    run = get_current_run_tree()
    if run is None:
        return
    run.add_tags(tags)
    run.add_metadata(metadata)
