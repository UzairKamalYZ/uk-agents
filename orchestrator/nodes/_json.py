"""
orchestrator/nodes/_json.py — tolerant JSON-object extraction for LLM replies.

Shared by the supervisor and normaliser nodes, whose LLMs are asked to reply
with a single JSON object but sometimes wrap it in markdown fences or prose.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Strips markdown code fences (```json ... ```) before JSON parsing
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Fallback: extracts the first {...} block when full JSON parsing fails
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_json_object(text: str, required_key: str) -> dict[str, Any] | None:
    """
    Return the JSON object in *text* that contains *required_key*, else None.

    Priority:
      1. Strip markdown fences.
      2. Full JSON parse of the cleaned text.
      3. Regex extraction of the first {...} block.
    """
    cleaned = _FENCE_RE.sub(r"\1", text).strip()

    candidates = [cleaned]
    m = _JSON_RE.search(cleaned)
    if m:
        candidates.append(m.group())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and required_key in data:
            return data
    return None
