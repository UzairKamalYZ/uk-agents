"""
core/src/output_guard.py — keeps tool-call markup out of answers shown to users.

A model that cannot actually call a tool (it is missing from its session, or the
call was cut short) sometimes writes the call out as text instead:

    <invoke name="Skill"><parameter name="skill">fbr-iris-filing</parameter></invoke>

That text would reach the user as the answer.  scrub() removes such blocks —
tool calls and results, parameters, thinking and system reminders, with or
without a namespace prefix — and reports which tags it found, so the caller
can log the leak and fall back to a safe reply when nothing is left.
StreamScrubber does the same for an answer streamed token by token.

Usage
-----
    result = scrub(answer)
    if result.found:
        logger.warning("stripped %s", result.found)
    text = result.text or WITHHELD_REPLY
"""
from __future__ import annotations

import re
from dataclasses import dataclass

WITHHELD_REPLY = "⚠️ Sorry, I couldn't put together a proper answer to that. Please try again."

# Outer tags first, so a function_calls block goes as a whole before its invokes are looked at
_TAGS = (
    "function_calls", "function_results", "invoke", "tool_use", "tool_result", "tool_call",
    "thinking", "system-reminder", "parameter",
)
_NAME = r"(?:[a-z_]+:)?(?:" + "|".join(re.escape(t) for t in _TAGS) + r")"
_BLOCK = re.compile(rf"<\s*({_NAME})\b[^>]*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_OPEN = re.compile(rf"<\s*({_NAME})\b[^>]*>", re.IGNORECASE)
_STRAY = re.compile(rf"<\s*/?\s*({_NAME})\b[^>]*>", re.IGNORECASE)
_BLANK_LINES = re.compile(r"\n\s*\n(\s*\n)+")


@dataclass(frozen=True)
class ScrubResult:
    text: str              # the answer without markup ("" when nothing else was in it)
    found: tuple[str, ...]  # tag names that were removed, e.g. ("invoke", "parameter")


def scrub(text: str) -> ScrubResult:
    """The answer with every tool-call / internal block removed, and what was removed."""
    found: set[str] = set()
    clean = _BLANK_LINES.sub("\n\n", _clean(text, found, final=True)).strip()
    return ScrubResult(clean, tuple(sorted(found)))


class StreamScrubber:
    """scrub() for streamed answers: feed() each token, then flush() once at the end.

    Text that might start a markup block is held back until the block closes (and is
    dropped) or turns out to be ordinary text, so markup never reaches the client.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._sent = 0
        self.found: set[str] = set()

    def feed(self, chunk: str) -> str:
        """The part of the answer that is now safe to send ("" while it is held back)."""
        self._raw += chunk
        return self._emit(_clean(self._safe_prefix(), self.found, final=False))

    def flush(self) -> str:
        """Whatever is left once the stream has ended; unfinished blocks are dropped."""
        return self._emit(_clean(self._raw, self.found, final=True))

    def _safe_prefix(self) -> str:
        """The raw text up to the first block that has not closed yet, or a tag still being written."""
        cut = len(self._raw)
        partial = self._raw.rfind("<")
        if partial > self._raw.rfind(">"):
            cut = partial
        for match in _OPEN.finditer(self._raw):
            if not re.search(rf"<\s*/\s*{re.escape(match.group(1))}\s*>", self._raw[match.end():], re.IGNORECASE):
                cut = min(cut, match.start())
                break
        return self._raw[:cut]

    def _emit(self, clean: str) -> str:
        delta = clean[self._sent:]
        self._sent = max(self._sent, len(clean))
        return delta


def _clean(text: str, found: set[str], final: bool) -> str:
    """Complete blocks out; when final, an unclosed block runs to the end; then stray tags out."""
    def drop(match: re.Match) -> str:
        found.add(_bare(match.group(1)))
        return ""

    previous = None
    while previous != text:  # repeat: removing an outer block can expose another
        previous, text = text, _BLOCK.sub(drop, text)
    if final and (match := _OPEN.search(text)) is not None:
        found.add(_bare(match.group(1)))  # a tool call cut short never closes
        text = text[:match.start()]
    return _STRAY.sub(drop, text)


def _bare(name: str) -> str:
    return name.split(":")[-1].lower()
