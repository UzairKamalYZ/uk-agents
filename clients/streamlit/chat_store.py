"""The Streamlit chat, saved to disk so it survives page reloads, new browser tabs and restarts.

Streamlit keeps ``st.session_state`` only for one browser session, so on its own every reload
starts an empty chat. This is a single-user tool, so there is one saved chat: every new
browser session resumes it, and "New chat" clears it. The REST API keeps the agents' side of
the conversation under the same ``session_id``, so a resumed chat really is the same chat.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class SavedChat:
    session_id: str | None = None
    messages: list[dict] = field(default_factory=list)


class ChatStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> SavedChat:
        """The saved chat, or an empty one when there is none or the file is unreadable."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return SavedChat()
        except (OSError, ValueError) as e:
            log.warning("Ignoring unreadable chat file %s: %s", self._path, e)
            return SavedChat()
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("messages", [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        session_id = data.get("session_id")
        return SavedChat(session_id if isinstance(session_id, str) else None, messages)

    def save(self, session_id: str | None, messages: list[dict]) -> None:
        """Write atomically (temp file + rename), so a crash never leaves half a file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"session_id": session_id, "messages": messages}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except OSError as e:   # the chat still works, it just won't be resumed
            log.warning("Could not save the chat to %s: %s", self._path, e)

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            log.warning("Could not clear the chat file %s: %s", self._path, e)
