"""Streamlit chat UI over the REST API (start `python -m clients.rest.main` first).

This file plays two roles: `python -m clients.streamlit.main` launches the Streamlit
server on it, and Streamlit then re-runs it as the page script on every interaction.
"""
import json
import subprocess
import sys
from pathlib import Path

# Streamlit runs this file as a plain script, so the project root isn't on sys.path yet.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests
import streamlit as st
from streamlit import runtime

from clients.base import BaseClient
from clients.streamlit.chat_store import ChatStore
from core.src.config.config import Config

AGENT_API_URL = f"{Config.REST_API_URL.rstrip('/')}/ask"
_BOT = ":material/smart_toy:"
_USER = ":material/person:"
_COLOURS = ("violet", "blue", "green", "orange", "red")
_CARDS_PER_ROW = 3
_STORE = ChatStore(Config.STREAMLIT_CHAT_FILE)
_DISCLAIMER = (
    "Private tool, built for Uzair Kamal's personal use only. "
    "It is not a general-purpose or public service."
)
# Keep everything inside the screen width on phones (also when embedded in the
# my-stats "Agents" tab): wide tables and code scroll inside the message, long words
# and starter prompts wrap, and the page itself never scrolls sideways.
_MOBILE_CSS = """<style>
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stAppScrollToBottomContainer"] { overflow-x: hidden; }
[data-testid="stChatMessageContent"], [data-testid="stMarkdownContainer"] {
  min-width: 0; max-width: 100%; overflow-wrap: anywhere;
}
[data-testid="stMarkdownContainer"] table { display: block; max-width: 100%; overflow-x: auto; }
[data-testid="stMarkdownContainer"] pre { max-width: 100%; overflow-x: auto; }
[data-testid="stButton"] button { max-width: 100%; }
[data-testid="stButton"] button p { white-space: normal; overflow-wrap: anywhere; }
/* iOS Safari zooms into any text field under 16px when it is tapped, pushing the
   send button off screen — keep text fields at 16px on touch screens. */
@media (pointer: coarse) {
  [data-testid="stChatInputTextArea"], input, textarea { font-size: 16px !important; }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {
    padding-left: 1rem; padding-right: 1rem;
  }
}
</style>"""


def _get_agent_response(user_query: str) -> str:
    # Send back the session id the API returned, so the chat is one conversation
    params = {"prompt": user_query}
    if st.session_state.get("session_id"):
        params["session_id"] = st.session_state["session_id"]
    headers = {"X-API-Key": Config.API_KEY} if Config.API_KEY else None
    with st.spinner("Thinking…"):
        try:
            response = requests.get(AGENT_API_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get("session_id"):
                st.session_state["session_id"] = data["session_id"]
            return data.get("response", "No response from agent.")
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to the Agent API. Please ensure the REST server is running."
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"


def _agents() -> list[dict]:
    """The enabled agents in agents.json — shown in the sidebar and as starter prompts."""
    try:
        specs = json.loads(Path(Config.AGENTS_FILE).read_text(encoding="utf-8"))["agents"]
    except (OSError, ValueError, KeyError):
        return []   # the chat still works without the list
    return [spec for spec in specs if spec.get("enabled", True)]


def _icon(agent: dict, position: int) -> str:
    """The agent's icon, in one of the theme's accent colours (by position in the list)."""
    return f":{_COLOURS[position % len(_COLOURS)]}[:material/{agent.get('icon', 'smart_toy')}:]"


def _new_chat() -> None:
    st.session_state.messages = []
    st.session_state.pop("session_id", None)
    _STORE.clear()


def _resume_chat() -> None:
    """On a new browser session, pick up the saved chat (see chat_store.py)."""
    if "messages" in st.session_state:
        return
    saved = _STORE.load()
    st.session_state.messages = saved.messages
    if saved.session_id:
        st.session_state["session_id"] = saved.session_id


def _name(agent: dict) -> str:
    return agent.get("display_name", agent["name"])


def _sidebar(agents: list[dict]) -> None:
    with st.sidebar:
        st.title(":primary[UK] Agents", icon=_BOT)
        st.caption("One chat, many agents. Your message goes to the one that can handle it.")
        st.button("New chat", icon=":material/add_comment:", type="primary", width="stretch", on_click=_new_chat)
        if agents:
            st.subheader("Agents")
        for position, agent in enumerate(agents):
            st.markdown(f"{_icon(agent, position)} {_name(agent)}", help=agent.get("summary"))


def _welcome(agents: list[dict]) -> str | None:
    """The empty-chat screen: a card per agent with a prompt to try.  Returns the one picked."""
    st.title("How can I :primary[help]?", text_alignment="center")
    st.caption("Ask anything below, or start with one of these.", text_alignment="center")
    st.space("small")
    picked = None
    starters = [(position, agent) for position, agent in enumerate(agents) if agent.get("example")]
    for row_start in range(0, len(starters), _CARDS_PER_ROW):
        row = starters[row_start:row_start + _CARDS_PER_ROW]
        for column, (position, agent) in zip(st.columns(_CARDS_PER_ROW), row):
            with column.container(border=True, height="stretch"):
                st.markdown(f"{_icon(agent, position)} **{_name(agent)}**")
                if st.button(agent["example"], key=f"starter_{agent['name']}", type="tertiary"):
                    picked = agent["example"]
    return picked


def _footer() -> None:
    """The usage disclaimer, pinned under the chat box so it is on screen at all times."""
    st.bottom.caption(_DISCLAIMER, text_alignment="center")


def _show(message: dict) -> None:
    is_user = message["role"] == "user"
    with st.chat_message(message["role"], avatar=_USER if is_user else _BOT):
        if message["content"].startswith("Error:"):
            st.error(message["content"], icon=":material/error:")
        else:
            st.markdown(message["content"])


def _run_ui() -> None:
    st.set_page_config(page_title="UK Agents", page_icon=_BOT)
    st.html(_MOBILE_CSS)
    _resume_chat()
    agents = _agents()
    _sidebar(agents)

    prompt = st.chat_input("Message UK Agents", submit_mode="disable")
    _footer()
    if not st.session_state.messages:
        welcome = st.empty()
        with welcome.container():
            prompt = _welcome(agents) or prompt
        if prompt:
            welcome.empty()

    for message in st.session_state.messages:
        _show(message)

    if prompt:
        user_message = {"role": "user", "content": prompt}
        _show(user_message)
        with st.chat_message("assistant", avatar=_BOT):
            reply = {"role": "assistant", "content": _get_agent_response(prompt)}
        st.session_state.messages += [user_message, reply]
        _STORE.save(st.session_state.get("session_id"), st.session_state.messages)
        st.rerun()   # redraw from history, so the reply renders like every other message


class StreamlitClient(BaseClient):
    """Launches the Streamlit server on this file."""

    def start(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__))],
            cwd=_PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    if runtime.exists():   # Streamlit is running this file as the page script
        _run_ui()
    else:
        StreamlitClient().start()
