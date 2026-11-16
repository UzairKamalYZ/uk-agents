"""Streamlit chat UI over the REST API (start `python -m clients.rest.main` first).

This file plays two roles: `python -m clients.streamlit.main` launches the Streamlit
server on it, and Streamlit then re-runs it as the page script on every interaction.
"""
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
from core.src.config.config import Config

AGENT_API_URL = f"{Config.REST_API_URL.rstrip('/')}/ask"


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


def _run_ui() -> None:
    st.set_page_config(page_title="UK Agents", layout="wide")
    st.title("UK Agents 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": _get_agent_response(prompt)})
        st.rerun()


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
