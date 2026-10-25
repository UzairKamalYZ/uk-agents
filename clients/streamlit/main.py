import os
import subprocess
import sys
from pathlib import Path

# Ensure the project root is importable regardless of how this file is launched:
# `python -m clients.streamlit.main`, `streamlit run`, or via subprocess.
# Must happen before any local package import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import requests
import random

from clients.base import BaseClient

# ── Streamlit UI ─────────────────────────────────────────────────────────────
# Everything below this block is executed by Streamlit on every app rerun.
# Do not call StreamlitClient().start() here — use `streamlit run` instead.

AGENT_API_URL = "http://localhost:8000/ask"

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "What do you call a fake noodle? An impasta!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "How do you organize a space party? You planet!",
]


def _get_agent_response(user_query: str) -> str:
    # Send back the session id the API returned, so the chat is one conversation
    params = {"prompt": user_query}
    if st.session_state.get("session_id"):
        params["session_id"] = st.session_state["session_id"]
    with st.spinner(random.choice(JOKES)):
        try:
            response = requests.get(AGENT_API_URL, params=params)
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
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.rawpixel.com/image/5908905/free-apartment-image-public-domain-cc0-photo");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(" Your own mini property agent 🕵️‍♂️🏡")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you find your next property?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "agent", "content": _get_agent_response(prompt)})
        st.rerun()


# ── BaseClient implementation ─────────────────────────────────────────────────

class StreamlitClient(BaseClient):
    """Launches the Streamlit UI server as a subprocess."""

    def start(self) -> None:
        env = os.environ.copy()
        # Make the project root importable inside the subprocess so that
        # `from clients.base import BaseClient` resolves when Streamlit runs
        # this script outside a `python -m` context.
        root = str(_PROJECT_ROOT)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
        # Signal the subprocess that it is the Streamlit worker, not the launcher,
        # so the __main__ guard calls _run_ui() instead of starting again.
        env["_STREAMLIT_APP"] = "1"
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__))],
            env=env,
            check=True,
        )


# When launched via `python -m clients.streamlit.main` (the launcher):
#   _STREAMLIT_APP is not set → call start(), which spawns the subprocess.
# When Streamlit re-executes this file inside the subprocess (the worker):
#   _STREAMLIT_APP="1" → call _run_ui() to render the UI.
if __name__ == "__main__":
    if os.environ.get("_STREAMLIT_APP"):
        _run_ui()
    else:
        StreamlitClient().start()
