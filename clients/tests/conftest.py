"""
Pytest configuration for clients/tests/.

Client modules are imported for real (fastapi, python-telegram-bot, streamlit
and the orchestrator are regular dependencies); each test patches what it
needs — OrchestratorAgent, Config, uvicorn.run, subprocess — so nothing here
touches sys.modules or leaks into other test folders.
"""

import pytest

from core.src.config.config import Config


@pytest.fixture(autouse=True)
def _isolated_streamlit_chat_file(tmp_path, monkeypatch):
    """Every test gets its own saved-chat file, so no test reads or writes the real one
    (the Streamlit page builds its ChatStore from Config on each AppTest run)."""
    monkeypatch.setattr(Config, "STREAMLIT_CHAT_FILE", str(tmp_path / "streamlit-chat.json"))
