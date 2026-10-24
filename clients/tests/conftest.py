"""
Pytest configuration for clients/tests/.

Client modules are imported for real (fastapi, python-telegram-bot, streamlit
and the orchestrator are regular dependencies); each test patches what it
needs — OrchestratorAgent, Config, uvicorn.run, subprocess — so nothing here
touches sys.modules or leaks into other test folders.
"""
