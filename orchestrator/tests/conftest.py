"""
Pytest configuration for orchestrator/tests/.

Sets the minimal environment variables Config needs at import time (same as
core/tests/conftest.py).  langchain_anthropic and claude_agent_sdk are real
installed packages and are deliberately NOT stubbed: stubs registered in
sys.modules are process-wide and would break core/tests, which exercise the
real claude_agent_sdk surface.
"""
import os

os.environ.setdefault("STORE_TYPE", "pinecone")
os.environ.setdefault("LLM_MODEL_NAME", "test-model")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("URLS_FILE", "../urls.txt")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
