"""
Pytest configuration for core/tests/.

Stubs out heavy third-party packages (pinecone, sentence_transformers,
langchain_anthropic, langchain_community) so that test modules can be collected
and imported without them installed.
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 0. Minimal environment variables required by Config at class-body import time
# ---------------------------------------------------------------------------
os.environ.setdefault("STORE_TYPE", "pinecone")
os.environ.setdefault("LLM_MODEL_NAME", "test-model")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("MEMORY_FILE", "memory.json")
os.environ.setdefault("URLS_FILE", "../urls.txt")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_ENVIRONMENT", "test-env")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("PROMPT_FILE", "agents/property/prompts/System_Prompt.txt")
os.environ.setdefault("INTERACTION_FILE", "agents/property/prompts/interaction.json")
os.environ.setdefault("REFORMULATION_PROMPT", "agents/property/prompts/reformulated_prompt.txt")

# ---------------------------------------------------------------------------
# Stub heavy third-party packages before any source module imports them
# ---------------------------------------------------------------------------

def _stub(name: str) -> MagicMock:
    """Register a MagicMock in sys.modules under *name* and return it."""
    mock = MagicMock()
    sys.modules[name] = mock
    return mock


# pinecone — used by persistence/pinecone_store.py
_pinecone = _stub("pinecone")
_pinecone.Pinecone = MagicMock
_pinecone.ServerlessSpec = MagicMock

# sentence_transformers — used by model/embedder.py
_st = _stub("sentence_transformers")
_st.SentenceTransformer = MagicMock

# langchain_anthropic — used by model/llm_factory.py
_stub("langchain_anthropic")

# langchain_community — used by model/session_manager.py
_lc_comm = _stub("langchain_community")
_lc_hist = _stub("langchain_community.chat_message_histories")
_lc_hist.SQLChatMessageHistory = MagicMock
_lc_comm.chat_message_histories = _lc_hist
