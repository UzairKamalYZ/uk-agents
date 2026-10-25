from unittest.mock import MagicMock, patch

from core.src.model.rag_context_manager import RagContextManager


def test_search_size_comes_from_rag_k():
    embedder = MagicMock()
    embedder.search.return_value = []
    with patch("core.src.model.rag_context_manager.Config.RAG_K", 8):
        RagContextManager(embedder).get_context("2-bed Warsaw")
    embedder.search.assert_called_once_with("2-bed Warsaw", k=8)


def test_explicit_k_overrides_rag_k():
    embedder = MagicMock()
    embedder.search.return_value = []
    RagContextManager(embedder).get_context("2-bed Warsaw", k=3)
    embedder.search.assert_called_once_with("2-bed Warsaw", k=3)
