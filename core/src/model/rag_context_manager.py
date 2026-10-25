from ..config.config import Config
from .embedder import Embedder
from .context_builder import build_context_from_listings


class RagContextManager:
    """Manages the retrieval and building of context for RAG prompts."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def get_context(self, user_prompt: str, k: int | None = None) -> str:
        """
        Retrieves relevant listings based on a user prompt and builds a
        formatted context string.  k defaults to RAG_K.
        """
        listings = self.embedder.search(user_prompt, k=k or Config.RAG_K)
        return build_context_from_listings(listings)
