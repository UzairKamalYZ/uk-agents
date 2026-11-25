"""Embeds crawled listings and upserts them into the property agent's vector index."""
import logging

from .extractor import Listing
from .seen_store import utc_now

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class ListingIndexer:

    def __init__(self, embedder=None, batch_size: int = BATCH_SIZE):
        self._embedder = embedder
        self._batch_size = batch_size

    @property
    def embedder(self):
        if self._embedder is None:   # lazy: loading the embedding model takes a while
            from core.src.model.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def index(self, listings: list[Listing]) -> int:
        """Upsert the listings in batches; returns how many were written."""
        crawled_at = utc_now()
        written = 0
        for start in range(0, len(listings), self._batch_size):
            batch = listings[start : start + self._batch_size]
            documents = {l.id: l.metadata(crawled_at) for l in batch}
            vectors = self.embedder.embed_documents_to_vectors(documents)
            self.embedder.save_vectors_in_store(vectors)
            written += len(batch)
        return written
