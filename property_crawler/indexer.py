"""Embeds crawled listings and upserts them into the vector index, in batches."""
import logging

from .extractor import Listing
from .seen_store import utc_now

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


class ListingIndexer:
    """vector_index: anything with upsert({id: metadata-with-"text"}) — VectorIndex in production."""

    def __init__(self, vector_index, batch_size: int = BATCH_SIZE):
        self._vector_index = vector_index
        self._batch_size = batch_size

    def index(self, listings: list[Listing]) -> int:
        """Upsert the listings in batches; returns how many were written."""
        crawled_at = utc_now()
        written = 0
        for start in range(0, len(listings), self._batch_size):
            batch = listings[start : start + self._batch_size]
            self._vector_index.upsert({l.id: l.metadata(crawled_at) for l in batch})
            written += len(batch)
        return written
