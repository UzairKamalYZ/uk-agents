from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):

    @abstractmethod
    def add_vectors(self, vectors: list[dict]) -> None:
        """Upsert [{"id", "values", "metadata"}, ...]."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        """Metadata of the k nearest vectors."""

    @abstractmethod
    def list_ids(self):
        """Every vector id in the store, in pages (an iterable of lists of ids)."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete the vectors with these ids."""
