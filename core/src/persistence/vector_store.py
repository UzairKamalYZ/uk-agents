from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):

    @abstractmethod
    def add_vectors(self, vectors: list[dict]) -> None:
        """Upsert [{"id", "values", "metadata"}, ...]."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        """Metadata of the k nearest vectors."""
