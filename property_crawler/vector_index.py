"""
The Pinecone index the crawler writes to, with the sentence-transformers model that
embeds listings. The searching application must embed queries with the same model
(SENTENCE_TRANSFORMER_MODEL) and search the same index (PINECONE_INDEX_NAME), default
namespace.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

DELETE_BATCH = 1000   # Pinecone deletes at most 1000 ids per call


class IncompatibleIndexError(RuntimeError):
    """The named Pinecone index exists but cannot hold this crawler's vectors."""


def check_compatible(description, index_name: str, dimension: int) -> None:
    """Raise IncompatibleIndexError unless the index takes plain vectors of `dimension`."""
    schema = getattr(description, "schema", None)
    if schema is not None and getattr(schema, "fields", None):
        raise IncompatibleIndexError(
            f"Pinecone index {index_name!r} has a document schema (fields: "
            f"{', '.join(schema.fields)}) and only accepts writes through the documents "
            "API. Set PINECONE_INDEX_NAME to a new index name; the crawler creates it.")
    if description.dimension != dimension:
        raise IncompatibleIndexError(
            f"Pinecone index {index_name!r} has dimension {description.dimension}, but "
            f"the embedding model produces {dimension}. Set PINECONE_INDEX_NAME to a new "
            "index name (the crawler creates it) or use the matching model.")


class VectorIndex:

    def __init__(self, api_key: str, index_name: str, model_name: str):
        if not api_key or not index_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set")
        self._api_key = api_key
        self._index_name = index_name
        self._model_name = model_name
        self._model = None
        self._index = None

    @property
    def model(self):
        if self._model is None:   # lazy: loading the model takes a while
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def index(self):
        if self._index is None:
            from pinecone import Pinecone, ServerlessSpec
            client = Pinecone(api_key=self._api_key)
            dimension = self.model.get_sentence_embedding_dimension()
            if self._index_name not in client.list_indexes().names():
                logger.info("[crawler] creating Pinecone index %s", self._index_name)
                client.create_index(
                    name=self._index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                check_compatible(client.describe_index(self._index_name), self._index_name, dimension)
            self._index = client.Index(self._index_name)
        return self._index

    def upsert(self, documents: dict[str, dict]) -> None:
        """documents: vector id → metadata, whose "text" is what gets embedded."""
        ids = list(documents)
        texts = [documents[i]["text"] for i in ids]
        embeddings = np.asarray(self.model.encode(texts), dtype="float32")
        self.index.upsert(vectors=[
            {"id": i, "values": embeddings[n].tolist(), "metadata": documents[i]}
            for n, i in enumerate(ids)
        ])

    def list_ids(self):
        """Every vector id, in pages (serverless indexes page through ids)."""
        return self.index.list()

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), DELETE_BATCH):
            self.index.delete(ids=ids[start : start + DELETE_BATCH])
