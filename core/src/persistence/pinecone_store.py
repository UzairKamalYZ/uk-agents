# vector_store/pinecone_store.py
from pinecone import Pinecone, ServerlessSpec
import uuid
from .vector_store import VectorStore

class PineconeStore(VectorStore):

    def __init__(self, index_name: str, dim: int, api_key: str):
        # Serverless: the region comes from the index spec, not a client "environment"
        # (pinecone>=3 rejects environment=)
        pinecone = Pinecone(api_key=api_key)
        if index_name not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', region='us-east-1')
            )
        self.index = pinecone.Index(index_name)

    def add(self, vectors, metadata):
        items = []
        for vector, meta in zip(vectors, metadata):
            items.append((str(uuid.uuid4()), vector.tolist(), meta))
        self.index.upsert(items)
    def add_vectors(self, vectors):
        self.index.upsert(vectors)
    def search(self, query_vector, k):
        results = self.index.query(
            vector=query_vector[0].tolist(),
            top_k=k,
            include_metadata=True
        )
        return [m["metadata"] for m in results["matches"]]
