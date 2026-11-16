from ..config.config import Config


def create_vector_store(store_type: str, dim: int, config: dict):
    if store_type == "pinecone":
        from .pinecone_store import PineconeStore  # lazy: only RAG agents need the pinecone package
        return PineconeStore(
            index_name=config.get("index_name", Config.PINECONE_INDEX_NAME),
            dim=dim,
            api_key=Config.PINECONE_API_KEY,
        )
    raise ValueError(f"Unknown vector store profile: {store_type}")
