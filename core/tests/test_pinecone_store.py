from unittest.mock import MagicMock, patch

from core.src.persistence.factory import create_vector_store
from core.src.persistence.pinecone_store import PineconeStore


def _client(existing=()):
    client = MagicMock()
    client.list_indexes.return_value.names.return_value = list(existing)
    return client


def test_client_takes_only_the_api_key():
    # pinecone>=3 rejects the old pod-era environment= argument
    with patch("core.src.persistence.pinecone_store.Pinecone", return_value=_client(["idx"])) as cls:
        PineconeStore(index_name="idx", dim=384, api_key="key")
    cls.assert_called_once_with(api_key="key")


def test_missing_index_is_created_serverless():
    client = _client()
    with patch("core.src.persistence.pinecone_store.Pinecone", return_value=client), \
         patch("core.src.persistence.pinecone_store.ServerlessSpec") as spec:
        PineconeStore(index_name="idx", dim=384, api_key="key")
    kwargs = client.create_index.call_args.kwargs
    assert (kwargs["name"], kwargs["dimension"], kwargs["metric"]) == ("idx", 384, "cosine")
    spec.assert_called_once_with(cloud="aws", region="us-east-1")


def test_factory_builds_the_store_from_config():
    with patch("core.src.persistence.pinecone_store.Pinecone", return_value=_client(["custom"])) as cls, \
         patch("core.src.persistence.factory.Config") as cfg:
        cfg.PINECONE_API_KEY = "key"
        store = create_vector_store("pinecone", 384, {"index_name": "custom"})
    cls.assert_called_once_with(api_key="key")
    assert isinstance(store, PineconeStore)
