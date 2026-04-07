"""Pytest configuration and fixtures for tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


# Create mocks for endpoint tests
mock_chroma = MagicMock()
mock_chroma.collection_name = "documents"

mock_collection = MagicMock()
mock_collection.count.return_value = 1
mock_collection.get.return_value = {
    'documents': ['Test document 1'],
    'metadatas': [{'source': 'test', 'type': 'txt', 'filepath': '/test/test.txt'}],
    'embeddings': [[0.1, 0.2, 0.3]],
    'ids': ['test_id_1']
}
mock_collection.query.return_value = {
    "documents": [[
        "This is a test document about artificial intelligence.",
        "Another relevant document"
    ]],
    "metadatas": [[
        {"source": "test.txt", "filepath": "/test/test.txt", "type": "txt"},
        {"source": "other.txt", "filepath": "/test/other.txt", "type": "txt"}
    ]],
    "distances": [[0.1, 0.3]],
    "embeddings": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
}

mock_client = MagicMock()
mock_client.get_or_create_collection.return_value = mock_collection
mock_client.delete_collection.return_value = None

mock_chroma.client = mock_client
mock_chroma.collection = mock_collection
mock_chroma.list_documents.return_value = [
    {
        "id": 1,
        "content": "Test document 1",
        "metadata": {"source": "test", "type": "txt"},
        "embedding": [0.1, 0.2, 0.3]
    }
]
mock_chroma.search.return_value = {
    "documents": [[
        "This is a test document about artificial intelligence.",
        "Another relevant document"
    ]],
    "metadatas": [[
        {"source": "test.txt", "filepath": "/test/test.txt", "type": "txt"},
        {"source": "other.txt", "filepath": "/test/other.txt", "type": "txt"}
    ]],
    "distances": [[0.1, 0.3]],
    "embeddings": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
}
mock_chroma.count.return_value = 1
mock_chroma.delete_collection.return_value = None
mock_chroma.add_document.return_value = None
mock_chroma.add_documents.return_value = None

mock_ollama = MagicMock()
mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
mock_ollama.generate_response.return_value = "This is a summary of the search results."


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked ChromaDB and Ollama clients."""
    from fastapi.testclient import TestClient
    import src.api.endpoints

    # Patch the clients at the module level
    monkeypatch.setattr('src.api.endpoints.chroma_client', mock_chroma)
    monkeypatch.setattr('src.api.endpoints.ollama_client', mock_ollama)
    monkeypatch.setattr('src.indexer.indexing.index_documents', MagicMock(return_value=0))

    return TestClient(src.api.endpoints.app)
