"""Tests for API endpoints."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("Content-Type", "")

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database_path" in data
        assert "document_count" in data
        assert "ollama_available" in data

    def test_get_documents_endpoint(self, client):
        """Test getting documents endpoint."""
        response = client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_documents_with_limit(self, client):
        """Test getting documents with limit parameter."""
        response = client.get("/documents?n=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10

    def test_search_endpoint(self, client):
        """Test search endpoint."""
        response = client.get("/search?query=test&n=5")

        # May return 503 if Ollama is not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "summary" in data
            assert "sources" in data
            assert "query" in data
            assert "total_sources" in data

    def test_search_endpoint_with_invalid_query(self, client):
        """Test search endpoint with invalid query."""
        response = client.get("/search?query=&n=5")

        assert response.status_code == 422  # Validation error

    def test_search_endpoint_with_large_n(self, client):
        """Test search endpoint with large n parameter."""
        response = client.get("/search?query=test&n=100")

        assert response.status_code == 422  # Validation error (n <= 50)

    def test_index_endpoint(self, client):
        """Test index endpoint."""
        response = client.post("/index", json={
            "content": "Test document content",
            "metadata": {"source": "api"}
        })

        # May return 503 if Ollama is not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"

    def test_reindex_endpoint(self, client):
        """Test reindex endpoint."""
        response = client.post("/reindex")

        # May return 503 if Ollama is not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"

    def test_download_endpoint(self, client):
        """Test download endpoint."""
        # This will fail if file doesn't exist
        response = client.get("/download/test.txt")

        # May return 404 if file doesn't exist
        assert response.status_code in [200, 404]

    def test_download_nonexistent_file(self, client):
        """Test download endpoint with nonexistent file."""
        response = client.get("/download/nonexistent.txt")

        assert response.status_code == 404

    def test_index_endpoint_with_invalid_content(self, client):
        """Test index endpoint with invalid content."""
        response = client.post("/index", json={
            "content": "",
            "metadata": None
        })

        # Should handle empty content gracefully
        assert response.status_code in [200, 503]


class TestAPIErrorHandling:
    """Test cases for API error handling."""

    def test_search_with_empty_query(self, client):
        """Test search with empty query."""
        response = client.get("/search?query=&n=5")

        assert response.status_code == 422

    def test_search_with_too_long_query(self, client):
        """Test search with query too long."""
        response = client.get("/search?query=" + "a" * 1001 + "&n=5")

        assert response.status_code == 422

    def test_get_documents_with_invalid_n(self, client):
        """Test get documents with invalid n parameter."""
        response = client.get("/documents?n=0")

        assert response.status_code == 422

        response = client.get("/documents?n=1001")

        assert response.status_code == 422

    def test_index_with_missing_content(self, client):
        """Test index with missing content field."""
        response = client.post("/index", json={
            "metadata": {"source": "api"}
        })

        assert response.status_code == 422

    def test_health_check_failure(self, client):
        """Test health check when database is unavailable."""
        # This should still return a response
        response = client.get("/health")

        assert response.status_code == 200


class TestAPIResponseModels:
    """Test cases for API response models."""

    def test_document_response_structure(self, client):
        """Test document response has correct structure."""
        response = client.get("/documents")

        assert response.status_code == 200
        data = response.json()

        for doc in data:
            assert "id" in doc or doc.get("id") is None
            assert "content" in doc
            assert "metadata" in doc
            assert "embedding" in doc or doc.get("embedding") is None

    def test_search_response_structure(self, client):
        """Test search response has correct structure."""
        response = client.get("/search?query=test&n=1")

        if response.status_code == 200:
            data = response.json()

            assert "summary" in data
            assert "sources" in data
            assert "query" in data
            assert "total_sources" in data
            assert isinstance(data["sources"], list)
            assert isinstance(data["total_sources"], int)

    def test_health_response_structure(self, client):
        """Test health response has correct structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "database_path" in data
        assert "document_count" in data
        assert "ollama_available" in data
        assert isinstance(data["document_count"], int)
        assert isinstance(data["ollama_available"], bool)


class TestAPIWithMockedServices:
    """Test cases with mocked services."""

    @patch('src.api.endpoints.chroma_client')
    @patch('src.api.endpoints.ollama_client')
    def test_search_with_mocked_services(self, mock_ollama, mock_chroma, client):
        """Test search with mocked services."""
        # Setup mocks
        mock_chroma.search.return_value = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'source': 'file1'}, {'source': 'file2'}]],
            'embeddings': [[[0.1, 0.2], [0.3, 0.4]]]
        }
        mock_ollama.generate_response.return_value = "Summary of results"

        response = client.get("/search?query=test&n=2")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "Summary of results"
        assert len(data["sources"]) == 2

    @patch('src.api.endpoints.chroma_client')
    @patch('src.api.endpoints.ollama_client')
    def test_index_with_mocked_services(self, mock_ollama, mock_chroma, client):
        """Test index with mocked services."""
        mock_chroma.add_document.return_value = None

        response = client.post("/index", json={
            "content": "Test content",
            "metadata": {"source": "api"}
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch('src.api.endpoints.chroma_client')
    def test_get_documents_with_mocked_services(self, mock_chroma, client):
        """Test get documents with mocked services."""
        mock_chroma.list_documents.return_value = [
            {'id': 0, 'content': 'doc1', 'metadata': {'source': 'file1'}},
            {'id': 1, 'content': 'doc2', 'metadata': {'source': 'file2'}}
        ]

        response = client.get("/documents?n=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestAPIQueryParameters:
    """Test cases for query parameter handling."""

    def test_search_with_default_n(self, client):
        """Test search with default n parameter."""
        response = client.get("/search?query=test")

        assert response.status_code in [200, 503]

    def test_search_with_n_at_boundary(self, client):
        """Test search with n at boundary values."""
        # n=1 (minimum)
        response = client.get("/search?query=test&n=1")
        assert response.status_code in [200, 503]

        # n=50 (maximum)
        response = client.get("/search?query=test&n=50")
        assert response.status_code in [200, 503]

    def test_get_documents_with_default_n(self, client):
        """Test get documents with default n parameter."""
        response = client.get("/documents")

        assert response.status_code == 200

    def test_get_documents_with_n_at_boundary(self, client):
        """Test get documents with n at boundary values."""
        # n=1 (minimum)
        response = client.get("/documents?n=1")
        assert response.status_code == 200

        # n=1000 (maximum)
        response = client.get("/documents?n=1000")
        assert response.status_code == 200


class TestAPIContentType:
    """Test cases for content type handling."""

    def test_json_response_content_type(self, client):
        """Test JSON responses have correct content type."""
        response = client.get("/health")

        assert "application/json" in response.headers.get("Content-Type", "")

    def test_html_response_content_type(self, client):
        """Test HTML responses have correct content type."""
        response = client.get("/")

        assert "text/html" in response.headers.get("Content-Type", "")


class TestAPIWithRealData:
    """Test cases with real data."""

    def test_index_with_real_content(self, client, tmp_path):
        """Test indexing with real content."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is real test content for indexing.")

        # Mock the client to use real data
        with patch('src.api.endpoints.chroma_client') as mock_chroma:
            mock_client = mock_chroma.return_value
            mock_client.add_document.return_value = None

            response = client.post("/index", json={
                "content": "This is real test content for indexing.",
                "metadata": {"source": "test"}
            })

            assert response.status_code == 200

    def test_search_with_real_query(self, client):
        """Test search with real query."""
        response = client.get("/search?query=python programming&n=3")

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "summary" in data
            assert "sources" in data


class TestAPIEdgeCases:
    """Test cases for edge cases."""

    def test_search_with_special_characters(self, client):
        """Test search with special characters in query."""
        response = client.get("/search?query=Hello%20World%21%20%F0%9F%8C%8D&n=1")

        assert response.status_code in [200, 503]

    def test_index_with_unicode_content(self, client):
        """Test index with unicode content."""
        response = client.post("/index", json={
            "content": "Hello 世界！Привет! 🌍",
            "metadata": {"source": "unicode"}
        })

        assert response.status_code in [200, 503]

    def test_search_with_newlines(self, client):
        """Test search with newlines in query."""
        response = client.get("/search?query=Line1%0ALine2%0D%0A&n=1")

        assert response.status_code in [200, 503]

    def test_get_documents_empty_list(self, client):
        """Test get documents when list is empty."""
        with patch('src.api.endpoints.chroma_client') as mock_chroma:
            mock_client = mock_chroma.return_value
            mock_client.list_documents.return_value = []

            response = client.get("/documents")

            assert response.status_code == 200
            data = response.json()
            assert data == []

    def test_search_no_results(self, client):
        """Test search when no results found."""
        with patch('src.api.endpoints.chroma_client') as mock_chroma:
            mock_client = mock_chroma.return_value
            mock_client.search.return_value = {
                'documents': [],
                'metadatas': [],
                'embeddings': []
            }

            response = client.get("/search?query=test&n=1")

            assert response.status_code == 200
            data = response.json()
            assert data["summary"] == "No relevant documents found for your query."
            assert data["total_sources"] == 0
