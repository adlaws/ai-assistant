"""Tests for Ollama API client."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from src.indexer.api_client import OllamaClient, create_client
from src.indexer.exceptions import OllamaError, EmbeddingError


class TestOllamaClient:
    """Test cases for OllamaClient."""

    def test_init(self):
        """Test client initialization."""
        client = OllamaClient(base_url="http://test:11434", timeout=30)
        assert client.base_url == "http://test:11434"
        assert client.timeout == 30

    @patch('src.indexer.api_client.requests.post')
    def test_call_api_success(self, mock_post):
        """Test successful API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"response": "test"}'
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.call_api("test", param="value")

        assert result == {"response": "test"}
        mock_post.assert_called_once()

    @patch('src.indexer.api_client.requests.post')
    def test_call_api_http_error(self, mock_post):
        """Test API call with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(OllamaError, match="HTTP 500"):
            client.call_api("test")

    @patch('src.indexer.api_client.requests.post')
    def test_call_api_connection_error(self, mock_post):
        """Test API call with connection error."""
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError("Connection failed")

        client = OllamaClient()

        with pytest.raises(OllamaError, match="Cannot connect to Ollama"):
            client.call_api("test")

    @patch('src.indexer.api_client.requests.post')
    def test_get_embedding_success(self, mock_post):
        """Test successful embedding generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"embedding": [0.1, 0.2, 0.3]}'
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.get_embedding("test text")

        assert result == [0.1, 0.2, 0.3]

    def test_get_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        client = OllamaClient()

        with pytest.raises(EmbeddingError, match="Cannot generate embedding for empty text"):
            client.get_embedding("")

    @patch('src.indexer.api_client.requests.post')
    def test_get_embedding_api_failure(self, mock_post):
        """Test embedding generation with API failure."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"embedding": []}'
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(EmbeddingError, match="Ollama returned empty embedding"):
            client.get_embedding("test")

    @patch('src.indexer.api_client.requests.post')
    def test_generate_response_success(self, mock_post):
        """Test successful response generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"response": "Test response"}'
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.generate_response("test prompt")

        assert result == "Test response"

    def test_create_client(self):
        """Test client factory function."""
        client = create_client()
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://localhost:11434"