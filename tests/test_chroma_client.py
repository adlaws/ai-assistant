"""Tests for ChromaDB client."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.indexer.chroma_client import ChromaClient, create_client
from src.indexer.exceptions import ChromaDBError, OllamaError


class TestChromaClient:
    """Test cases for ChromaClient."""

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.create_ollama_client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_init_success(self, mock_config, mock_create_client, mock_chroma_client):
        """Test successful client initialization."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        mock_ollama = Mock()
        mock_create_client.return_value = mock_ollama

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None
        mock_chroma_client.return_value = mock_client

        client = ChromaClient("test_collection")

        assert client.collection_name == "test_collection"
        assert client.collection == mock_collection

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_init_chroma_failure(self, mock_config, mock_chroma_client):
        """Test initialization failure."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        mock_chroma_client.side_effect = Exception("ChromaDB error")

        with pytest.raises(ChromaDBError):
            ChromaClient("test_collection")

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.create_ollama_client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_add_document_success(self, mock_config, mock_create_client, mock_chroma_client):
        """Test successful document addition."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        # Setup mocks
        mock_ollama = Mock()
        mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_create_client.return_value = mock_ollama

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None
        mock_chroma_client.return_value = mock_client

        client = ChromaClient("test_collection")
        client.add_document("test content", {"key": "value"})

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]['documents'] == ["test content"]
        assert call_args[1]['embeddings'] == [[0.1, 0.2, 0.3]]
        assert call_args[1]['metadatas'] == [{"key": "value"}]

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.create_ollama_client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_add_document_no_embedding_fn(self, mock_config, mock_create_client, mock_chroma_client):
        """Test document addition without embedding function."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        mock_create_client.return_value = None  # Simulate Ollama failure

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        client = ChromaClient("test_collection")
        client.embedding_fn = None  # Force no embedding function

        with pytest.raises(ChromaDBError, match="No embedding function available"):
            client.add_document("test content")

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.create_ollama_client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_search_success(self, mock_config, mock_create_client, mock_chroma_client):
        """Test successful search."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        # Setup mocks
        mock_ollama = Mock()
        mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_create_client.return_value = mock_ollama

        mock_collection = Mock()
        mock_collection.query.return_value = {"results": "test"}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        client = ChromaClient("test_collection")
        result = client.search("test query")

        assert result == {"results": "test"}
        mock_collection.query.assert_called_once()

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.create_ollama_client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_count_success(self, mock_config, mock_create_client, mock_chroma_client):
        """Test successful count retrieval."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None
        mock_chroma_client.return_value = mock_client

        client = ChromaClient("test_collection")
        result = client.count()

        assert result == 42

    @patch('src.indexer.chroma_client.chromadb.Client')
    @patch('src.indexer.chroma_client.create_ollama_client')
    @patch('src.indexer.chroma_client.IndexerConfig')
    def test_get_ids_success(self, mock_config, mock_create_client, mock_chroma_client):
        """Test successful ID retrieval."""
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance

        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.return_value = None
        mock_chroma_client.return_value = mock_client

        client = ChromaClient("test_collection")
        result = client.get_ids()

        assert result == {"id1", "id2", "id3"}

    def test_create_client(self):
        """Test client factory function."""
        with patch('src.indexer.chroma_client.ChromaClient') as mock_client_class:
            mock_instance = Mock()
            mock_client_class.return_value = mock_instance

            result = create_client("/tmp/db")

            assert result == mock_instance
            mock_client_class.assert_called_once_with(
                collection_name="documents",
                db_path="/tmp/db"
            )