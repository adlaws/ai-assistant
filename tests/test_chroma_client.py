"""Tests for ChromaDB client."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.indexer.chroma_client import ChromaClient, create_client
from src.indexer.exceptions import ChromaDBError, OllamaError


class TestChromaClient:
    """Test cases for ChromaClient class."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Create a mock configuration."""
        mock_config = mocker.patch('src.indexer.chroma_client.IndexerConfig')
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance
        return mock_config

    @pytest.fixture
    def mock_ollama_client(self, mocker):
        """Create a mock Ollama client."""
        mock_ollama = Mock()
        mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_ollama

    def test_init(self, mock_config, mock_ollama_client, mocker):
        """Test client initialization."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mocker.patch('chromadb.PersistentClient')

        client = ChromaClient("test_collection", db_path=None)

        assert client.collection_name == "test_collection"
        assert client.db_path is None

    def test_init_with_db_path(self, mock_config, mock_ollama_client, mocker):
        """Test client initialization with database path."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.PersistentClient')

        client = ChromaClient("test_collection", db_path="/tmp/db")

        assert client.db_path == "/tmp/db"

    def test_add_document(self, mock_config, mock_ollama_client, mocker):
        """Test adding a document."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        client.add_document("test content", {"key": "value"})

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]['documents'] == ["test content"]
        assert call_args[1]['embeddings'] == [[0.1, 0.2, 0.3]]
        assert call_args[1]['metadatas'] == [{"key": "value"}]

    def test_add_document_no_embedding_fn(self, mock_config, mocker):
        """Test document addition without embedding function."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=None)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        client.embedding_fn = None  # Force no embedding function

        with pytest.raises(ChromaDBError, match="No embedding function available"):
            client.add_document("test content")

    def test_search_success(self, mock_config, mock_ollama_client, mocker):
        """Test successful search."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.query.return_value = {"results": "test"}
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        result = client.search("test query")

        assert result == {"results": "test"}
        mock_collection.query.assert_called_once()

    def test_count_success(self, mock_config, mock_ollama_client, mocker):
        """Test successful count retrieval."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        result = client.count()

        assert result == 42

    def test_get_ids_success(self, mock_config, mock_ollama_client, mocker):
        """Test successful ID retrieval."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        result = client.get_ids()

        assert result == {"id1", "id2", "id3"}

    def test_create_client(self, mocker):
        """Test client factory function."""
        mock_client_class = mocker.patch('src.indexer.chroma_client.ChromaClient')
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        result = create_client("/tmp/db")

        assert result == mock_client_instance
        mock_client_class.assert_called_once_with(
            collection_name="documents",
            db_path="/tmp/db"
        )


class TestChromaClientListDocuments:
    """Test cases for list_documents method."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Create a mock configuration."""
        mock_config = mocker.patch('src.indexer.chroma_client.IndexerConfig')
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance
        return mock_config

    @pytest.fixture
    def mock_ollama_client(self, mocker):
        """Create a mock Ollama client."""
        mock_ollama = Mock()
        mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_ollama

    def test_list_documents(self, mock_config, mock_ollama_client, mocker):
        """Test listing documents."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'documents': ['doc1', 'doc2'],
            'metadatas': [{'source': 'file1'}, {'source': 'file2'}],
            'embeddings': [[0.1], [0.2]]
        }
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        documents = client.list_documents()

        assert len(documents) == 2
        assert documents[0]['content'] == 'doc1'
        assert documents[0]['metadata'] == {'source': 'file1'}

    def test_list_documents_empty(self, mock_config, mock_ollama_client, mocker):
        """Test listing empty documents."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'documents': [],
            'metadatas': [],
            'embeddings': []
        }
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        documents = client.list_documents()

        assert documents == []

    def test_list_documents_no_embeddings(self, mock_config, mock_ollama_client, mocker):
        """Test listing documents without embeddings."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'documents': ['doc1', 'doc2'],
            'metadatas': [{'source': 'file1'}, {'source': 'file2'}],
            'embeddings': None
        }
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        documents = client.list_documents()

        assert len(documents) == 2
        assert documents[0]['embedding'] is None


class TestChromaClientDeleteCollection:
    """Test cases for delete_collection method."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Create a mock configuration."""
        mock_config = mocker.patch('src.indexer.chroma_client.IndexerConfig')
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance
        return mock_config

    @pytest.fixture
    def mock_ollama_client(self, mocker):
        """Create a mock Ollama client."""
        mock_ollama = Mock()
        mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_ollama

    def test_delete_collection(self, mock_config, mock_ollama_client, mocker):
        """Test deleting collection."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mock_chroma_client = Mock()
        mocker.patch('chromadb.PersistentClient').return_value = mock_chroma_client
        mock_collection = Mock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        client.delete_collection()

        mock_chroma_client.delete_collection.assert_called_once_with("test_collection")
        assert client.collection is None

    def test_delete_collection_error(self, mock_config, mock_ollama_client, mocker):
        """Test deleting collection with error."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.client.delete_collection.side_effect = Exception("Delete failed")
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        client.delete_collection()

        assert client.collection is None


class TestChromaClientAddDocuments:
    """Test cases for add_documents method."""

    @pytest.fixture
    def mock_config(self, mocker):
        """Create a mock configuration."""
        mock_config = mocker.patch('src.indexer.chroma_client.IndexerConfig')
        mock_config_instance = Mock()
        mock_config_instance.ollama_base_url = "http://localhost:11434"
        mock_config_instance.embedding_model = "nomic-embed-text"
        mock_config.return_value = mock_config_instance
        return mock_config

    @pytest.fixture
    def mock_ollama_client(self, mocker):
        """Create a mock Ollama client."""
        mock_ollama = Mock()
        mock_ollama.get_embedding.return_value = [0.1, 0.2, 0.3]
        return mock_ollama

    def test_add_documents(self, mock_config, mock_ollama_client, mocker):
        """Test adding multiple documents."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        documents = ["doc1", "doc2", "doc3"]
        ids = ["id1", "id2", "id3"]

        client.add_documents(documents, ids)

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]['documents'] == documents
        assert call_args[1]['ids'] == ids

    def test_add_documents_with_embeddings(self, mock_config, mock_ollama_client, mocker):
        """Test adding documents with embeddings."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=mock_ollama_client)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        documents = ["doc1", "doc2"]
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        client.add_documents(documents, ids, embeddings=embeddings)

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]['embeddings'] == embeddings

    def test_add_documents_no_embedding_fn(self, mock_config, mocker):
        """Test adding documents without embedding function."""
        mocker.patch('src.indexer.chroma_client.create_ollama_client', return_value=None)
        mocker.patch('chromadb.Client')
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mocker.patch('chromadb.PersistentClient').return_value.get_or_create_collection.return_value = mock_collection

        client = ChromaClient("test_collection")
        client.embedding_fn = None

        documents = ["doc1", "doc2"]
        ids = ["id1", "id2"]

        with pytest.raises(ChromaDBError, match="No embedding function available and no embeddings provided"):
            client.add_documents(documents, ids)
