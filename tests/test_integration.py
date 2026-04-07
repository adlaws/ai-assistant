"""Integration tests for the full indexing workflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.indexer.chroma_client import ChromaClient, create_client
from src.indexer.api_client import OllamaClient, create_client as create_ollama_client
from src.indexer.file_processing import get_handler
from src.indexer.indexing import index_file, index_documents
from src.indexer.chunking import chunk_text
from src.indexer.utils import compute_file_hash, load_cache, save_cache


class TestFullIndexingWorkflow:
    """Integration tests for the complete indexing workflow."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create a temporary data directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "chroma_db"

    @pytest.fixture
    def temp_cache_file(self, tmp_path):
        """Create a temporary cache file."""
        cache_file = tmp_path / "document_cache.json"
        return cache_file

    def test_index_single_text_file(self, tmp_path, mocker):
        """Test indexing a single text file."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a text file
        text_file = data_dir / "test.txt"
        text_file.write_text("This is test content for indexing.")

        # Mock the embedding function
        mocker.patch('src.indexer.indexing.get_ollama_embedding', return_value=[0.1, 0.2, 0.3])

        content, embedding = index_file(str(text_file))

        assert content == "This is test content for indexing."
        assert embedding == [0.1, 0.2, 0.3]

    def test_index_multiple_files(self, tmp_path, mocker):
        """Test indexing multiple files."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create multiple text files
        for i in range(5):
            text_file = data_dir / f"test{i}.txt"
            text_file.write_text(f"Content {i}")

        # Mock the client and embedding function
        mock_collection = MagicMock()
        mock_collection.add = MagicMock()
        mock_collection.count = MagicMock(return_value=5)
        mock_collection.get = MagicMock(return_value={'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []})

        mock_client = MagicMock()
        mock_client.collection = mock_collection
        mock_client.delete_collection = MagicMock()

        # Mock the embedding function
        mocker.patch('src.indexer.indexing.get_ollama_embedding', return_value=[0.1, 0.2, 0.3])

        count = index_documents(mock_client, str(data_dir))

        assert count == 5

    def test_index_with_chunking(self, tmp_path):
        """Test indexing with text chunking."""
        # Create a large text file
        large_text = "This is a test sentence. " * 100
        text_file = tmp_path / "large.txt"
        text_file.write_text(large_text)

        # Chunk the text
        chunks = chunk_text(large_text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1

    def test_cache_functionality(self, tmp_path, mocker):
        """Test cache functionality during indexing."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a text file
        text_file = data_dir / "test.txt"
        text_file.write_text("Test content")

        # Mock the client
        mock_collection = MagicMock()
        mock_collection.add = MagicMock()
        mock_collection.count = MagicMock(return_value=1)
        mock_collection.get = MagicMock(return_value={'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []})

        mock_client = MagicMock()
        mock_client.collection = mock_collection
        mock_client.delete_collection = MagicMock()

        # Mock embedding function
        mocker.patch('src.indexer.indexing.get_ollama_embedding', return_value=[0.1, 0.2, 0.3])

        # First indexing
        count1 = index_documents(mock_client, str(data_dir))
        assert count1 == 1

    def test_file_hash_computation(self, tmp_path):
        """Test file hash computation for cache."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("Test content")

        # Compute hash
        hash1 = compute_file_hash(str(text_file))

        # Verify hash is consistent
        hash2 = compute_file_hash(str(text_file))

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_handler_selection(self, tmp_path):
        """Test automatic handler selection based on file extension."""
        # Create files with different extensions
        extensions = ['.txt', '.md', '.json', '.pdf', '.docx', '.png']

        for ext in extensions:
            file_path = tmp_path / f"test{ext}"
            file_path.write_text("test")

            handler = get_handler(str(file_path))
            assert handler is not None

    def test_unsupported_file_extension(self, tmp_path):
        """Test handling of unsupported file extensions."""
        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test")

        with pytest.raises(Exception):
            get_handler(str(unsupported_file))

    def test_index_with_metadata(self, tmp_path, mocker):
        """Test indexing with metadata."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a text file
        text_file = data_dir / "test.txt"
        text_file.write_text("Test content")

        # Mock the embedding function
        mocker.patch('src.indexer.indexing.get_ollama_embedding', return_value=[0.1, 0.2, 0.3])

        # Index with metadata
        content, embedding = index_file(str(text_file))

        assert content == "Test content"
        assert embedding == [0.1, 0.2, 0.3]


class TestOllamaClientIntegration:
    """Integration tests for Ollama API client."""

    def test_call_api_success(self, mocker):
        """Test successful API call."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.text = '{"response": "test"}'

        mock_post = mocker.patch('src.indexer.api_client.requests.post')
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.call_api("test", param="value")

        assert result == {"response": "test"}

    def test_call_api_http_error(self, mocker):
        """Test API call with HTTP error."""
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_post = mocker.patch('src.indexer.api_client.requests.post')
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(Exception):
            client.call_api("test")

    def test_get_embedding(self, mocker):
        """Test getting an embedding."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.text = '{"embedding": [0.1, 0.2, 0.3]}'

        mock_post = mocker.patch('src.indexer.api_client.requests.post')
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.get_embedding("test text")

        assert result == [0.1, 0.2, 0.3]

    def test_generate_response(self, mocker):
        """Test generating a response."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.text = '{"response": "Generated response"}'

        mock_post = mocker.patch('src.indexer.api_client.requests.post')
        mock_post.return_value = mock_response

        client = OllamaClient()
        result = client.generate_response("test prompt")

        assert result == "Generated response"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def setup_test_environment(self, tmp_path):
        """Set up a test environment with data and database."""
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create database directory
        db_path = tmp_path / "chroma_db"
        db_path.mkdir()

        # Create cache file
        cache_file = tmp_path / "document_cache.json"

        return {
            'data_dir': data_dir,
            'db_path': db_path,
            'cache_file': cache_file
        }

    def test_complete_indexing_workflow(self, setup_test_environment, mocker):
        """Test complete indexing workflow from start to finish."""
        env = setup_test_environment

        # Create test files
        for i in range(3):
            text_file = env['data_dir'] / f"test{i}.txt"
            text_file.write_text(f"Content {i}")

        # Mock the client
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)
        mock_client.delete_collection = mocker.Mock()

        # Mock embedding function
        mock_embedding = mocker.Mock()
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        # Mock get_handler
        mock_handler = mocker.Mock()
        mock_handler.load_document.return_value = "test content"

        # Mock index_file
        mocker.patch('src.indexer.indexing.get_handler', return_value=mock_handler)
        mocker.patch('src.indexer.indexing.get_ollama_embedding', return_value=mock_embedding)

        # Run indexing
        count = index_documents(mock_client, str(env['data_dir']))

        assert count == 3
        assert mock_client.collection.add.call_count == 3

    def test_cache_persistence(self, setup_test_environment, mocker):
        """Test that cache persists across indexing runs."""
        env = setup_test_environment

        # Create test file
        text_file = env['data_dir'] / "test.txt"
        text_file.write_text("Test content")

        # Mock the client
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)

        # Mock embedding function
        mock_embedding = mocker.Mock()
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        # First indexing
        count1 = index_documents(mock_client, str(env['data_dir']))
        assert count1 == 1

        # Load cache
        cache = load_cache()
        assert str(text_file) in cache

        # Second indexing should skip cached file
        count2 = index_documents(mock_client, str(env['data_dir']))
        assert count2 == 0
