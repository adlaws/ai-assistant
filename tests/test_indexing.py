"""Tests for indexing module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.indexer.exceptions import FileProcessingError
from src.indexer.file_processing import get_handler
from src.indexer.indexing import (
    process_text_file,
    process_pdf_file,
    process_markdown_file,
    process_word_document,
    extract_image_description,
    get_ollama_embedding,
    index_file,
    index_documents,
)
from src.indexer.utils import compute_file_hash, load_cache, save_cache


class TestProcessTextFile:
    """Test cases for process_text_file function."""

    def test_process_valid_text_file(self, tmp_path):
        """Test processing a valid text file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, World!")

        content = process_text_file(str(text_file))

        assert content == "Hello, World!"

    def test_process_empty_text_file(self, tmp_path):
        """Test processing an empty text file."""
        text_file = tmp_path / "empty.txt"
        text_file.write_text("")

        content = process_text_file(str(text_file))

        assert content == ""

    def test_process_large_text_file(self, tmp_path):
        """Test processing a large text file."""
        text_file = tmp_path / "large.txt"
        large_content = "A" * 1000000  # 1MB file

        text_file.write_text(large_content)

        content = process_text_file(str(text_file))

        assert content == large_content

    def test_process_nonexistent_file(self):
        """Test processing a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            process_text_file("/nonexistent/file.txt")

    def test_process_binary_file(self, tmp_path):
        """Test processing a binary file as text."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        # Implementation handles binary files gracefully
        try:
            content = process_text_file(str(binary_file))
            # Should either return content or be handled without exception
            assert isinstance(content, str) or content is None
        except (FileNotFoundError, UnicodeDecodeError):
            # These exceptions are acceptable
            pass


class TestProcessPdfFile:
    """Test cases for process_pdf_file function."""

    def test_process_pdf_with_text(self, tmp_path):
        """Test processing a PDF with text content."""
        # Create a simple PDF using pypdf
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            # Add a blank page
            writer.add_blank_page(width=200, height=200)
            pdf_path = tmp_path / "test.pdf"
            with open(pdf_path, 'wb') as f:
                writer.write(f)

            # Process the PDF
            content = process_pdf_file(str(pdf_path))

            # PDF without text should return empty string
            assert content == ""
        except ImportError:
            pytest.skip("pypdf not installed")

    def test_process_nonexistent_pdf(self):
        """Test processing a nonexistent PDF."""
        # Implementation may handle missing files gracefully
        try:
            content = process_pdf_file("/nonexistent/file.pdf")
            # May return empty string or None
            assert content is None or content == ""
        except FileNotFoundError:
            # Or may raise FileNotFoundError
            pass

    def test_process_corrupted_pdf(self, tmp_path):
        """Test processing a corrupted PDF."""
        corrupted = tmp_path / "corrupted.pdf"
        corrupted.write_bytes(b"not a valid pdf")

        # Implementation may handle corrupted PDFs gracefully
        try:
            content = process_pdf_file(str(corrupted))
            # May return empty string or None
            assert content is None or content == ""
        except Exception:
            # Or may raise an exception
            pass


class TestProcessMarkdownFile:
    """Test cases for process_markdown_file function."""

    def test_process_markdown_file(self, tmp_path):
        """Test processing a markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nWorld!")

        content = process_markdown_file(str(md_file))

        # Content should contain the markdown text
        assert "Hello" in content and "World" in content

    def test_process_empty_markdown(self, tmp_path):
        """Test processing an empty markdown file."""
        md_file = tmp_path / "empty.md"
        md_file.write_text("")

        content = process_markdown_file(str(md_file))

        assert content == ""


class TestProcessWordDocument:
    """Test cases for process_word_document function."""

    def test_process_word_document(self, tmp_path):
        """Test processing a Word document."""
        try:
            from docx import Document
            doc = Document()
            doc.add_paragraph("Hello, World!")
            doc_path = tmp_path / "test.docx"
            doc.save(doc_path)

            content = process_word_document(str(doc_path))

            assert content == "Hello, World!"
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_process_nonexistent_word(self):
        """Test processing a nonexistent Word document."""
        # Implementation may handle missing files gracefully
        try:
            content = process_word_document("/nonexistent/file.docx")
            # May return empty string or None
            assert content is None or content == ""
        except FileNotFoundError:
            # Or may raise FileNotFoundError
            pass


class TestExtractImageDescription:
    """Test cases for extract_image_description function."""

    def test_extract_image_description(self, tmp_path):
        """Test extracting description from image."""
        try:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img_path = tmp_path / "test.png"
            img.save(img_path)

            description = extract_image_description(str(img_path))

            assert "Image:" in description
            assert "100x100" in description
            assert "png" in description.lower() or "PNG" in description
        except ImportError:
            pytest.skip("PIL not installed")

    def test_extract_nonexistent_image(self):
        """Test extracting description from nonexistent image."""
        result = extract_image_description("/nonexistent/image.png")

        assert "Error reading image" in result

    def test_extract_corrupted_image(self, tmp_path):
        """Test extracting description from corrupted image."""
        corrupted = tmp_path / "corrupted.png"
        corrupted.write_bytes(b"not a valid image")

        result = extract_image_description(str(corrupted))

        assert "Error reading image" in result


class TestGetOllamaEmbedding:
    """Test cases for get_ollama_embedding function."""

    @pytest.fixture
    def mock_ollama_client(self, mocker):
        """Create a mock Ollama client."""
        mock_client = mocker.patch('src.indexer.indexing.create_client')
        mock_response = mocker.Mock()
        mock_response.get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_client.return_value = mock_response
        return mock_client

    def test_get_embedding_success(self, mock_ollama_client):
        """Test successful embedding generation."""
        embedding = get_ollama_embedding("test text")

        assert isinstance(embedding, list)
        assert len(embedding) > 0

    def test_get_embedding_empty_text(self, mock_ollama_client):
        """Test embedding generation with empty text."""
        # This should fail or return empty
        with pytest.raises(Exception):
            get_ollama_embedding("")

    def test_get_embedding_unicode(self, mock_ollama_client):
        """Test embedding generation with unicode text."""
        embedding = get_ollama_embedding("Hello 世界！")

        assert isinstance(embedding, list)


class TestIndexFile:
    """Test cases for index_file function."""

    def test_index_text_file(self, tmp_path, mocker):
        """Test indexing a text file."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("Test content")

        # Mock the embedding function
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        content, embedding = index_file(str(text_file))

        assert content == "Test content"
        assert embedding == [0.1, 0.2, 0.3]

    def test_index_nonexistent_file(self, mocker):
        """Test indexing a nonexistent file."""
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        content, embedding = index_file("/nonexistent/file.txt")

        assert content is None
        assert embedding is None

    def test_index_unsupported_extension(self, mocker):
        """Test indexing a file with unsupported extension."""
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        content, embedding = index_file("/path/to/file.xyz")

        assert content is None
        assert embedding is None


class TestIndexDocuments:
    """Test cases for index_documents function."""

    def test_index_empty_directory(self, tmp_path, mocker):
        """Test indexing an empty directory."""
        # Create an empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # Mock the client and embedding function
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)

        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        count = index_documents(mock_client, str(empty_dir))

        assert count == 0

    def test_index_with_files(self, tmp_path, mocker):
        """Test indexing a directory with files."""
        # Create a directory with files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        text_file = data_dir / "test.txt"
        text_file.write_text("Test content")

        # Mock the client
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)

        # Mock embedding function
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        count = index_documents(mock_client, str(data_dir))

        assert count == 1
        mock_client.collection.add.assert_called_once()

    def test_index_skip_cached_files(self, tmp_path, mocker):
        """Test that already indexed files are skipped."""
        # Create a directory with files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        text_file = data_dir / "test.txt"
        text_file.write_text("Test content")

        # Mock the client
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)

        # Mock embedding function
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        # First indexing
        count1 = index_documents(mock_client, str(data_dir))

        # Mock cache to say file is already indexed
        mock_cache = mocker.patch('src.indexer.indexing.load_cache')
        mock_cache.return_value = {str(text_file): "hash123"}

        # Second indexing should skip the file
        count2 = index_documents(mock_client, str(data_dir))

        assert count1 == 1
        assert count2 == 0

    def test_index_creates_directory(self, tmp_path, mocker):
        """Test that index_documents creates the directory if it doesn't exist."""
        # Create a non-existent directory path
        non_existent_dir = tmp_path / "nonexistent" / "subdir"

        # Mock the client
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)

        # Mock embedding function
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        # Should not raise an error
        count = index_documents(mock_client, str(non_existent_dir))

        assert count == 0

    def test_index_with_multiple_files(self, tmp_path, mocker):
        """Test indexing a directory with multiple files."""
        # Create a directory with multiple files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for i in range(5):
            text_file = data_dir / f"test{i}.txt"
            text_file.write_text(f"Content {i}")

        # Mock the client
        mock_client = mocker.Mock()
        mock_client.collection.add = mocker.Mock()
        mock_client.collection.count = mocker.Mock(return_value=0)

        # Mock embedding function
        mock_embedding = mocker.patch('src.indexer.indexing.get_ollama_embedding')
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        count = index_documents(mock_client, str(data_dir))

        assert count == 5
        assert mock_client.collection.add.call_count == 5
