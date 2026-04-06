"""Tests for file handlers."""

from __future__ import annotations

import pytest
import tempfile
import os
from pathlib import Path
from src.indexer.file_handlers import (
    BaseHandler, TextHandler, PDFHandler, WordHandler,
    PNGHandler, get_handler
)
from src.indexer.exceptions import FileProcessingError


class TestBaseHandler:
    """Test cases for BaseHandler."""

    def test_abstract_methods(self):
        """Test that BaseHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHandler("test.txt")

    def test_get_filename(self):
        """Test filename extraction."""
        # Create a concrete subclass for testing
        class ConcreteHandler(BaseHandler):
            def load_document(self) -> str:
                return "test"

        handler = ConcreteHandler("/path/to/test.txt")
        assert handler.get_filename() == "test.txt"


class TestTextHandler:
    """Test cases for TextHandler."""

    def test_load_text_file(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, world!")
            temp_path = f.name

        try:
            handler = TextHandler(temp_path)
            content = handler.load_document()
            assert content == "Hello, world!"
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        handler = TextHandler("/nonexistent/file.txt")

        with pytest.raises(FileProcessingError):
            handler.load_document()


class TestGetHandler:
    """Test cases for get_handler function."""

    def test_get_text_handler(self):
        """Test getting handler for text file."""
        handler = get_handler("test.txt")
        assert isinstance(handler, TextHandler)

    def test_get_pdf_handler(self):
        """Test getting handler for PDF file."""
        handler = get_handler("test.pdf")
        assert isinstance(handler, PDFHandler)

    def test_get_word_handler(self):
        """Test getting handler for Word file."""
        handler = get_handler("test.docx")
        assert isinstance(handler, WordHandler)

    def test_get_png_handler(self):
        """Test getting handler for PNG file."""
        handler = get_handler("test.png")
        assert isinstance(handler, PNGHandler)

    def test_unsupported_extension(self):
        """Test getting handler for unsupported extension."""
        with pytest.raises(FileProcessingError, match="No handler available"):
            get_handler("test.xyz")

    def test_path_traversal_prevention(self):
        """Test that path traversal is prevented."""
        with pytest.raises(FileProcessingError, match="Invalid filepath"):
            get_handler("../etc/passwd")

        with pytest.raises(FileProcessingError, match="Invalid filepath"):
            get_handler("/absolute/path.txt")