"""File processing functions for different document types."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Type

import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image
from io import BytesIO

from src.indexer.exceptions import FileProcessingError

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Base class for file handlers."""

    def __init__(self, filepath: str) -> None:
        """Initialize handler with file path.

        :param filepath: Path to the file to handle
        """
        self.filepath = filepath

    @abstractmethod
    def load_document(self) -> str:
        """Load the document content.

        :return: Document text content
        :raises FileProcessingError: If loading fails
        """
        pass

    def get_filename(self) -> str:
        """Get the filename.

        :return: Filename
        """
        return os.path.basename(self.filepath)


class TextHandler(BaseHandler):
    """Handler for text files (.txt, .md)."""

    def load_document(self) -> str:
        """Load text file content.

        :return: File content
        :raises FileProcessingError: If file cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileProcessingError(f"Failed to read text file {self.filepath}: {e}") from e


class MarkdownHandler(TextHandler):
    """Handler for Markdown files (.md)."""

    def load_document(self) -> str:
        """Load markdown file content.

        :return: File content
        :raises FileProcessingError: If file cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileProcessingError(f"Failed to read markdown file {self.filepath}: {e}") from e


class CSVHandler(BaseHandler):
    """Handler for CSV files."""

    def load_document(self) -> str:
        """Load CSV file as formatted string.

        :return: CSV content as string
        :raises FileProcessingError: If CSV cannot be read
        """
        try:
            df = pd.read_csv(self.filepath)
            return df.to_string()
        except Exception as e:
            raise FileProcessingError(f"Failed to read CSV file {self.filepath}: {e}") from e


class JSONHandler(BaseHandler):
    """Handler for JSON files."""

    def load_document(self) -> str:
        """Load JSON file as formatted string.

        :return: JSON content as formatted string
        :raises FileProcessingError: If JSON cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), indent=2)
        except Exception as e:
            raise FileProcessingError(f"Failed to read JSON file {self.filepath}: {e}") from e


class PythonHandler(BaseHandler):
    """Handler for Python files."""

    def load_document(self) -> str:
        """Load Python file content.

        :return: File content
        :raises FileProcessingError: If file cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileProcessingError(f"Failed to read Python file {self.filepath}: {e}") from e


class PDFHandler(BaseHandler):
    """Handler for PDF files."""

    def load_document(self) -> str:
        """Load PDF file as text.

        :return: PDF text content
        :raises FileProcessingError: If PDF cannot be read
        """
        try:
            reader = PdfReader(self.filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise FileProcessingError(f"Failed to read PDF file {self.filepath}: {e}") from e


class WordHandler(BaseHandler):
    """Handler for Word documents (.docx)."""

    def load_document(self) -> str:
        """Load Word document as text.

        :return: Document text content
        :raises FileProcessingError: If document cannot be read
        """
        try:
            doc = Document(self.filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise FileProcessingError(f"Failed to read Word document {self.filepath}: {e}") from e


class ImageHandler(BaseHandler):
    """Handler for image files."""

    def load_document(self) -> str:
        """Load image file as base64 string.

        :return: Base64 encoded image
        :raises FileProcessingError: If image cannot be read
        """
        try:
            with Image.open(self.filepath) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return f"Image: {self.get_filename()}\nBase64: {buffered.getvalue()!r}"
        except Exception as e:
            raise FileProcessingError(f"Failed to read image file {self.filepath}: {e}") from e


# Map file extensions to handlers
FILE_HANDLERS: dict[str, Type[BaseHandler]] = {
    '.txt': TextHandler,
    '.md': MarkdownHandler,
    '.csv': CSVHandler,
    '.json': JSONHandler,
    '.py': PythonHandler,
    '.pdf': PDFHandler,
    '.docx': WordHandler,
    '.png': ImageHandler,
    '.jpg': ImageHandler,
    '.jpeg': ImageHandler,
    '.gif': ImageHandler,
    '.bmp': ImageHandler,
}


def get_handler(filepath: str) -> BaseHandler:
    """Get appropriate handler for a file.

    :param filepath: Path to the file
    :return: Appropriate handler instance
    :raises FileProcessingError: If no handler is available for the file type
    """
    _, ext = os.path.splitext(filepath.lower())
    handler_class = FILE_HANDLERS.get(ext)

    if handler_class is None:
        raise FileProcessingError(f"No handler available for file type: {ext}")

    return handler_class(filepath)
