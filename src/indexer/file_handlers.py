"""File handlers for different document types."""

from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import Type
import pandas as pd
import hashlib
from io import BytesIO

from pypdf import PdfReader
from docx import Document
from PIL import Image

from .exceptions import FileProcessingError


class BaseHandler(ABC):
    """Base class for file handlers."""

    def __init__(self, filepath: str) -> None:
        """Initialize handler with file path.

        Args:
            filepath: Path to the file to handle
        """
        self.filepath = filepath

    @abstractmethod
    def load_document(self) -> str:
        """
        Load the document content.

        Returns:
            str: Document text content

        Raises:
            FileProcessingError: If loading fails
        """
        pass

    def get_filename(self) -> str:
        """
        Get the filename.

        Returns:
            str: Filename
        """
        return os.path.basename(self.filepath)


class TextHandler(BaseHandler):
    """Handler for text files (.txt, .md)."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Load text file content.

        Returns:
            str: File content

        Raises:
            FileProcessingError: If file cannot be read
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

        Returns:
            str: File content

        Raises:
            FileProcessingError: If file cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileProcessingError(f"Failed to read markdown file {self.filepath}: {e}") from e


class CSVHandler(BaseHandler):
    """Handler for CSV files."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Load CSV file as formatted string.

        Returns:
            str: CSV content as string

        Raises:
            FileProcessingError: If CSV cannot be read
        """
        try:
            df = pd.read_csv(self.filepath)
            return df.to_string()
        except Exception as e:
            raise FileProcessingError(f"Failed to read CSV file {self.filepath}: {e}") from e


class JSONHandler(BaseHandler):
    """Handler for JSON files."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Load JSON file as formatted string.

        Returns:
            str: JSON content as formatted string

        Raises:
            FileProcessingError: If JSON cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), indent=2)
        except Exception as e:
            raise FileProcessingError(f"Failed to read JSON file {self.filepath}: {e}") from e


class PythonHandler(BaseHandler):
    """Handler for Python files."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Load Python file content.

        Returns:
            str: File content

        Raises:
            FileProcessingError: If file cannot be read
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileProcessingError(f"Failed to read Python file {self.filepath}: {e}") from e


class PDFHandler(BaseHandler):
    """Handler for PDF files."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Extract text from PDF file.

        Returns:
            str: Extracted text content

        Raises:
            FileProcessingError: If PDF cannot be read
        """
        try:
            reader = PdfReader(self.filepath)
            text_chunks = [page.extract_text() or "" for page in reader.pages]
            return "".join(text_chunks)
        except Exception as e:
            raise FileProcessingError(f"Failed to read PDF file {self.filepath}: {e}") from e


class WordHandler(BaseHandler):
    """Handler for Word documents (.docx)."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Extract text from Word document.

        Returns:
            str: Extracted text content

        Raises:
            FileProcessingError: If document cannot be read
        """
        try:
            doc = Document(self.filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise FileProcessingError(f"Failed to read Word document {self.filepath}: {e}") from e


class PNGHandler(BaseHandler):
    """Handler for PNG images."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Extract image metadata.

        Returns:
            str: Image metadata description

        Raises:
            FileProcessingError: If image cannot be read
        """
        try:
            with Image.open(self.filepath) as img:
                # Get image dimensions and format
                width, height = img.size
                mode = img.mode
                format_name = img.format

                return f"Image: {format_name} ({width}x{height}, mode={mode})"
        except Exception as e:
            raise FileProcessingError(f"Failed to read PNG image {self.filepath}: {e}") from e


class JPGHandler(BaseHandler):
    """Handler for JPG images."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Extract image metadata.

        Returns:
            str: Image metadata description

        Raises:
            FileProcessingError: If image cannot be read
        """
        try:
            with Image.open(self.filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format

                return f"Image: {format_name} ({width}x{height}, mode={mode})"
        except Exception as e:
            raise FileProcessingError(f"Failed to read JPG image {self.filepath}: {e}") from e


class JPEGHandler(JPGHandler):
    """Handler for JPEG images."""
    pass


class GIFHandler(BaseHandler):
    """Handler for GIF images."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Extract image metadata including frame count.

        Returns:
            str: Image metadata description

        Raises:
            FileProcessingError: If image cannot be read
        """
        try:
            with Image.open(self.filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                n_frames = getattr(img, 'n_frames', 1)

                return f"Image: {format_name} ({width}x{height}, mode={mode}, frames={n_frames})"
        except Exception as e:
            raise FileProcessingError(f"Failed to read GIF image {self.filepath}: {e}") from e


class BMPHandler(BaseHandler):
    """Handler for BMP images."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

    def load_document(self) -> str:
        """Extract image metadata.

        Returns:
            str: Image metadata description

        Raises:
            FileProcessingError: If image cannot be read
        """
        try:
            with Image.open(self.filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format

                return f"Image: {format_name} ({width}x{height}, mode={mode})"
        except Exception as e:
            raise FileProcessingError(f"Failed to read BMP image {self.filepath}: {e}") from e


def get_handler(filepath: str) -> BaseHandler:
    """
    Get the appropriate handler for a file based on its extension.

    Args:
        filepath: Path to the file

    Returns:
        BaseHandler: Handler instance for the file type

    Raises:
        FileProcessingError: If no handler is available or path is invalid
    """
    # Validate filepath to prevent path traversal
    if not filepath or '..' in filepath or filepath.startswith('/'):
        raise FileProcessingError(f"Invalid filepath: {filepath}")

    _, ext = os.path.splitext(filepath.lower())

    handlers: dict[str, Type[BaseHandler]] = {
        '.txt': TextHandler,
        '.md': MarkdownHandler,
        '.csv': CSVHandler,
        '.json': JSONHandler,
        '.py': PythonHandler,
        '.pdf': PDFHandler,
        '.docx': WordHandler,
        '.png': PNGHandler,
        '.jpg': JPGHandler,
        '.jpeg': JPEGHandler,
        '.gif': GIFHandler,
        '.bmp': BMPHandler,
    }

    handler_class = handlers.get(ext)

    if handler_class is None:
        raise FileProcessingError(f"No handler available for file extension: {ext}")

    return handler_class(filepath)