"""File handlers for different document types.
This module has been refactored into:
- src/indexer/file_processing.py: File handler classes
- src/indexer/utils.py: Utility functions
"""

from __future__ import annotations

from .file_processing import (
    BaseHandler,
    CSVHandler,
    JSONHandler,
    MarkdownHandler,
    PDFHandler,
    PythonHandler,
    TextHandler,
    WordHandler,
    ImageHandler,
    FILE_HANDLERS,
    get_handler,
)

__all__ = [
    'BaseHandler',
    'CSVHandler',
    'JSONHandler',
    'MarkdownHandler',
    'PDFHandler',
    'PythonHandler',
    'TextHandler',
    'WordHandler',
    'ImageHandler',
    'FILE_HANDLERS',
    'get_handler',
]
