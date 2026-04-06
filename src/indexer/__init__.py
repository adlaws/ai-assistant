"""Indexer module exports."""

from __future__ import annotations

from .indexer import (
    index_file,
    index_documents,
    get_file_handler,
)

__all__ = [
    'index_file',
    'index_documents',
    'get_file_handler',
]
