"""
Standalone API for AI assistant indexing.
Use src/indexer/indexer.py for full submodule functionality.
"""

from __future__ import annotations

import logging
import os

from src.indexer.indexer import (
    index_file,
    _chunk_text,
    index_documents,
    get_ollama_embedding,
    get_file_handler,
    FILE_HANDLERS,
)

logger = logging.getLogger(__name__)

# Constants for indexer configuration
DEFAULT_DATA_DIR = "data"
DEFAULT_DB_PATH = "chroma_db"


def get_indexer_paths() -> tuple[str, str]:
    """Get the data directory and database path for the indexer.

    :return: Tuple of (data_dir, db_path)
    """
    data_dir = os.path.join(os.path.dirname(__file__), DEFAULT_DATA_DIR)
    db_path = os.path.join(data_dir, DEFAULT_DB_PATH)
    return data_dir, db_path
