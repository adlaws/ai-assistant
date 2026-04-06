"""
Standalone API for AI assistant indexing.
Use src/indexer/indexer.py for full submodule functionality.
"""

from __future__ import annotations

import os
from .indexer import (
    index_file,
    _chunk_text,
    index_documents,
    get_ollama_embedding,
    get_file_handler,
    FILE_HANDLERS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DB_PATH = os.path.join(DATA_DIR, 'indexer.db')