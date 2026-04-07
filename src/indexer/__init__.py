"""Indexer module for semantic document indexing."""

from __future__ import annotations

from .api_client import create_client as create_ollama_client
from .chunking import chunk_text
from .config import (
    COLLECTION_NAME,
    DATA_DIR,
    DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    ENABLE_API_KEY_AUTH,
)
from .exceptions import ChromaDBError, ConfigurationError, EmbeddingError, FileProcessingError, OllamaError
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
from .indexing import (
    extract_image_description,
    get_ollama_embedding,
    index_documents,
    index_file,
    process_markdown_file,
    process_pdf_file,
    process_text_file,
    process_word_document,
)
from .utils import compute_file_hash, load_cache, save_cache

__all__ = [
    # API Client
    'create_ollama_client',
    # Chunking
    'chunk_text',
    # Config
    'COLLECTION_NAME',
    'DATA_DIR',
    'DB_PATH',
    'EMBEDDING_MODEL',
    'LLM_MODEL',
    'OLLAMA_BASE_URL',
    'OLLAMA_TIMEOUT',
    'ENABLE_API_KEY_AUTH',
    # Exceptions
    'ChromaDBError',
    'ConfigurationError',
    'EmbeddingError',
    'FileProcessingError',
    'OllamaError',
    # File Processing
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
    # Indexing
    'extract_image_description',
    'get_ollama_embedding',
    'index_documents',
    'index_file',
    'process_markdown_file',
    'process_pdf_file',
    'process_text_file',
    'process_word_document',
    # Utils
    'compute_file_hash',
    'load_cache',
    'save_cache',
]
