"""Custom exceptions for the Semantic Document Indexer."""

from __future__ import annotations


class IndexerError(Exception):
    """Base exception for indexer-related errors."""
    pass


class ConfigurationError(IndexerError):
    """Raised when configuration is invalid or missing."""
    pass


class OllamaError(IndexerError):
    """Raised when Ollama API operations fail."""
    pass


class EmbeddingError(OllamaError):
    """Raised when embedding generation fails."""
    pass


class ChromaDBError(IndexerError):
    """Raised when ChromaDB operations fail."""
    pass


class FileProcessingError(IndexerError):
    """Raised when file processing fails."""
    pass