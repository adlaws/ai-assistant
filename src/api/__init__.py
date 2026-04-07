"""API module for the Semantic Document Indexer."""

from __future__ import annotations

from .endpoints import app
from .models import DocumentResponse, SearchResponse, IndexRequest, HealthResponse
from .templates import HTML_TEMPLATE

__all__ = [
    'app',
    'DocumentResponse',
    'SearchResponse',
    'IndexRequest',
    'HealthResponse',
    'HTML_TEMPLATE',
]
