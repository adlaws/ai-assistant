"""Pydantic models for the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """Response model for a document."""
    id: Optional[int] = None
    content: str
    metadata: dict
    embedding: Optional[list[float]] = None


class SearchResponse(BaseModel):
    """Response model for search results."""
    summary: str
    sources: List[dict]
    query: str
    total_sources: int


class IndexRequest(BaseModel):
    """Request model for indexing a document."""
    content: str
    metadata: Optional[dict[str, str]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    database_path: str
    document_count: int
    ollama_available: bool
