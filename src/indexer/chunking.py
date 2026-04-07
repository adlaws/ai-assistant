"""Text chunking utilities for document processing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """Chunk text into overlapping segments.

    :param text: Text to chunk
    :param chunk_size: Size of each chunk in characters (default: 300)
    :param chunk_overlap: Number of overlapping characters between chunks (default: 50)
    :return: List of text chunks
    """
    if not text:
        return []

    # Handle edge case: chunk_size <= 0 returns entire text as single chunk
    if chunk_size <= 0:
        return [text]

    chunks = []
    start = 0

    # If overlap >= chunk_size, it doesn't make sense semantically; set to 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = 0

    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step

    return chunks
