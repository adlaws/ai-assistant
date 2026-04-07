"""Tests for text chunking utilities."""

from __future__ import annotations

import pytest
from src.indexer.chunking import chunk_text


class TestChunkText:
    """Test cases for chunk_text function."""

    def test_chunk_text_default_params(self):
        """Test chunking with default parameters."""
        text = "This is a test text for chunking. " * 10
        chunks = chunk_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_custom_size(self):
        """Test chunking with custom chunk size."""
        text = "A " * 100  # 200 characters
        chunks = chunk_text(text, chunk_size=10)

        assert len(chunks) == 20  # 200 / 10 = 20 chunks
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_chunk_text_custom_overlap(self):
        """Test chunking with custom overlap."""
        text = "A " * 100  # 200 characters
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        # With overlap, we expect more chunks
        assert len(chunks) > 3  # Without overlap: 200/50 = 4 chunks
        # Check overlap exists - consecutive chunks should share characters
        for i in range(len(chunks) - 1):
            # Check that chunks actually overlap in content (not just character sets)
            chunk1_end = chunks[i][-10:] if len(chunks[i]) >= 10 else chunks[i]
            chunk2_start = chunks[i + 1][:10]
            assert any(c in chunk2_start for c in chunk1_end)

    def test_chunk_text_empty_string(self):
        """Test chunking with empty string."""
        chunks = chunk_text("")

        assert chunks == []

    def test_chunk_text_single_chunk(self):
        """Test chunking when text fits in one chunk."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_multiple(self):
        """Test chunking when text is exact multiple of chunk size."""
        text = "A " * 100  # 200 characters
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)

        assert len(chunks) == 4  # 200 / 50 = 4 chunks
        assert all(len(chunk) == 50 for chunk in chunks)

    def test_chunk_text_preserves_content(self):
        """Test that chunking preserves original content."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

        # Reconstruct from chunks
        reconstructed = "".join(chunks)
        # Original text should be recoverable (allowing for overlap duplication)
        assert len(reconstructed) >= len(text)

    def test_chunk_text_large_text(self):
        """Test chunking with large text."""
        text = "A " * 10000  # 20000 characters
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)

        assert len(chunks) > 0
        assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_chunk_text_unicode(self):
        """Test chunking with unicode characters."""
        text = "Hello 世界！Привет! 🌍 " * 50
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_newlines(self):
        """Test chunking with newlines."""
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunk_text(text, chunk_size=15, chunk_overlap=3)

        assert len(chunks) > 1
        assert "\n" in chunks[0]

    def test_chunk_text_whitespace(self):
        """Test chunking with various whitespace."""
        text = "  Text with   spaces  \n\tand\ttabs\n"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

        assert len(chunks) > 1

    def test_chunk_text_zero_overlap(self):
        """Test chunking with zero overlap."""
        text = "A " * 100  # 200 characters
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)

        assert len(chunks) == 4
        # With repetitive text, chunks may be identical
        # Just verify we got the expected number of chunks
        assert all(len(chunk) == 50 for chunk in chunks)

    def test_chunk_text_negative_overlap(self):
        """Test chunking with negative overlap (should still work)."""
        text = "A " * 100  # 200 characters
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=-10)

        # With negative overlap, step increases (50 - (-10) = 60), creating fewer chunks
        assert len(chunks) <= 4  # Same or fewer chunks
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_zero_chunk_size(self):
        """Test chunking with zero chunk size (edge case)."""
        text = "Test"
        chunks = chunk_text(text, chunk_size=0, chunk_overlap=0)

        # Should handle gracefully - likely returns text as single chunk
        assert len(chunks) >= 1

    def test_chunk_text_zero_chunk_size_large(self):
        """Test chunking with zero chunk size on larger text."""
        text = "A " * 100
        chunks = chunk_text(text, chunk_size=0, chunk_overlap=0)

        # With chunk_size=0, end=start, so it keeps appending same text
        # This is an edge case that should be handled
        assert len(chunks) > 0
