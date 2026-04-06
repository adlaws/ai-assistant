"""Tests for configuration module."""

from __future__ import annotations

import pytest
import tempfile
import json
from pathlib import Path
from src.indexer.config import (
    IndexerConfig, compute_file_hash, load_cache, save_cache, CACHE_FILE
)
from src.indexer.exceptions import ConfigurationError


class TestIndexerConfig:
    """Test cases for IndexerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IndexerConfig()

        assert config.document_folders == ["data"]
        assert config.collection_name == "documents"
        assert config.embedding_model == "nomic-embed-text"
        assert config.llm_model == "qwen3.5:9b"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_timeout == 120

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = IndexerConfig(
            document_folders=["data", "docs"],
            ollama_timeout=60
        )
        assert config.document_folders == ["data", "docs"]
        assert config.ollama_timeout == 60

    def test_invalid_document_folder(self):
        """Test validation of nonexistent document folder."""
        with pytest.raises(ValueError, match="Document folder does not exist"):
            IndexerConfig(document_folders=["/nonexistent/folder"])

    def test_invalid_ollama_url(self):
        """Test validation of invalid Ollama URL."""
        with pytest.raises(ValueError, match="Ollama base URL must start with"):
            IndexerConfig(ollama_base_url="invalid-url")

    def test_invalid_timeout(self):
        """Test validation of invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            IndexerConfig(ollama_timeout=0)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_compute_file_hash(self):
        """Test file hash computation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            hash1 = compute_file_hash(temp_path)
            hash2 = compute_file_hash(temp_path)

            # Same content should produce same hash
            assert hash1 == hash2
            assert isinstance(hash1, str)
            assert len(hash1) == 32  # MD5 hex length
        finally:
            Path(temp_path).unlink()

    def test_load_cache_nonexistent(self):
        """Test loading cache when file doesn't exist."""
        # Ensure cache file doesn't exist
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

        cache = load_cache()
        assert isinstance(cache, dict)
        assert cache == {}

    def test_save_and_load_cache(self):
        """Test saving and loading cache."""
        test_cache = {"file1": {"id": "123", "path": "/test"}}

        # Save cache
        save_cache(test_cache)

        # Load cache
        loaded_cache = load_cache()

        assert loaded_cache == test_cache

        # Clean up
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()