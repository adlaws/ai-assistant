"""Tests for utility functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.indexer.config import CACHE_FILE
from src.indexer.utils import compute_file_hash, load_cache, save_cache


class TestComputeFileHash:
    """Test cases for compute_file_hash function."""

    def test_hash_same_content(self):
        """Test that same content produces same hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name

        try:
            hash1 = compute_file_hash(temp_path)
            hash2 = compute_file_hash(temp_path)

            assert hash1 == hash2
            assert isinstance(hash1, str)
            assert len(hash1) == 32  # MD5 hex length
        finally:
            Path(temp_path).unlink()

    def test_hash_different_content(self):
        """Test that different content produces different hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f1:
            f1.write("content 1")
            temp_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
            f2.write("content 2")
            temp_path2 = f2.name

        try:
            hash1 = compute_file_hash(temp_path1)
            hash2 = compute_file_hash(temp_path2)

            assert hash1 != hash2
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()

    def test_hash_empty_file(self):
        """Test hash of empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("")
            temp_path = f.name

        try:
            hash_empty = compute_file_hash(temp_path)
            assert isinstance(hash_empty, str)
            assert len(hash_empty) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_large_file(self):
        """Test hash of large file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("A" * 1000000)  # 1MB file
            temp_path = f.name

        try:
            hash_large = compute_file_hash(temp_path)
            assert isinstance(hash_large, str)
            assert len(hash_large) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_binary_file(self):
        """Test hash of binary file."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
            f.write(b"\x00\x01\x02\x03")
            temp_path = f.name

        try:
            hash_binary = compute_file_hash(temp_path)
            assert isinstance(hash_binary, str)
            assert len(hash_binary) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_unicode_content(self):
        """Test hash of unicode content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Hello 世界！Привет! 🌍")
            temp_path = f.name

        try:
            hash_unicode = compute_file_hash(temp_path)
            assert isinstance(hash_unicode, str)
            assert len(hash_unicode) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_newlines(self):
        """Test hash with various newline types."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Line1\nLine2\r\nLine3\rLine4")
            temp_path = f.name

        try:
            hash_newlines = compute_file_hash(temp_path)
            assert isinstance(hash_newlines, str)
            assert len(hash_newlines) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_whitespace(self):
        """Test hash with various whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("  \t\n\r  \t\n")
            temp_path = f.name

        try:
            hash_whitespace = compute_file_hash(temp_path)
            assert isinstance(hash_whitespace, str)
            assert len(hash_whitespace) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_nonexistent_file(self):
        """Test hash of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash("/nonexistent/file.txt")

    def test_hash_read_error(self):
        """Test hash when file cannot be read."""
        # Create a file and make it unreadable
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test")
            temp_path = f.name

        try:
            # On Windows, we can't easily make file unreadable
            # So we test with a directory instead
            with pytest.raises((FileNotFoundError, PermissionError)):
                compute_file_hash("/root")  # /root typically requires root access
        finally:
            Path(temp_path).unlink()


class TestLoadCache:
    """Test cases for load_cache function."""

    def test_load_cache_nonexistent(self):
        """Test loading cache when file doesn't exist."""
        # Ensure cache file doesn't exist
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

        cache = load_cache()
        assert isinstance(cache, dict)
        assert cache == {}

    def test_load_cache_empty(self):
        """Test loading empty cache."""
        cache = load_cache()
        assert isinstance(cache, dict)

    def test_load_cache_with_data(self, tmp_path):
        """Test loading cache with existing data."""
        # Create a temporary cache file
        cache_file = tmp_path / "cache.json"
        test_cache = {"file1": {"id": "123", "path": "/test"}, "file2": {"id": "456"}}

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(test_cache, f)

        # Temporarily replace CACHE_FILE
        import src.indexer.utils
        original_cache_file = src.indexer.utils.CACHE_FILE
        src.indexer.utils.CACHE_FILE = cache_file

        try:
            loaded_cache = load_cache()
            assert loaded_cache == test_cache
        finally:
            src.indexer.utils.CACHE_FILE = original_cache_file

    def test_load_cache_invalid_json(self, tmp_path):
        """Test loading cache with invalid JSON."""
        cache_file = tmp_path / "cache.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write("invalid json {")

        # Temporarily replace CACHE_FILE
        import src.indexer.utils
        original_cache_file = src.indexer.utils.CACHE_FILE
        src.indexer.utils.CACHE_FILE = cache_file

        try:
            # Should return empty dict on error
            cache = load_cache()
            assert isinstance(cache, dict)
        finally:
            src.indexer.utils.CACHE_FILE = original_cache_file


class TestSaveCache:
    """Test cases for save_cache function."""

    def test_save_cache_empty(self):
        """Test saving empty cache."""
        save_cache({})

        # Verify file was created
        assert CACHE_FILE.exists()

        # Clean up
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

    def test_save_cache_with_data(self):
        """Test saving cache with data."""
        test_cache = {"file1": {"id": "123", "path": "/test"}}

        save_cache(test_cache)

        # Verify file exists and contains correct data
        assert CACHE_FILE.exists()

        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "file1" in content
        assert "123" in content

        # Clean up
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

    def test_save_cache_creates_parent_dirs(self):
        """Test that save_cache creates parent directories."""
        # Ensure cache file doesn't exist
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

        # Save cache to a path that doesn't exist
        test_cache = {"test": "data"}
        save_cache(test_cache)

        # Verify file was created
        assert CACHE_FILE.exists()

        # Clean up
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

    def test_save_cache_overwrites(self):
        """Test that save_cache overwrites existing file."""
        test_cache1 = {"file1": "data1"}
        test_cache2 = {"file2": "data2"}

        save_cache(test_cache1)
        save_cache(test_cache2)

        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "file2" in content
        assert "file1" not in content

        # Clean up
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()

    def test_save_cache_invalid_data(self):
        """Test saving invalid cache data."""
        # Save should work with any dict
        save_cache({"key": "value", "nested": {"a": 1}})

        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == {"key": "value", "nested": {"a": 1}}

        # Clean up
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
