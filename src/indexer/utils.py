"""Utility functions for the indexer module."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src.indexer.config import CACHE_FILE

logger = logging.getLogger(__name__)


def compute_file_hash(filepath: str) -> str:
    """Compute a hash of a file's content.

    :param filepath: Path to the file
    :return: MD5 hash of the file content
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_cache() -> dict:
    """Load the document cache from disk.

    :return: Cache data with file hashes
    :raises FileNotFoundError: If cache file does not exist
    :raises IOError: If cache file cannot be read
    """
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load cache: %s", str(e))
            return {}
    return {}


def save_cache(cache: dict) -> None:
    """Save the document cache to disk.

    :param cache: Cache data to save
    :raises IOError: If cache file cannot be written
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
