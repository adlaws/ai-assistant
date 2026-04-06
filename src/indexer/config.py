"""Configuration settings for the Semantic Document Indexer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class IndexerConfig(BaseSettings):
    """Configuration for the Semantic Document Indexer."""

    # Document settings
    document_folders: List[str] = Field(default=["data"], description="Folders to scan for documents")
    collection_name: str = Field(default="documents", description="ChromaDB collection name")
    default_n_results: int = Field(default=2, description="Default number of search results")

    # Model settings
    embedding_model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    llm_model: str = Field(default="qwen3.5:9b", description="Ollama LLM model")

    # API settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_timeout: int = Field(default=120, description="Ollama API timeout in seconds")

    # Storage settings
    chromadb_path: str = Field(default="chroma_db", description="ChromaDB storage path")

    # Security settings
    enable_api_key_auth: bool = Field(default=False, description="Enable API key authentication")

    @field_validator('document_folders')
    def validate_document_folders(cls, v: List[str]) -> List[str]:
        """Validate document folders exist."""
        for folder in v:
            if not Path(folder).exists():
                raise ValueError(f"Document folder does not exist: {folder}")
        return v

    @field_validator('ollama_base_url')
    def validate_ollama_url(cls, v: str) -> str:
        """Validate Ollama URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Ollama base URL must start with http:// or https://")
        return v

    @field_validator('ollama_timeout')
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    model_config = ConfigDict(
        env_file=".env",
        env_prefix="INDEXER_",
    )


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load configuration
try:
    CONFIG_FILE = PROJECT_ROOT / 'src' / 'indexer' / 'config.json'
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config = IndexerConfig(**config_data)
    else:
        # Use defaults if config file doesn't exist
        config = IndexerConfig()
except Exception as e:
    raise ConfigurationError(f"Failed to load configuration: {e}") from e

# Extract configuration values for backward compatibility
DOCUMENT_FOLDERS = [Path(PROJECT_ROOT) / folder for folder in config.document_folders]
CHROMADB_PATH = Path(PROJECT_ROOT) / config.chromadb_path
DB_PATH = CHROMADB_PATH  # Alias for backward compatibility with server.py
DATA_DIR = Path(PROJECT_ROOT) / "data"  # Separate data directory from DB
EMBEDDING_MODEL = config.embedding_model
LLM_MODEL = config.llm_model
COLLECTION_NAME = config.collection_name
DEFAULT_N_RESULTS = config.default_n_results
OLLAMA_BASE_URL = config.ollama_base_url
OLLAMA_TIMEOUT = config.ollama_timeout
API_TIMEOUT = OLLAMA_TIMEOUT  # Alias for backward compatibility
ENABLE_API_KEY_AUTH = config.enable_api_key_auth

# Cache file for tracking indexed documents
CACHE_FILE = CHROMADB_PATH / 'document_cache.json'

def compute_file_hash(filepath: str) -> str:
    """
    Compute a hash of a file's content.

    Args:
        filepath: Path to the file

    Returns:
        str: MD5 hash of the file content
    """
    import hashlib
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_cache() -> dict:
    """
    Load the document cache from disk.

    Returns:
        dict: Cache data with file hashes
    """
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    """
    Save the document cache to disk.

    Args:
        cache: Cache data to save
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)