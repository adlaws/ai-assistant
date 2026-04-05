"""Configuration settings for the Semantic Document Indexer."""

import hashlib
from pathlib import Path

# Get the project root directory (go up three levels from src/indexer to workspace root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directory for documents
DATA_DIR = PROJECT_ROOT / 'my_documents'

# ChromaDB database path
DB_PATH = PROJECT_ROOT / 'chroma_db'

# Embedding model to use
EMBEDDING_MODEL = 'nomic-embed-text'

# LLM model to use for generating responses
LLM_MODEL = 'qwen3.5:9b'

# Default number of results to return from query
DEFAULT_N_RESULTS = 2

# Base URL for Ollama API
OLLAMA_BASE_URL = 'http://localhost:11434'

# API timeout in seconds
API_TIMEOUT = 120

# Cache file for tracking indexed documents (stores file hash and index timestamp)
CACHE_FILE = DB_PATH / 'document_cache.json'


def compute_file_hash(filepath: str) -> str:
    """
    Compute a hash of a file's content.
    
    Args:
        filepath: Path to the file
        
    Returns:
        str: MD5 hash of the file content
    """
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
            import json
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
    import json
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)