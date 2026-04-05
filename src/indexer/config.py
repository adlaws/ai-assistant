"""Configuration settings for the Semantic Document Indexer."""

# === DIRECTORIES ===
DATA_DIR = "my_documents"  # Folder containing documents to index
DB_PATH = "./chroma_db"    # ChromaDB persistent storage path

# === MODEL CONFIGURATION ===
EMBEDDING_MODEL = "nomic-embed-text"  # Text embedding model from Ollama
LLM_MODEL = "qwen3.5:9b"              # Primary LLM for generating responses

# === API CONFIGURATION ===
OLLAMA_BASE_URL = "http://localhost:11434"
API_TIMEOUT = 120  # seconds

# === SEARCH CONFIGURATION ===
DEFAULT_N_RESULTS = 2  # Number of top results to retrieve per query

# === DEPENDENCIES ===
# Required dependencies (install with pip):
# pip install chromadb ollama requests pdfplumber pillow markdown beautifulsoup4
REQUIRED_DEPENDENCIES = [
    "chromadb",
    "ollama",
    "requests",
    "pdfplumber",
    "pillow",
    "markdown",
    "beautifulsoup4",
    "python-docx",
]