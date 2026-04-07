# Semantic Document Indexer

A Retrieval-Augmented Generation (RAG) system for indexing and searching documents using local Ollama embeddings and ChromaDB vector storage.

## Features

* **Document Indexing**: Support for multiple file formats (PDF, DOCX, TXT, MD, CSV, JSON, images)
* **Semantic Search**: Find documents using natural language queries
* **Local AI**: Uses Ollama for embeddings and LLM responses
* **Vector Storage**: ChromaDB for efficient similarity search
* **Web Interface**: FastAPI-based REST API with interactive search
* **CLI Tool**: Command-line interface for indexing and querying

## Quick Start

Assuming all pre-requisites are met (see below), you can start the server by activating the Python
virtualenv and running the server.

The virtualenv is contained in the `.venv` folder in the project root, so the following one-liner
run in Powershell from the root folder should start the server:

```powershell
.\.venv\Scripts\activate && python src/server.py
```

## Prerequisites

* Python 3.10+
* Ollama running locally with required models:
    * `nomic-embed-text` (for embeddings)
    * `qwen3.5:9b` (for LLM responses)

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd semantic-doc-indexer
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** and pull required models:

   ```bash
   # Install Ollama from [https://ollama.ai](https://ollama.ai)

   # Pull the required models
   ollama pull nomic-embed-text
   ollama pull qwen3.5:9b
   ```

## Configuration

The application uses Pydantic-based configuration. You can customize settings by:

1. **Environment variables**:

   ```bash
   export INDEXER_OLLAMA_BASE_URL="http://localhost:11434"
   export INDEXER_DOCUMENT_FOLDERS='["data", "docs"]'
   ```

2. **JSON config file** (`src/indexer/config.json`):

   ```json
   {
     "document_folders": ["data"],
     "embedding_model": "nomic-embed-text",
     "llm_model": "qwen3.5:9b",
     "ollama_base_url": "http://localhost:11434",
     "collection_name": "documents"
   }
   ```

## Usage

### CLI Interface

**Index documents**:

```bash
python -m src.indexer.main
```

This will:

* Scan the `data/` directory for supported files
* Generate embeddings using Ollama
* Store vectors in ChromaDB
* Provide an interactive query interface

**Supported file types**:

* `.txt`, `.md` - Text files
* `.pdf` - PDF documents
* `.docx` - Word documents
* `.csv`, `.json` - Structured data
* `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp` - Images (metadata only)

### Web API

**Start the FastAPI server**:

```bash
python -m src.api
```

The API will be available at `http://localhost:8000`

**API Endpoints**:

* `GET /` - Web interface
* `GET /health` - Health check
* `GET /documents` - List all indexed documents
* `GET /search?query=<text>&n=<count>` - Search documents
* `POST /index` - Add new document

**Example API usage**:

```bash
# Search for documents
curl "http://localhost:8000/search?query=artificial%20intelligence&n=3"

# Add a new document
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your document text here", "metadata": {"source": "api"}}'
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy src/

# Formatting
black src/
isort src/

# Linting
flake8 src/
```

### Project Structure

```
src/
├── indexer/
│   ├── __init__.py
│   ├── api_client.py      # Ollama API client
│   ├── chroma_client.py   # ChromaDB client
│   ├── config.py          # Configuration management
│   ├── exceptions.py      # Custom exceptions
│   ├── file_handlers.py   # Document parsers
│   ├── logging_config.py  # Logging setup
│   └── main.py           # CLI entry point
├── api.py                # FastAPI server
└── server.py            # Legacy HTTP server (deprecated)

tests/                    # Test suite
docs/                    # Documentation and web interface
data/                    # Sample documents
```

## Architecture

1. **Document Processing**: Files are parsed by format-specific handlers
2. **Embedding Generation**: Ollama generates vector embeddings
3. **Vector Storage**: ChromaDB stores and indexes vectors
4. **Query Processing**: Search queries are embedded and matched against stored vectors
5. **Response Generation**: Relevant documents are provided as context for LLM responses

## Troubleshooting

### Common Issues

**Ollama connection failed**:

* Ensure Ollama is running: `ollama serve`
* Check models are pulled: `ollama list`
* Verify base URL in configuration

**No documents found**:

* Check files exist in configured document folders
* Ensure files have supported extensions
* Check file permissions

**Import errors**:

* Install dependencies: `pip install -r requirements.txt`
* Ensure Python 3.10+ is used

### Logs

Logs are written to console by default. For file logging, modify the logging configuration in `src/indexer/logging_config.py`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]
