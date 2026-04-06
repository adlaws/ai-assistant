"""ChromaDB client for vector storage."""

from __future__ import annotations

import chromadb
import os
import json

from src.indexer.config import DB_PATH, COLLECTION_NAME


class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function for Ollama."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or 'nomic-embed-text'

        # Ollama API base URL and timeout
        import os
        OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
        self.timeout = int(os.environ.get('OLLAMA_TIMEOUT', '120'))
        self.embedding_model = model_name or 'nomic-embed-text'
        self.base_url = OLLAMA_BASE_URL

    def _call_ollama_api(self, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request to the Ollama API."""
        import requests
        url = f"{self.base_url or OLLAMA_BASE_URL}/api/{endpoint}"
        kwargs["stream"] = False
        response = requests.post(url, json=kwargs, timeout=self.timeout)

        if response.status_code != 200:
            raise requests.HTTPError(f"HTTP {response.status_code}: {response.text}")

        # Clean response (strip INFO/DEBUG lines)
        response_text = response.text.strip()
        lines = response_text.split('\n')
        cleaned_lines = [line for line in lines if not line.startswith(('DEBUG:', 'INFO:', 'WARNING:'))]
        cleaned_response = '\n'.join(cleaned_lines).strip()

        return json.loads(cleaned_response)

    def get_embedding(self, text: str) -> list:
        """Generate embedding for text via Ollama API."""
        import requests
        if not text or len(text.strip()) == 0:
            return [0.0] * 512

        try:
            response = self._call_ollama_api(
                "embeddings",
                model=self.embedding_model,
                prompt=text
            )
            return response.get("embedding", [])
        except Exception as e:
            print(f"[WARNING] Embedding error: {str(e)}")
            return [0.0] * 512

    def __call__(self, input):
        """Generate embeddings for multiple texts."""
        input_list = input if isinstance(input, list) else [input]
        embeddings = []

        for text in input_list:
            if text and len(text) > 0:
                embeddings.append(self.get_embedding(text))
            else:
                embeddings.append([0.0] * 512)

        return embeddings

    def embed_query(self, text: str) -> list:
        """Generate embedding for a single query."""
        return self.get_embedding(text)

    def name(self) -> str:
        return "ollama-" + self.model_name


class ChromaClient:
    """Client for interacting with ChromaDB."""

    def __init__(self, path: str = None):
        """
        Initialize the ChromaDB client.

        Args:
            path: Path to persistent storage (default from config)
        """
        self.path = path or DB_PATH
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.get_collection()

    def get_collection(self, name: str = COLLECTION_NAME) -> chromadb.Collection:
        """
        Get or create a collection for storing text documents.

        Args:
            name: Collection name (default from config)

        Returns:
            chromadb.Collection: The collection instance
        """
        # Use OllamaEmbeddingFunction for consistency with indexing
        emb_fn = OllamaEmbeddingFunction()

        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=emb_fn,
            metadata={"description": "Text document embeddings"}
        )
        return collection

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            int: Number of documents
        """
        return self.collection.count()

    def get_ids(self) -> set[str]:
        """
        Get all document IDs in the collection.

        Returns:
            set[str]: Set of document IDs
        """
        result = self.collection.get()
        return set(result.get('ids', []))

    def add_documents(self, documents: list[str], ids: list[str],
                      embeddings: list[list[float]], metadatas: list[dict] = None) -> None:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts
            ids: List of unique IDs for each document
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
        """
        self.collection.add(
            documents=documents,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas or []
        )

    def query(self, query_embeddings: list[list[float]], n_results: int = 2) -> dict:
        """
        Query the collection for similar documents.

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return

        Returns:
            dict: Query results containing documents, ids, distances, metadatas
        """
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )

    def delete_collection(self, name: str = None) -> None:
        """
        Delete a collection (use with caution).

        Args:
            name: Collection name to delete (default: default collection)
        """
        if name is None:
            self.client.delete_collection(self.collection.name)
        else:
            self.client.delete_collection(name)


def create_client(path: str = None) -> ChromaClient:
    """Create and return a ChromaDB client instance."""
    return ChromaClient(path=path)