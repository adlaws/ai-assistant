"""ChromaDB client with Ollama embeddings."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .api_client import create_client as create_ollama_client
from . import index_file, get_file_handler
from .config import COLLECTION_NAME, DB_PATH, IndexerConfig
from .exceptions import ChromaDBError, OllamaError


class ChromaClient:
    """Client for ChromaDB with Ollama embeddings."""

    def __init__(self, collection_name: str, embedding_model: Optional[str] = None, db_path: Optional[str] = None, create_if_missing: bool = False):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the collection to create/use
            embedding_model: Name of the Ollama embedding model (e.g., "nomic-embed-text")
            db_path: Path to persistent ChromaDB storage
            create_if_missing: Whether to create collection if it doesn't exist

        Raises:
            ChromaDBError: If ChromaDB initialization fails
        """
        self.collection_name = collection_name
        self.db_path = db_path

        # Get configuration
        config = IndexerConfig()

        try:
            if db_path:
                self.client = chromadb.PersistentClient(path=str(db_path))
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction()  # Placeholder
            )

            self.embedding_fn: Optional[Callable[[str], list[float]]] = self._create_embedding_fn(embedding_model)

            if create_if_missing and self.embedding_fn is not None:
                # Check if collection already has documents
                try:
                    existing_count = self.collection.count()
                    if existing_count == 0:
                        # Recreate collection with correct embedding function only if empty
                        self.client.delete_collection(collection_name)
                        self.collection = self.client.get_or_create_collection(
                            name=collection_name,
                            embedding_function=embedding_functions.OllamaEmbeddingFunction(
                                url=config.ollama_base_url,
                                model_name=embedding_model or config.embedding_model
                            )
                        )
                except Exception:
                    # If we can't check count, assume recreation is needed
                    self.client.delete_collection(collection_name)
                    self.collection = self.client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=embedding_functions.OllamaEmbeddingFunction(
                            url=config.ollama_base_url,
                            model_name=embedding_model or config.embedding_model
                        )
                    )
        except Exception as e:
            raise ChromaDBError(f"Failed to initialize ChromaDB client: {e}") from e

    def _create_embedding_fn(self, model: Optional[str] = None) -> Optional[Callable[[str], list[float]]]:
        """Create Ollama embedding function.

        Args:
            model: Name of the Ollama embedding model

        Returns:
            Callable that takes text and returns embedding vector, or None if creation fails

        Raises:
            OllamaError: If Ollama client creation fails
        """
        if not model:
            model = "nomic-embed-text"

        try:
            ollama_client = create_ollama_client()

            def embedding_fn(text: str) -> list[float]:
                return ollama_client.get_embedding(text)

            return embedding_fn
        except OllamaError as e:
            print(f"Error creating embedding function: {e}")
            return None

    def add_document(self, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """
        Add a document to the collection.

        Args:
            content: Document content
            metadata: Optional metadata dictionary

        Raises:
            ChromaDBError: If adding document fails
            OllamaError: If embedding generation fails
        """
        if self.embedding_fn is None:
            raise ChromaDBError("No embedding function available")

        try:
            embedding = self.embedding_fn(content)

            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else [{"source": "api"}]
            )
        except Exception as e:
            raise ChromaDBError(f"Failed to add document: {e}") from e

    def add_documents(self, documents: list[str], ids: list[str], embeddings: Optional[list[list[float]]] = None, metadatas: Optional[list[dict[str, Any]]] = None) -> None:
        """
        Add multiple documents to the collection.

        Args:
            documents: List of document contents
            ids: List of document IDs
            embeddings: Optional list of embeddings
            metadatas: Optional list of metadata dictionaries

        Raises:
            ChromaDBError: If adding documents fails
        """
        if self.embedding_fn is None and embeddings is None:
            raise ChromaDBError("No embedding function available and no embeddings provided")

        try:
            if embeddings is None and self.embedding_fn is not None:
                embeddings = [self.embedding_fn(doc) for doc in documents]

            self.collection.add(
                documents=documents,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            raise ChromaDBError(f"Failed to add documents: {e}") from e

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query string
            n_results: Number of results to return

        Returns:
            List of matching results

        Raises:
            ChromaDBError: If search fails
            OllamaError: If embedding generation fails
        """
        if self.embedding_fn is None:
            raise ChromaDBError("No embedding function available")

        try:
            query_embedding = self.embedding_fn(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            return results
        except Exception as e:
            raise ChromaDBError(f"Search failed: {e}") from e

    def list_documents(self) -> list[dict[str, Any]]:
        """
        Retrieve all documents stored in the collection.

        Returns:
            List of all documents found.

        Raises:
            ChromaDBError: If retrieval fails
        """
        try:
            # ChromaDB's get() method retrieves all documents metadata, embeddings, etc.
            results = self.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )

            documents = []
            if results.get('documents') and results.get('metadatas'):
                docs = results['documents']
                metas = results['metadatas']
                embeddings = results.get('embeddings', [])

                for i, (doc, meta) in enumerate(zip(docs, metas)):
                    embedding = None
                    if embeddings and len(embeddings) > i:
                        embedding = embeddings[i]
                    documents.append({
                        'id': i,
                        'content': doc,
                        'metadata': meta,
                        'embedding': embedding
                    })
            return documents
        except Exception as e:
            raise ChromaDBError(f"Failed to list documents: {e}") from e

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            int: Number of documents

        Raises:
            ChromaDBError: If count retrieval fails
        """
        try:
            return self.collection.count()
        except Exception as e:
            raise ChromaDBError(f"Failed to get document count: {e}") from e

    def get_ids(self) -> set[str]:
        """
        Get all document IDs in the collection.

        Returns:
            set[str]: Set of document IDs

        Raises:
            ChromaDBError: If ID retrieval fails
        """
        try:
            results = self.collection.get(include=[])
            return set(results.get('ids', []))
        except Exception as e:
            raise ChromaDBError(f"Failed to get document IDs: {e}") from e

    def delete_collection(self) -> None:
        """Delete the current collection.

        Raises:
            ChromaDBError: If deletion fails
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = None  # Clear the reference
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Could not delete collection {self.collection_name}: {e}")
            self.collection = None  # Clear reference even on error

def create_client(db_path: Optional[str] = None) -> ChromaClient:
    """Create and return a ChromaClient instance.

    Args:
        db_path: Optional path to ChromaDB storage

    Returns:
        ChromaClient: Instance of the ChromaDB client
    """
    return ChromaClient(
        collection_name=COLLECTION_NAME,
        db_path=db_path
    )