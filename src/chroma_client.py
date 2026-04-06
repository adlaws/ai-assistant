"""ChromaDB client with Ollama embeddings."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.indexer.api_client import create_client as create_ollama_client
from src.indexer import index_file, get_file_handler
from src.indexer.config import COLLECTION_NAME, DB_PATH, IndexerConfig
from src.indexer.exceptions import ChromaDBError, OllamaError

logger = logging.getLogger(__name__)

# Constants for ChromaDB configuration
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_DB_PATH = "chroma_db"


class ChromaClient:
    """Client for ChromaDB with Ollama embeddings."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        db_path: Optional[str] = DEFAULT_DB_PATH,
        create_if_missing: bool = False
    ) -> None:
        """Initialize ChromaDB client.

        :param collection_name: Name of the collection to create/use
        :param embedding_model: Name of the Ollama embedding model (e.g., "nomic-embed-text")
        :param db_path: Path to persistent ChromaDB storage
        :param create_if_missing: Whether to create collection if it doesn't exist
        :raises ChromaDBError: If ChromaDB initialization fails
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

    def _create_embedding_fn(self, model: Optional[str] = DEFAULT_EMBEDDING_MODEL) -> Optional[Callable[[str], list[float]]]:
        """Create Ollama embedding function.

        :param model: Name of the Ollama embedding model
        :return: Callable that takes text and returns embedding vector, or None if creation fails
        :raises OllamaError: If Ollama client creation fails
        """
        if not model:
            model = DEFAULT_EMBEDDING_MODEL

        try:
            ollama_client = create_ollama_client()

            def embedding_fn(text: str) -> list[float]:
                return ollama_client.get_embedding(text)

            return embedding_fn
        except OllamaError as e:
            logger.error("Error creating embedding function: %s", str(e))
            return None

    def add_document(self, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Add a document to the collection.

        :param content: Document content
        :param metadata: Optional metadata dictionary
        :raises ChromaDBError: If adding document fails
        :raises OllamaError: If embedding generation fails
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

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        embeddings: Optional[list[list[float]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None
    ) -> None:
        """Add multiple documents to the collection.

        :param documents: List of document contents
        :param ids: List of document IDs
        :param embeddings: Optional list of embeddings
        :param metadatas: Optional list of metadata dictionaries
        :raises ChromaDBError: If adding documents fails
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
        """Search for similar documents.

        :param query: Search query string
        :param n_results: Number of results to return
        :return: List of matching results
        :raises ChromaDBError: If search fails
        :raises OllamaError: If embedding generation fails
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
        """Retrieve all documents stored in the collection.

        :return: List of all documents found
        :raises ChromaDBError: If retrieval fails
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
        """Get the number of documents in the collection.

        :return: Number of documents
        :raises ChromaDBError: If count retrieval fails
        """
        try:
            return self.collection.count()
        except Exception as e:
            raise ChromaDBError(f"Failed to get document count: {e}") from e

    def get_ids(self) -> set[str]:
        """Get all document IDs in the collection.

        :return: Set of document IDs
        :raises ChromaDBError: If ID retrieval fails
        """
        try:
            results = self.collection.get(include=[])
            return set(results.get('ids', []))
        except Exception as e:
            raise ChromaDBError(f"Failed to get document IDs: {e}") from e

    def delete_collection(self) -> None:
        """Delete the current collection.

        :raises ChromaDBError: If deletion fails
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = None  # Clear the reference
            logger.info("Deleted collection: %s", self.collection_name)
        except Exception as e:
            logger.error("Could not delete collection %s: %s", self.collection_name, str(e))
            self.collection = None  # Clear reference even on error

    def create_client(self, db_path: Optional[str] = DEFAULT_DB_PATH) -> "ChromaClient":
        """Create and return a ChromaClient instance.

        :param db_path: Optional path to ChromaDB storage
        :return: ChromaClient instance
        :raises ChromaDBError: If client creation fails
        """
        return ChromaClient(
            collection_name=COLLECTION_NAME,
            db_path=db_path
        )


def create_client(db_path: Optional[str] = DEFAULT_DB_PATH) -> ChromaClient:
    """Create and return a ChromaClient instance.

    :param db_path: Optional path to ChromaDB storage
    :return: ChromaClient instance
    :raises ChromaDBError: If client creation fails
    """
    return ChromaClient(
        collection_name=COLLECTION_NAME,
        db_path=db_path
    )
