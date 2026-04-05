"""ChromaDB client for vector storage."""

import chromadb

from .config import DB_PATH


class ChromaClient:
    """Client for interacting with ChromaDB."""
    
    def __init__(self, path: str = DB_PATH):
        """
        Initialize the ChromaDB client.
        
        Args:
            path: Path to persistent storage
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.get_collection()
    
    def get_collection(self, name: str = "text_docs") -> chromadb.Collection:
        """
        Get or create a collection for storing text documents.
        
        Args:
            name: Collection name (default: "text_docs")
            
        Returns:
            chromadb.Collection: The collection instance
        """
        collection = self.client.get_or_create_collection(
            name=name,
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


def create_client(path: str = DB_PATH) -> ChromaClient:
    """Create and return a ChromaDB client instance."""
    return ChromaClient(path=path)