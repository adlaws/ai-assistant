"""Test script to debug search results."""

from __future__ import annotations

import sys
sys.path.insert(0, 'src')
from src.indexer.api_client import create_client as create_ollama_client
from src.indexer.chroma_client import create_client as create_chroma_client
from src.indexer.config import EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME, DEFAULT_N_RESULTS

# Check if collection exists and has data
ollama_client = create_ollama_client()
chroma_client = create_chroma_client(DB_PATH)

# Test query
query = "What is AI?"
query_embedding = ollama_client.get_embedding(query)
print(f"Query: {query}")
print(f"Query embedding shape: {len(query_embedding)}")

# Query the collection
results = chroma_client.query(
    query_embeddings=[query_embedding],
    n_results=DEFAULT_N_RESULTS
)
print(f"Query results: {results}")
print(f"Has documents: {'documents' in results}")
print(f"Number of results: {len(results.get('documents', []))}")