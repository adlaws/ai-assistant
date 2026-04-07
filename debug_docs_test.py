"""Debug script to see what's happening with get_documents endpoint."""

import sys
import logging
from unittest.mock import MagicMock

# Enable all logging
logging.basicConfig(level=logging.DEBUG)

# Mock before importing app
mock_chroma = MagicMock()
mock_chroma.collection_name = "documents"
mock_chroma.list_documents.return_value = [
    {
        "id": 1,  # Use integer, not string
        "content": "Test document 1",
        "metadata": {"source": "test", "type": "txt"},
        "embedding": [0.1, 0.2, 0.3]
    }
]

mock_ollama = MagicMock()

# Patch before importing anything else
import src.api.endpoints
src.api.endpoints.chroma_client = mock_chroma
src.api.endpoints.ollama_client = mock_ollama

from fastapi.testclient import TestClient
app = src.api.endpoints.app
client = TestClient(app)

# Test the endpoint
try:
    response = client.get('/documents')
    print(f'Status: {response.status_code}')
    if response.status_code != 200:
        print(f'Error: {response.text}')
        print(f'Response details: {response}')
except Exception as e:
    print(f'Exception: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
