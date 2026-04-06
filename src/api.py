"""FastAPI backend for the Semantic Document Indexer Web Interface."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .indexer.chroma_client import create_client as create_chroma_client
from .indexer.api_client import create_client as create_ollama_client
from .indexer.config import DB_PATH
from .indexer.exceptions import OllamaError, ChromaDBError, EmbeddingError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Document Indexer API",
    description="API for semantic document search using Ollama and ChromaDB",
    version="1.0.0"
)

# Create clients
chroma_client = create_chroma_client(str(DB_PATH))
ollama_client = create_ollama_client()


@app.on_event("startup")
async def startup_event() -> None:
    """Index documents on startup.

    :param chroma_client: ChromaDB client instance
    :raises Exception: If document indexing fails
    """
    try:
        from .indexer.indexer import index_documents
        from .indexer.config import DATA_DIR

        # Try to delete existing collection (ignore if it doesn't exist)
        try:
            chroma_client.delete_collection()
        except Exception as e:
            logger.debug("[DEBUG] Failed to delete collection: %s", e)

        # Recreate collection
        chroma_client.collection = chroma_client.client.get_or_create_collection(
            name=chroma_client.collection_name,
            embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        )

        # Index documents
        count = index_documents(chroma_client, str(DATA_DIR))
        logger.info("[STARTUP] Indexed %d documents", count)
    except Exception as e:
        logger.warning("[STARTUP WARNING] Could not index documents: %s", e)
        # Try to recreate collection even if indexing failed
        try:
            chroma_client.collection = chroma_client.client.get_or_create_collection(
                name=chroma_client.collection_name,
                embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
            )
        except Exception as e2:
            logger.debug("[DEBUG] Failed to recreate collection: %s", e2)


# Pydantic models
class DocumentResponse(BaseModel):
    """Response model for a document."""
    id: Optional[int] = None
    content: str
    metadata: dict


class SearchResponse(BaseModel):
    """Response model for search results."""
    summary: str
    sources: List[dict]
    query: str
    total_sources: int


class IndexRequest(BaseModel):
    """Request model for indexing a document."""
    content: str
    metadata: Optional[dict[str, str]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    database_path: str
    document_count: int
    ollama_available: bool


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Document Indexer</title>
</head>
<body>
    <h1>404 - Web Interface Not Found</h1>
    <p>The web interface (indexer-web.html) is not available.</p>
    <p>Start the indexing tool first: python src/indexer/main.py</p>
</body>
</html>
"""


@app.get("/")
async def root() -> FileResponse | str:
    """Serve the web interface HTML file.

    :return: HTML file response or fallback template
    """
    web_path = Path(__file__).parent.parent / 'docs' / 'indexer-web.html'
    if web_path.exists():
        return FileResponse(web_path)
    return HTML_TEMPLATE


@app.get("/documents", response_model=List[DocumentResponse])
async def get_documents(n: int = Query(100, ge=1, le=1000)) -> List[DocumentResponse]:
    """Get all indexed documents.

    :param n: Maximum number of documents to return
    :return: List of documents

    :raises HTTPException: If database error occurs
    """
    try:
        documents = chroma_client.list_documents()
        # Limit results
        limited_docs = documents[:n]

        return [
            DocumentResponse(
                id=doc.get('id'),
                content=doc.get('content', ''),
                metadata=doc.get('metadata', {}),
                embedding=doc.get('embedding')
            )
            for doc in limited_docs
        ]
    except ChromaDBError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/search", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, max_length=1000),
    n: int = Query(5, ge=1, le=50)
) -> SearchResponse:
    """Search for documents similar to the query text.

    :param query: Search query text
    :param n: Number of results to return
    :return: Search results with matching documents

    :raises HTTPException: If service or database error occurs
    """
    try:
        results = chroma_client.search(query, n_results=n)

        sources = []
        context_text = ""

        if results and 'documents' in results and 'metadatas' in results:
            docs = results['documents'][0] if isinstance(results['documents'], list) and results['documents'] else []
            metas = results['metadatas'][0] if isinstance(results['metadatas'], list) and results['metadatas'] else []

            for i, (doc, meta) in enumerate(zip(docs, metas)):
                # Truncate document content for context (first 1000 chars)
                truncated_content = doc[:1000] + "..." if len(doc) > 1000 else doc
                context_text += f"\nDocument {i+1}: {truncated_content}\n"

                sources.append({
                    "filename": meta.get("source", "Unknown"),
                    "filepath": meta.get("filepath", ""),
                    "type": meta.get("type", ""),
                    "preview": truncated_content[:200] + "..." if len(truncated_content) > 200 else truncated_content
                })

        # Generate summary using LLM
        if context_text.strip():
            prompt = f"""Based on the following documents, please provide a comprehensive summary answer to this question: "{query}"

Documents:
{context_text}

Please provide a clear, concise summary that directly answers the question using information from the documents. If the documents don't contain relevant information, say so."""

            try:
                summary = ollama_client.generate_response(prompt, temperature=0.3)
            except Exception as e:
                summary = f"I found {len(sources)} relevant documents but couldn't generate a summary due to: {str(e)}. Please check the sources below."
        else:
            summary = "No relevant documents found for your query."

        return SearchResponse(
            summary=summary,
            sources=sources,
            query=query,
            total_sources=len(sources)
        )
    except (OllamaError, EmbeddingError) as e:
        raise HTTPException(status_code=503, detail=f"Service error: {e}")
    except ChromaDBError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    :return: Health status information

    :raises HTTPException: If health check fails
    """
    try:
        doc_count = chroma_client.count()

        # Test Ollama availability
        ollama_available = False
        try:
            ollama_client.get_embedding("test")
            ollama_available = True
        except (OllamaError, EmbeddingError):
            pass

        return HealthResponse(
            status="healthy" if ollama_available else "degraded",
            database_path=str(DB_PATH),
            document_count=doc_count,
            ollama_available=ollama_available
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/index")
async def index_documents(request: IndexRequest) -> dict:
    """Add a new document to the index.

    :param request: Document content and metadata
    :return: Success message

    :raises HTTPException: If embedding or database error occurs
    """
    try:
        chroma_client.add_document(
            content=request.content,
            metadata=request.metadata or {"source": "api"}
        )

        return {"status": "success", "message": "Document indexed successfully"}
    except (OllamaError, EmbeddingError) as e:
        raise HTTPException(status_code=503, detail=f"Embedding service error: {e}")
    except ChromaDBError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/reindex")
async def reindex_documents() -> dict:
    """Re-index all documents from the data directory.

    :return: Success message with count of indexed documents

    :raises HTTPException: If re-indexing fails
    """
    try:
        from .indexer.indexer import index_documents
        from .indexer.config import DATA_DIR

        # Delete existing collection to clear old data
        chroma_client.delete_collection()
        # Recreate the collection
        chroma_client.collection = chroma_client.client.get_or_create_collection(
            name=chroma_client.collection_name,
            embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        )
        # Re-index all documents
        count = index_documents(chroma_client, str(DATA_DIR))

        return {"status": "success", "message": f"Re-indexed {count} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {e}")


@app.get("/download/{filename}")
async def download_file(filename: str) -> FileResponse:
    """Download a file from the data directory.

    :param filename: Name of the file to download
    :return: File response for download

    :raises HTTPException: If file not found
    """
    from .indexer.config import DATA_DIR

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )


# Mount static files if docs directory exists
docs_path = Path(__file__).parent.parent / 'docs'
if docs_path.exists():
    app.mount("/docs", StaticFiles(directory=str(docs_path)), name="docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
