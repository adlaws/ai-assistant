"""Server module for the custom HTTP server."""

from __future__ import annotations

import os
import sys

from .handler import DocumentHandler
from .response import ErrorResponse, HTMLResponse, JSONResponse, SuccessResponse


def run_server(host: str = '127.0.0.1', port: int = 8000, chroma_db_path: str | None = None) -> None:
    """Start the web server and run initial indexing.

    :param host: Host address to bind to
    :param port: Port number to listen on
    :param chroma_db_path: Optional path to ChromaDB storage
    """
    print(f"Starting server on http://{host}:{port}")
    print(f"Open http://localhost:{port} in your browser")
    print("\nTest questions:")
    print("  - What is AI?")
    print("  - List the places that Andrew has worked")
    print("  - How does machine learning work?")
    print("\nPress Ctrl+C to stop the server\n")

    # Set paths for handler
    if chroma_db_path:
        DocumentHandler.chroma_db_path = chroma_db_path
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DocumentHandler.root_path = root_path

    # Create server
    from http.server import HTTPServer
    from socketserver import ThreadingMixIn

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Handle requests in separate threads."""
        daemon_threads = True

    server = ThreadedHTTPServer((host, port), DocumentHandler)

    print(f"Server listening on {host}:{port}")
    print("Press Ctrl+C to stop the server\n")

    # --- INITIALIZATION STEP ADDED ---
    print("\n[INFO] Attempting initial document indexing...")
    try:
        from src.chroma_client import create_client
        from src.indexer.indexing import index_documents
        from src.indexer.config import DB_PATH, DATA_DIR
        chroma_client = create_client(DB_PATH)
        # Index documents
        index_count = index_documents(chroma_client, DATA_DIR)
        print(f"[SUCCESS] Initial Indexing complete. Indexed {index_count} documents.")
    except Exception as e:
        print(f"[WARNING] Could not perform initial indexing: {e}")
    # --------------------------------

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n=== Shutdown signal received ===")
    finally:
        print("\n\n=== Initiating shutdown ===")
        print("Stopping server...")
        server.shutdown()
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    run_server()
