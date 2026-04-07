"""Custom HTTP server for the Semantic Document Indexer Web Interface."""

from __future__ import annotations

import os
import sys
import threading
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# Add workspace root to Python path for module imports
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


class DocumentHandler(BaseHTTPRequestHandler):
    """Custom request handler that serves ChromaDB data and the SPA."""

    protocol_version = 'HTTP/1.1'
    chroma_db_path: str | None = None
    root_path: str | None = None

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Initialize the request handler.

        :param args: Positional arguments passed to parent class
        :param kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.root_path = kwargs.get('root_path',
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def log_message(self, format: str, *args: str) -> None:
        """Log messages.

        :param format: Log format string
        :param args: Format arguments
        """
        import traceback
        try:
            sys.stdout.write(f"{self.address_string()} - {format % args}\n")
            sys.stdout.flush()
        except Exception:
            pass

    def handle_error(self, status: int, message: str) -> None:
        """Handle errors with detailed logging.

        :param status: HTTP status code
        :param message: Error message
        """
        import traceback
        sys.stderr.write(f"\n=== ERROR HANDLED ===\n")
        sys.stderr.write(f"Path: {self.path}\n")
        sys.stderr.write(f"Status: {status}\n")
        sys.stderr.write(f"Message: {message}\n")
        sys.stderr.write(f"Traceback:\n")
        traceback.print_exc()
        sys.stderr.write("\n\n")

    def send_json_response(self, data: dict, status: int = 200) -> None:
        """Send a JSON response.

        :param data: JSON data to send
        :param status: HTTP status code
        """
        response = json.dumps(data, indent=2)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode())

    def send_html_response(self, html: bytes | str, status: int = 200) -> None:
        """Send an HTML response - handles both bytes and str.

        :param html: HTML content as bytes or string
        :param status: HTTP status code
        """
        if isinstance(html, str):
            response = html.encode('utf-8')
        else:
            response = html
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
        self.wfile.flush()

    def handle_root(self) -> None:
        """Serve the web interface or API info.

        :raises Exception: If serving the web interface fails
        """
        try:
            # Build path to spa.html
            spa_path = os.path.join(self.root_path, 'docs', 'spa.html')
            print(f"Looking for spa.html at: {spa_path}", flush=True)

            if os.path.exists(spa_path):
                print(f"Found spa.html", flush=True)
                with open(spa_path, 'rb') as f:
                    content = f.read()
                print(f"Sending HTML response ({len(content)} bytes)", flush=True)
                self.send_html_response(content)
                return

            # Fallback HTML if spa.html doesn't exist
            html_content = b"""<!DOCTYPE html>
<html>
<head><title>Semantic Document Indexer API</title></head>
<body style="font-family: sans-serif;">
<h1 style="color: #333;">Semantic Document Indexer API</h1>
<p>Web frontend not found. This server provides API endpoints for interacting with the document index.</p>
<h2 style="border-bottom: 1px solid #ccc; padding-bottom: 5px;">Available Endpoints:</h2>
<ul style="list-style-type: none; padding-left: 0;">
<li style="margin-bottom: 10px;"><a href="/documents" style="color: #007bff; text-decoration: none;">/documents</a> - Get all indexed documents</li>
<li style="margin-bottom: 10px;"><a href="/health" style="color: #007bff; text-decoration: none;">/health</a> - Health check</li>
<li style="margin-bottom: 10px;"><a href="/search" style="color: #007bff; text-decoration: none;">/search</a> - Search with text query</li>
</ul>
<h2 style="border-bottom: 1px solid #ccc; padding-bottom: 5px;">Test Questions:</h2>
<ul style="list-style-type: none; padding-left: 0;">
<li style="margin-bottom: 8px;">What is AI?</li>
<li style="margin-bottom: 8px;">List the places that Andrew has worked</li>
<li style="margin-bottom: 8px;">How does machine learning work?</li>
</ul>
<p>Use /search endpoint to search for answers to questions.</p>
</body>
</html>"""
            print(f"Sending fallback HTML", flush=True)
            self.send_html_response(html_content)
        except Exception as e:
            print(f"Error in handle_root: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.send_json_response({'error': str(e)}, 500)

    def do_GET(self) -> None:
        """Handle GET requests.

        :raises Exception: If request handling fails
        """
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        print(f"GET request for: {self.path}", flush=True)

        # Handle favicon.ico
        if path == '/favicon.ico':
            self.send_response(404)
            self.end_headers()
            return

        try:
            if path == '/search':
                query = query_params.get('query', [None])[0]
                if not query:
                    self.send_json_response({'error': 'Query parameter is required'}, 400)
                else:
                    self.handle_search(query)

            elif path == '/documents':
                self.handle_documents()

            elif path == '/count':
                self.handle_count()

            elif path == '/health':
                self.send_json_response({'status': 'ok'})

            elif path == '/docs' or path == '/':
                self.handle_root()

            else:
                self.send_json_response({'error': 'Not found'}, 404)
        except Exception as e:
            print(f"Error in handler: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.send_json_response({'error': str(e)}, 500)

    def do_POST(self) -> None:
        """Handle POST requests.

        :raises Exception: If request handling fails
        """
        if self.path == '/search':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                query = self.rfile.read(content_length).decode('utf-8')
                self.handle_search(query)
            except Exception as e:
                self.send_json_response({'error': str(e)}, 400)

        elif self.path == '/index':
            try:
                self.handle_index()
            except Exception as e:
                self.send_json_response({'error': str(e)}, 500)
        else:
            self.send_json_response({'error': 'Not found'}, 404)

    def handle_search(self, query: str) -> None:
        """Handle text-based search queries.

        :param query: Search query string
        :raises Exception: If search fails
        """
        try:
            # Import required components dynamically to avoid circular dependency issues on initial load
            from src.chroma_client import create_client as create_chroma_client
            from src.indexer.api_client import create_client as create_ollama_client
            from src.indexer.config import LLM_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL

            # Query ChromaDB using the search method (which handles embedding generation internally)
            chroma_client = create_chroma_client()

            results = chroma_client.search(
                query,
                n_results=5
            )

            results_list = []
            if results and results.get('documents') and len(results['documents']) > 0:
                # Extract data from ChromaDB query result structure
                documents = results['documents'][0]  # List of document texts
                metadatas = results['metadatas'][0] if results.get('metadatas') else []  # List of metadata dicts
                ids = results['ids'][0] if results.get('ids') else []  # List of document IDs
                distances = results['distances'][0] if results.get('distances') else []  # List of distances

                # Build context from retrieved documents
                context_parts = []
                for i, (doc_content, metadata) in enumerate(zip(documents, metadatas)):
                    # Use 'source' key from metadata (not 'filename')
                    source = metadata.get('source', 'unknown')
                    filepath = metadata.get('filepath', '')
                    filetype = metadata.get('type', '')
                    context_parts.append(f"=== Document: {source} ===\n{doc_content}")

                context = '\n\n---\n\n'.join(context_parts)

                # Generate summary response using LLM
                ollama_client = create_ollama_client()
                prompt = f"""You are an AI assistant with access to documents. Please answer the following question based on the provided context:

Context: {context}

Question: {query}

Answer:
"""

                response = ollama_client.generate_response(prompt)

                # Parse response to extract answer and references
                answer = response.strip()

                # Build references list using 'source' from metadata
                references = []
                for i, (doc_content, metadata) in enumerate(zip(documents, metadatas)):
                    source = metadata.get('source', 'unknown')
                    references.append(f"Source: {source}")

                results_list.append({
                    'answer': answer,
                    'references': references,
                    'documents': documents,
                    'metadatas': metadatas,
                    'ids': ids,
                    'distances': distances
                })

            self.send_json_response({'results': results_list})

        except Exception as e:
            import traceback
            error_msg = f"[ERROR] {str(e)}\n[DEBUG] {traceback.format_exc()}"
            print(error_msg, flush=True)
            self.send_json_response({'error': str(e)}, 500)

    def handle_documents(self) -> None:
        """Return all indexed documents.

        :raises Exception: If document retrieval fails
        """
        try:
            # Import required components dynamically
            from src.chroma_client import create_client
            from src.indexer.config import COLLECTION_NAME, DB_PATH

            client = create_client(DB_PATH)

            # Use the dedicated list_documents method
            documents = client.list_documents()

            self.send_json_response({'documents': documents})

        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)

    def handle_count(self) -> None:
        """Return document count.

        :raises Exception: If count retrieval fails
        """
        try:
            from src.chroma_client import create_client
            from src.indexer.config import DB_PATH

            client = create_client(DB_PATH)
            count = client.collection.count()

            self.send_json_response({'count': count})

        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)

    def handle_index(self) -> None:
        """Handle document indexing via API.

        :raises Exception: If indexing fails
        """
        try:
            from src.chroma_client import create_client
            from src.indexer.indexer import index_documents
            from src.indexer.config import DB_PATH, DATA_DIR

            chroma_client = create_client(DB_PATH)
            index_count = index_documents(chroma_client, DATA_DIR)

            response = {
                'status': 'success',
                'indexed': index_count,
                'message': f'Successfully indexed {index_count} documents'
            }

            self.send_json_response(response, 200)

        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)


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
    DocumentHandler.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create server
    server = ThreadedHTTPServer((host, port), DocumentHandler)

    print(f"Server listening on {host}:{port}")
    print("Press Ctrl+C to stop the server\n")

    # --- INITIALIZATION STEP ADDED ---
    print("\n[INFO] Attempting initial document indexing...")
    try:
        from src.chroma_client import create_client
        from src.indexer.indexer import index_documents
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
