"""
Semantic Document Indexer
=========================

A local RAG (Retrieval-Augmented Generation) system that indexes documents
from specified directories and enables natural language querying using
Ollama-based LLMs and ChromaDB for vector storage.

Supported File Types:
- Text files (.txt)
- PDF documents (.pdf)
- Images (.png, .jpg, .jpeg, .gif, .bmp, .webp)

Features:
- Semantic search using embedding models
- Interactive query interface
- Persistent vector database storage
- Modular file processing with appropriate handlers for each file type

Usage:
    python src/indexer.py

Requirements:
    pip install chromadb ollama requests python-magic pdfplumber pillow
"""

import os
import json
import time
import requests
import traceback
import pdfplumber
from PIL import Image

# =====================
# CONFIGURATION
# =====================

DATA_DIR = "M:\\Documents\\2026"  # Folder containing your documents to index
DB_PATH = "./chroma_db"           # Where the vector database will be stored persistently
EMBEDDING_MODEL = "nomic-embed-text"  # Text embedding model from Ollama
LLM_MODEL = "qwen3.5:9b"          # Primary LLM for generating responses


# =====================
# OLLAMA API HELPER FUNCTIONS
# =====================

def call_ollama_api(endpoint: str, **kwargs) -> dict:
    """
    Make an HTTP request to the Ollama API.

    Args:
        endpoint: The API endpoint to call (e.g., 'embeddings', 'generate', 'embed')
        **kwargs: Keyword arguments passed to the API request

    Returns:
        dict: JSON response from the Ollama API

    Raises:
        requests.HTTPError: If the API request fails with a non-2xx status code
        requests.ConnectionError: If unable to connect to Ollama server
    """
    url = f"http://localhost:11434/api/{endpoint}"
    
    # Add streaming=false to get full response at once
    kwargs["stream"] = False
    
    response = requests.post(url, json=kwargs, timeout=120)

    # Debug: print raw response for troubleshooting on any non-2xx status or error
    if response.status_code != 200:
        print(f"\n=== RAW API RESPONSE (Status {response.status_code}) ===")
        print(f"Response Text:\n{response.text}")
        try:
            print(f"Response JSON:\n{response.json()}")
        except Exception:
            print("Could not parse as JSON")
        print(f"=== ============================\n")
        raise requests.HTTPError(f"HTTP {response.status_code}: {response.text}")

    # Strip common Ollama prefix lines (DEBUG:, INFO:, etc.) before parsing JSON
    response_text = response.text.strip()
    lines = response_text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip common prefix lines that Ollama adds before JSON
        if line.startswith(('DEBUG:', 'INFO:', 'WARNING:', 'stderr:')):
            continue
        cleaned_lines.append(line)
    
    cleaned_response = '\n'.join(cleaned_lines).strip()

    # Try to parse cleaned response as JSON
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Could not parse Ollama response as JSON: {response_text[:200]}...")
        raise


def get_ollama_embedding(text: str) -> list[float]:
    """
    Generate a text embedding using the configured Ollama embedding model.

    Args:
        text: The text string to generate an embedding for

    Returns:
        list[float]: The embedding vector as a list of floats

    Note:
        The text should be reasonably short (recommended < 8000 tokens)
        as Ollama has context limits.
    """
    if not text or len(text.strip()) == 0:
        return [0.0] * 512

    try:
        response = call_ollama_api("embeddings", model=EMBEDDING_MODEL, prompt=text)
        return response.get("embedding", [])
    except requests.exceptions.JSONDecodeError as e:
        # JSON decode error - likely malformed JSON from Ollama
        print(f"\n[WARNING] JSON decode error while getting embedding for: {text[:50]}...")
        print(f"[DEBUG] Raw response:\n{str(e)}")
        return [0.0] * 512
    except Exception as e:
        print(f"\n[ERROR] Error getting embedding: {str(e)}")
        return [0.0] * 512


def get_ollama_response(prompt: str) -> str:
    """
    Generate a text response using the configured Ollama LLM.

    Args:
        prompt: The prompt text to send to the LLM

    Returns:
        str: The generated response from the LLM

    Note:
        The temperature is set to 0.7 for balanced creativity and consistency.
    """
    try:
        response = call_ollama_api(
            "generate",
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        return response.get("response", "")
    except requests.exceptions.JSONDecodeError as e:
        # JSON decode error - likely malformed JSON from Ollama
        print(f"\n[WARNING] JSON decode error from Ollama:")
        print(f"[DEBUG] Error: {str(e)}")
        print(f"[DEBUG] This often happens when Ollama outputs INFO lines before the JSON")
        print(f"[DEBUG] Try reloading the model in Ollama or check the model is ready.")
        print(f"[DEBUG] Raw error traceback:\n{traceback.format_exc()}")
        return "I encountered an error processing your request. Please try again in a moment."
    except Exception as e:
        print(f"\n[ERROR] Error getting response: {str(e)}")
        print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
        return f"I'm sorry, I encountered an error: {str(e)}"


# =====================
# CHROMADB COLLECTION MANAGEMENT
# =====================

class OllamaEmbeddingFunction:
    """
    ChromaDB-compatible embedding function wrapper for Ollama.
    
    This class provides the required methods for ChromaDB integration.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding function."""
        self.model_name = model_name

    def __call__(self, input) -> list[list[float]]:
        """
        Generate embeddings for a list of text inputs.

        Args:
            input: A list of text strings to embed (ChromaDB passes a list)

        Returns:
            A list of embeddings (each embedding is a list of floats)
        """
        # ChromaDB passes input as a list of strings
        input_list = input if isinstance(input, list) else [input]
        
        embeddings = []
        for text in input_list:
            if text is not None and len(text) > 0:
                embedding = get_ollama_embedding(text)
                embeddings.append(embedding)
            else:
                embeddings.append([0.0] * 512)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text: The query text to embed

        Returns:
            The embedding vector as a list of floats
        """
        return get_ollama_embedding(text)
    
    def name(self) -> str:
        """Return the name of this embedding function for ChromaDB."""
        return "ollama-" + self.model_name
    
    def get(self) -> list[float]:
        """Generate embedding for a single document (ChromaDB compatibility)."""
        if not self.model_name:
            return [0.0] * 512
        return get_ollama_embedding(self.model_name)


# Create a global embedding function instance
emb_fn = OllamaEmbeddingFunction()


def get_text_collection(client):
    """Get or create the text documents collection using Ollama embeddings."""
    import chromadb

    # Create the collection with Ollama embedding function
    collection = client.get_or_create_collection(
        name="text_docs",
        embedding_function=emb_fn
    )

    return collection


# =====================
# FILE PROCESSING HANDLERS
# =====================

def process_text_file(filepath: str) -> str:
    """Extract raw text content from a plain text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def process_pdf_file(filepath: str) -> str:
    """Extract text content from a PDF document."""
    text_chunks = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    text_chunks.append(text)
        return "\n\n".join(text_chunks)
    except Exception as e:
        print(f"Error reading PDF {filepath}: {str(e)}")
        return ""


def extract_image_description(image_path: str) -> str:
    """Extract metadata and description from an image file for indexing."""
    try:
        with Image.open(image_path) as img:
            desc = f"Image: {img.size[0]}x{img.size[1]} pixels, {img.mode} mode, {img.format or 'unknown'} format."
            try:
                img.load()
                exif = img._getexif()
                if exif:
                    desc += f" Camera: {exif.get(0x010F, 'Unknown')}, Date: {exif.get(0x0132, 'Unknown')}"
            except Exception:
                pass
            return desc
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error reading image {image_path}: {type(e).__name__}"


FILE_HANDLERS = {
    ".txt": process_text_file,
    ".pdf": process_pdf_file,
    ".png": lambda p: extract_image_description(p),
    ".jpg": lambda p: extract_image_description(p),
    ".jpeg": lambda p: extract_image_description(p),
    ".gif": lambda p: extract_image_description(p),
    ".bmp": lambda p: extract_image_description(p),
    ".webp": lambda p: extract_image_description(p),
}


# =====================
# INDEXING
# =====================

def index_documents(collection, data_dir: str) -> int:
    """Index all supported document files from a directory recursively."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir}. Please add some documents and run again.")
        return 0

    print(f"Indexing documents from {data_dir} recursively. ..")

    indexed_count = 0
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            handler = FILE_HANDLERS.get(ext)

            if handler is None:
                print(f"Skipping {filename}: unsupported file type")
                continue

            try:
                content = handler(filepath)
                content = content.strip()
                if not content:
                    print(f"Skipping {filename}: empty content")
                    continue

                # Add document with Ollama embeddings
                embedding = emb_fn([content])[0]
                
                collection.add(
                    documents=[content],
                    ids=[f"{ext}_{filename}"],
                    embeddings=[embedding],
                    metadatas=[
                        {"source": filename, "type": ext, "filepath": filepath}
                    ]
                )
                print(f"Indexed: {filename}")
                indexed_count += 1
            except Exception as e:
                print(f"Error indexing {filename}: {str(e)}")
                continue

    print(f"Total files indexed: {indexed_count}")
    return indexed_count


# =====================
# QUERY
# =====================

def query_documents(query_text: str, collection, n_results: int = 2) -> str:
    """
    Search indexed documents and generate an answer using the LLM.

    Args:
        query_text: The user's natural language query
        collection: The ChromaDB collection to search
        n_results: Number of top results to retrieve

    Returns:
        str: The LLM's generated answer based on retrieved documents
    """
    # Check if collection is empty
    count = collection.count()
    if count == 0:
        return "No documents have been indexed yet. Please add some files to your data directory first."

    # Generate embedding for the query text using Ollama
    query_embedding = emb_fn.embed_query(query_text)

    # Search for the most relevant snippets
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
    except Exception as e:
        print(f"Query error: {e}")
        return "Error performing search. Please try again."

    # Build context from results
    context_parts = []
    if results.get('documents'):
        for doc_group in results['documents']:
            for doc in doc_group:
                if doc:
                    context_parts.append(doc)

    context = "\n\n".join(context_parts)

    # Construct the prompt for the LLM
    if context:
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query_text}
Answer:"""
    else:
        prompt = f"""I searched my indexed documents but found no relevant information matching your question:
"{query_text}"

Please let me know if you'd like to try a different query or if you should tell me you don't know the answer.
Answer:"""

    try:
        response = get_ollama_response(prompt)
        return response
    except Exception as e:
        # Print detailed error info
        print(f"\n=== RAW LLM RESPONSE ERROR ===")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return f"Sorry, an error occurred: {str(e)}"


# =====================
# INTERACTIVE INTERFACE
# =====================

def run_interactive_query(collection) -> None:
    """Run an interactive query loop."""
    while True:
        user_query = input("\nAsk a question about your docs (or 'exit' to quit): ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break

        answer = query_documents(user_query, collection)
        print(f"\nAI: {answer}")


# =====================
# MAIN ENTRY POINT
# =====================

if __name__ == "__main__":
    print("=" * 60)
    print("       SEMANTIC DOCUMENT INDEXER".center(40))
    print("=" * 60)
    print(f"Supported formats: .txt, .pdf, .png, .jpg, .jpeg, .gif, .bmp, .webp")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"LLM model: {LLM_MODEL}")
    print("=" * 60)

    # Initialize ChromaDB client
    import chromadb
    client = chromadb.PersistentClient(path=DB_PATH)

    # Initialize collection
    text_collection = get_text_collection(client)

    print("\n" + "=" * 60)
    print("INDEXING PHASE".center(40))
    print("=" * 60)
    index_count = index_documents(text_collection, DATA_DIR)

    # Step 2: Interactive Query Loop
    print("\n" + "=" * 60)
    print("QUERY INTERFACE".center(40))
    print("=" * 60)
    run_interactive_query(text_collection)