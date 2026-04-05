"""Main entry point for the Semantic Document Indexer."""

import sys
import uuid
import time
import os

from indexer.config import (
    DATA_DIR,
    DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    DEFAULT_N_RESULTS,
    OLLAMA_BASE_URL,
    API_TIMEOUT,
    compute_file_hash,
    load_cache,
    save_cache
)
from indexer.api_client import OllamaClient, create_client
from indexer.chroma_client import ChromaClient, create_client as create_chroma_client
from indexer.file_handlers import get_handler

__all__ = ['run_indexer']


def get_default_prompt_template() -> str:
    """
    Get a default prompt for generating responses.
    
    Returns:
        str: Default prompt template
    """
    return "You are an AI assistant with access to documents. Please answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"


def generate_embedding(text: str, client: OllamaClient) -> list[float]:
    """
    Generate embedding for a text.
    
    Args:
        text: Text to embed
        client: Ollama client
        
    Returns:
        list[float]: Embedding vector
    """
    return client.get_embedding(text)


def run_indexer() -> None:
    """
    Run the document indexing system.
    
    This is the main entry point for the application.
    """
    print("=" * 60)
    print("Semantic Document Indexer")
    print("=" * 60)
    print()
    
    # Check Ollama availability
    print(f"Checking Ollama availability at {OLLAMA_BASE_URL}...")
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve model list: {response.status_code}")
    except Exception as e:
        print(f"  Cannot connect to Ollama: {e}")
        print()
        print("Attempting to continue without Ollama...")
        print("(Note: Some features may not work without Ollama)")
        print()
        client = None
        print()
    else:
        print("Connected to Ollama")
        print()
        # Check model availability
        models_response = response.json()
        available_models = [m['name'] for m in models_response.get('models', [])]
        print(f"Available models: {', '.join(available_models)}" if available_models else "Available models: None")
        print()
        
        # Check if required models are available
        models_to_check = [EMBEDDING_MODEL, LLM_MODEL]
        missing_models = []
        for model_name in models_to_check:
            # Allow partial matching (e.g., "nomic-embed-text" matches "nomic-embed-text:latest")
            found = any(model_name in avail for avail in available_models)
            if not found:
                missing_models.append(model_name)
        
        if missing_models:
            print(f"WARNING: Following models not found: {', '.join(missing_models)}")
            print(f"  Available models: {', '.join(available_models)}" if available_models else "  Available models: None")
            print("  Using default models or skipping indexing if models are required")
            print()
            # Don't exit, let the application continue with limited functionality
            client = None
        else:
            print(f"Using embedding model: {EMBEDDING_MODEL}")
            print(f"Using LLM model: {LLM_MODEL}")
            # Initialize client if models are available
            try:
                client = create_client()
            except Exception as e:
                print(f"Failed to create client: {e}")
                sys.exit(1)
            print()
        print()
    print()
    # Print models if client is available
    if client and available_models:
        print(f"Using embedding model: {EMBEDDING_MODEL}")
        print(f"Using LLM model: {LLM_MODEL}")
        print()
    print()
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB at {DB_PATH}...")
    try:
        chroma_client = create_chroma_client(DB_PATH)
        print(f"ChromaDB initialized")
        print(f"Document count: {chroma_client.count()}")
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        sys.exit(1)
    print()
    
    # Check for documents to index
    supported_extensions = {'.txt', '.md', '.csv', '.json', '.py', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    files = []
    
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filename.lower())
                if ext in supported_extensions:
                    files.append(filepath)
    else:
        print(f"[WARN] Data directory not found: {DATA_DIR}")
    
    if not files:
        print(f"[INFO] No supported files found in {DATA_DIR}")
        print(f"    Supported types: .txt, .md, .csv, .json, .py, .pdf, .docx, .png, .jpg, .gif")
    else:
        print(f"Found {len(files)} file(s) to index")
        for filepath in sorted(files):
            filename = filepath.split('/')[-1]
            print(f"    - {filename}")
    print()
    
    # Load cache
    cache = load_cache()
    existing_ids = chroma_client.get_ids()
    print("Checking documents...")
    
    files_to_index = []
    skipped_by_cache = []
    skipped_no_id = []
    
    for filepath in files:
        filename = filepath.split('/')[-1]
        doc_id = str(uuid.uuid4())
        
        # Compute file hash
        file_hash = compute_file_hash(filepath)
        
        # Check if already in cache
        if file_hash in cache:
            if doc_id in existing_ids:
                print(f"    - {filename} (unchanged, skipped)")
                skipped_by_cache.append(filename)
                continue
            else:
                # File changed, need to re-index
                print(f"    - {filename} (changed, re-indexing)")
        else:
            # New file
            print(f"    - {filename} (new file)")
        
        files_to_index.append({
            'filepath': filepath,
            'filename': filename,
            'doc_id': doc_id,
            'file_hash': file_hash
        })
    
    if files_to_index:
        print()
        print("Indexing documents...")
        
        for item in files_to_index:
            handler = get_handler(item['filepath'])
            
            try:
                # Load document
                document = handler.load_document()
                metadata = {'filename': item['filename']}
                
                # Generate embedding for the document
                embedding = generate_embedding(document, client)
                
                # Add to ChromaDB with embedding
                chroma_client.add_documents(
                    documents=[document],
                    ids=[item['doc_id']],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                
                print(f"   Indexed: {item['filename']}")
                
                # Update cache
                cache[item['file_hash']] = {
                    'doc_id': item['doc_id'],
                    'filepath': item['filepath']
                }
            
            except Exception as e:
                print(f"   Could not index {item['filename']}: {e}")
        
        print()
        
        # Save cache
        save_cache(cache)
        
        # Print cache stats
        print(f"Cache updated. {len(cache)} document(s) cached.")
        if skipped_by_cache:
            print(f"    Skipped unchanged: {', '.join(skipped_by_cache)}")
        if skipped_no_id:
            print(f"    Skipped (no ID): {', '.join(skipped_no_id)}")
    else:
        print("All documents already indexed")
    print()
    
    # Print cache info
    cache_file = DATA_DIR.parent / 'chroma_db' / 'document_cache.json'
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = f.read()
            print(f"Current cache size: {len(cache_data)} bytes")
    print()
    
    # Run interactive query loop
    print("Enter your question or 'exit' to quit")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ('exit', 'quit', 'q'):
                print()
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate query embedding
            print(f"Processing: {user_input}...")
            
            try:
                query_embedding = client.get_embedding(user_input)
                print(f"   Embedding generated ({len(query_embedding)} dimensions)")
            except Exception as e:
                print(f"   Error generating embedding: {e}")
                continue
            
            # Query ChromaDB
            print(f"Searching vector database...")
            
            try:
                results = chroma_client.query(
                    query_embeddings=[query_embedding],
                    n_results=DEFAULT_N_RESULTS
                )
                
                if not results or not results.get('documents'):
                    print("   No relevant documents found.")
                    continue
                
                documents = results['documents'][0]
                metadatas = results.get('metadatas', [[]])[0]
                
                # Collect context
                context_parts = []
                for doc, metadata in zip(documents, metadatas):
                    filename = metadata.get('filename', 'unknown')
                    context_parts.append(f"=== Document: {filename} ===\n{doc}")
                
                context = '\n\n---\n\n'.join(context_parts)
                
                # Generate response
                print(f"   Found {len(documents)} relevant documents")
                print()
                
                prompt = get_default_prompt_template().format(context=context, question=user_input)
                print(f"Generating response...")
                
                response = client.generate_response(prompt)
                
                # Display response
                print("-" * 60)
                print(f"Question: {user_input}")
                print("-" * 60)
                print(response)
                print("-" * 60)
                
            except Exception as e:
                print(f"   Error querying database or generating response: {e}")
            
            print()
        
        except KeyboardInterrupt:
            print()
            print("Goodbye!")
            break
        except EOFError:
            print()
            print("Goodbye!")
            break


if __name__ == '__main__':
    run_indexer()