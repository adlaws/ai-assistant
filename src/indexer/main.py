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
    API_TIMEOUT
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
    
    # Initialize clients
    print(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")
    try:
        client = create_client()
        print(f"Connected to Ollama")
        print(f"Using embedding model: {EMBEDDING_MODEL}")
        print(f"Using LLM model: {LLM_MODEL}")
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        sys.exit(1)
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
    
    # Check if documents need indexing
    existing_ids = chroma_client.get_ids()
    print("Checking documents...")
    
    files_to_index = []
    for filepath in files:
        filename = filepath.split('/')[-1]
        doc_id = str(uuid.uuid4())
        
        # Check if already indexed
        if doc_id in existing_ids:
            print(f"    - {filename} (already indexed)")
            continue
        
        files_to_index.append({
            'filepath': filepath,
            'filename': filename,
            'doc_id': doc_id
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
                
            except Exception as e:
                print(f"   Could not index {item['filename']}: {e}")
        
        print()
        print(f"Total documents in database: {chroma_client.count()}")
    else:
        print("All documents already indexed")
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