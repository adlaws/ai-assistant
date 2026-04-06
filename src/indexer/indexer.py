"""Module for indexing text files with Ollama embeddings."""

from __future__ import annotations

import os

from .api_client import create_client
from .config import DATA_DIR, DB_PATH


def process_text_file(filepath: str) -> str:
    """Process text file and return content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def process_pdf_file(filepath: str) -> str:
    """Process PDF file and extract text."""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            text_chunks = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    text_chunks.append(text)
        return "\n\n".join(text_chunks)
    except ImportError:
        return ""
    except Exception as e:
        print(f"PDF read error: {str(e)}")
        return ""


def process_markdown_file(filepath: str) -> str:
    """Process markdown file and return text content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        import markdown
        html = markdown.markdown(markdown_text)
        import re
        text = re.sub('<.*?>', '', html)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    except Exception as e:
        print(f"Markdown read error: {str(e)}")
        return ""


def process_word_document(filepath: str) -> str:
    """Process Word document and extract text."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)
    except ImportError:
        return ""
    except Exception as e:
        print(f"Word doc read error: {str(e)}")
        return ""


def extract_image_description(image_path: str) -> str:
    """Extract description from image file."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            desc = f"Image: {img.size[0]}x{img.size[1]} pixels, {img.format or 'unknown'}."
            return desc
    except Exception as e:
        return f"Error reading image: {str(e)}"


def get_ollama_embedding(text: str) -> list:
    """
    Generate embedding for text using Ollama.

    Args:
        text: Text to embed

    Returns:
        list[float]: Embedding vector
    """
    ollama_client = create_client()
    return ollama_client.get_embedding(text)


def get_file_handler(ext: str) -> callable:
    """
    Get file handler for a given extension.

    Args:
        ext: File extension

    Returns:
        callable: Handler function or None
    """
    handlers = {
        '.txt': process_text_file,
        '.pdf': process_pdf_file,
        '.md': process_markdown_file,
        '.docx': process_word_document,
    }
    return handlers.get(ext.lower())


def index_file(filepath: str, embedding_fn: callable = None) -> tuple[str, list[float]]:
    """
    Read text file and generate embedding.

    Args:
        filepath: Path to file to index
        embedding_fn: Function to generate embeddings for text (optional, auto-creates if None)

    Returns:
        tuple: (content, embedding) or (None, None) on error
    """
    handler = get_file_handler(os.path.splitext(filepath)[1])
    if handler:
        try:
            content = handler(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None, None
    else:
        print(f"Unsupported file extension: {os.path.splitext(filepath)[1]}")
        return None, None

    if embedding_fn is None:
        embedding_fn = get_ollama_embedding

    return content, embedding_fn(content)


def _chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Chunk text into overlapping segments."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def index_documents(client, data_dir: str) -> int:
    """
    Index all supported documents from a directory.

    Args:
        client: ChromaDB client instance
        data_dir: Directory containing documents to index

    Returns:
        int: Number of documents indexed
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir}. Add documents and run again.")
        return 0

    print(f"Indexing from {data_dir}...")

    indexed_count = 0

    for root, _, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filepath)[1].lower()

            content, embedding = index_file(filepath)

            if content is not None and embedding is not None:
                client.collection.add(
                    documents=[content],
                    ids=[f"{ext}_{filename}"],
                    embeddings=[embedding],
                    metadatas=[{"source": filename, "type": ext, "filepath": filepath}]
                )
                print(f"Indexed: {filename}")
                indexed_count += 1

    print(f"Total indexed: {indexed_count}")
    return indexed_count