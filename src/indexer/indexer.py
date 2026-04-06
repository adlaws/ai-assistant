"""Module for indexing text files with Ollama embeddings."""

from __future__ import annotations

import logging
import os

from src.indexer.api_client import create_client
from src.indexer.config import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)


def process_text_file(filepath: str) -> str:
    """Process text file and return content.

    :param filepath: Path to the text file
    :return: File content as string
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def process_pdf_file(filepath: str) -> str:
    """Process PDF file and extract text.

    :param filepath: Path to the PDF file
    :return: Extracted text content
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
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
        logger.warning("pdfplumber not installed, skipping PDF processing")
        return ""
    except Exception as e:
        logger.error("PDF read error: %s", str(e))
        return ""


def process_markdown_file(filepath: str) -> str:
    """Process markdown file and return text content.

    :param filepath: Path to the markdown file
    :return: Extracted text content
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
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
        logger.error("Markdown read error: %s", str(e))
        return ""


def process_word_document(filepath: str) -> str:
    """Process Word document and extract text.

    :param filepath: Path to the Word document
    :return: Extracted text content
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)
    except ImportError:
        logger.warning("python-docx not installed, skipping Word document processing")
        return ""
    except Exception as e:
        logger.error("Word doc read error: %s", str(e))
        return ""


def extract_image_description(image_path: str) -> str:
    """Extract description from image file.

    :param image_path: Path to the image file
    :return: Image description string
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            desc = f"Image: {img.size[0]}x{img.size[1]} pixels, {img.format or 'unknown'}."
            return desc
    except Exception as e:
        logger.error("Error reading image: %s", str(e))
        return f"Error reading image: {str(e)}"


def get_ollama_embedding(text: str) -> list[float]:
    """Generate embedding for text using Ollama.

    :param text: Text to embed
    :return: Embedding vector as list of floats
    :raises EmbeddingError: If embedding generation fails
    """
    ollama_client = create_client()
    return ollama_client.get_embedding(text)


def get_file_handler(ext: str) -> callable:
    """Get file handler for a given extension.

    :param ext: File extension (e.g., '.txt', '.pdf')
    :return: Handler function or None if no handler available
    """
    handlers = {
        '.txt': process_text_file,
        '.pdf': process_pdf_file,
        '.md': process_markdown_file,
        '.docx': process_word_document,
    }
    return handlers.get(ext.lower())


def index_file(filepath: str, embedding_fn: callable | None = None) -> tuple[str | None, list[float] | None]:
    """Read text file and generate embedding.

    :param filepath: Path to file to index
    :param embedding_fn: Function to generate embeddings for text (optional, auto-creates if None)
    :return: Tuple of (content, embedding) or (None, None) on error
    :raises FileNotFoundError: If file does not exist
    :raises IOError: If file cannot be read
    """
    handler = get_file_handler(os.path.splitext(filepath)[1])
    if handler:
        try:
            content = handler(filepath)
        except Exception as e:
            logger.error("Error reading %s: %s", filepath, str(e))
            return None, None
    else:
        logger.warning("Unsupported file extension: %s", os.path.splitext(filepath)[1])
        return None, None

    if embedding_fn is None:
        embedding_fn = get_ollama_embedding

    return content, embedding_fn(content)


def _chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Chunk text into overlapping segments.

    :param text: Text to chunk
    :param chunk_size: Size of each chunk in characters (default: 500)
    :param chunk_overlap: Number of overlapping characters between chunks (default: 100)
    :return: List of text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def index_documents(client, data_dir: str) -> int:
    """Index all supported documents from a directory.

    :param client: ChromaDB client instance
    :param data_dir: Directory containing documents to index
    :return: Number of documents indexed
    :raises FileNotFoundError: If data directory does not exist
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info("Created %s. Add documents and run again.", data_dir)
        return 0

    logger.info("Indexing from %s...", data_dir)

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
                logger.info("Indexed: %s", filename)
                indexed_count += 1

    logger.info("Total indexed: %d", indexed_count)
    return indexed_count
