"""File handlers for different document types."""

import os
import json
import pandas as pd
import hashlib
from io import BytesIO

from pypdf import PdfReader
from docx import Document
from PIL import Image


class BaseHandler:
    """Base class for file handlers."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load_document(self) -> str:
        """
        Load the document content.
        
        Returns:
            str: Document text content
        """
        raise NotImplementedError
    
    def get_filename(self) -> str:
        """
        Get the filename.
        
        Returns:
            str: Filename
        """
        return os.path.basename(self.filepath)


class TextHandler(BaseHandler):
    """Handler for text files (.txt, .md)."""
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
    
    def load_document(self) -> str:
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()


class MarkdownHandler(TextHandler):
    """Handler for Markdown files (.md)."""
    
    def load_document(self) -> str:
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()


class CSVHandler(BaseHandler):
    """Handler for CSV files."""
    
    def load_document(self) -> str:
        df = pd.read_csv(self.filepath)
        return df.to_string()


class JSONHandler(BaseHandler):
    """Handler for JSON files."""
    
    def load_document(self) -> str:
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f), indent=2)


class PythonHandler(BaseHandler):
    """Handler for Python files."""
    
    def load_document(self) -> str:
        with open(self.filepath, 'r', encoding='utf-8') as f:
            return f.read()


class PDFHandler(BaseHandler):
    """Handler for PDF files."""
    
    def load_document(self) -> str:
        try:
            reader = PdfReader(self.filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"


class WordHandler(BaseHandler):
    """Handler for Word documents (.docx)."""
    
    def load_document(self) -> str:
        doc = Document(self.filepath)
        return "\n".join([para.text for para in doc.paragraphs])


class PNGHandler(BaseHandler):
    """Handler for PNG images."""
    
    def load_document(self) -> str:
        try:
            with Image.open(self.filepath) as img:
                # Get image dimensions and format
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                return f"Image: {format_name} ({width}x{height}, mode={mode})"
        except Exception as e:
            return f"Error reading image: {e}"


class JPGHandler(BaseHandler):
    """Handler for JPG images."""
    
    def load_document(self) -> str:
        try:
            with Image.open(self.filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                return f"Image: {format_name} ({width}x{height}, mode={mode})"
        except Exception as e:
            return f"Error reading image: {e}"


class JPEGHandler(JPGHandler):
    """Handler for JPEG images."""
    pass


class GIFHandler(BaseHandler):
    """Handler for GIF images."""
    
    def load_document(self) -> str:
        try:
            with Image.open(self.filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                n_frames = getattr(img, 'n_frames', 1)
                
                return f"Image: {format_name} ({width}x{height}, mode={mode}, frames={n_frames})"
        except Exception as e:
            return f"Error reading image: {e}"


class BMPHandler(BaseHandler):
    """Handler for BMP images."""
    
    def load_document(self) -> str:
        try:
            with Image.open(self.filepath) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                return f"Image: {format_name} ({width}x{height}, mode={mode})"
        except Exception as e:
            return f"Error reading image: {e}"


def get_handler(filepath: str) -> BaseHandler:
    """
    Get the appropriate handler for a file based on its extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        BaseHandler: Handler instance for the file type
    """
    _, ext = os.path.splitext(filepath.lower())
    
    handlers = {
        '.txt': TextHandler,
        '.md': MarkdownHandler,
        '.csv': CSVHandler,
        '.json': JSONHandler,
        '.py': PythonHandler,
        '.pdf': PDFHandler,
        '.docx': WordHandler,
        '.png': PNGHandler,
        '.jpg': JPGHandler,
        '.jpeg': JPEGHandler,
        '.gif': GIFHandler,
        '.bmp': BMPHandler,
    }
    
    handler_class = handlers.get(ext)
    
    if handler_class is None:
        raise ValueError(f"No handler available for file extension: {ext}")
    
    return handler_class(filepath)