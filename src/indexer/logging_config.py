"""Logging configuration for the Semantic Document Indexer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from src.indexer.config import DB_PATH


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration.

    :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_file: Optional log file path
    :param format_string: Optional custom format string
    :return: Logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get or create logger
    logger = logging.getLogger("semantic_indexer")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logging()
