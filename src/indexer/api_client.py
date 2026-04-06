"""Ollama API client for text embedding and generation."""

from __future__ import annotations

import logging
import json
import requests
from typing import Any

from src.indexer.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    API_TIMEOUT
)
from src.indexer.exceptions import OllamaError, EmbeddingError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with the Ollama API."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = API_TIMEOUT) -> None:
        """Initialize the Ollama client.

        :param base_url: Base URL for the Ollama API
        :param timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def call_api(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Make an HTTP request to the Ollama API.

        :param endpoint: The API endpoint to call (e.g., 'embeddings', 'generate')
        :param kwargs: Keyword arguments passed to the API request
        :return: JSON response from the Ollama API
        :raises OllamaError: If the API request fails
        :raises requests.ConnectionError: If unable to connect
        """
        url = f"{self.base_url}/api/{endpoint}"

        # Ensure streaming is false for synchronous requests
        kwargs["stream"] = False

        try:
            response = requests.post(url, json=kwargs, timeout=self.timeout)

            if response.status_code != 200:
                # Log raw response for troubleshooting
                logger.debug("RAW API RESPONSE (Status %d): %s", response.status_code, response.text[:500])
                raise OllamaError(f"HTTP {response.status_code}: {response.text}")

            # Strip common Ollama prefix lines (DEBUG:, INFO:, etc.)
            response_text = response.text.strip()
            lines = response_text.split('\n')
            cleaned_lines = [
                line for line in lines
                if not line.startswith(('DEBUG:', 'INFO:', 'WARNING:', 'stderr:'))
            ]
            cleaned_response = '\n'.join(cleaned_lines).strip()

            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error("Could not parse Ollama response as JSON: %s", response_text[:200])
                raise OllamaError(f"Invalid JSON response from Ollama: {e}") from e

        except requests.ConnectionError as e:
            raise OllamaError(f"Cannot connect to Ollama at {url}: {e}") from e
        except requests.Timeout as e:
            raise OllamaError(f"Timeout connecting to Ollama: {e}") from e

    def get_embedding(self, text: str) -> list[float]:
        """Generate a text embedding using the configured Ollama embedding model.

        :param text: The text string to generate an embedding for
        :return: The embedding vector as a list of floats
        :raises EmbeddingError: If embedding generation fails
        """
        if not text or len(text.strip()) == 0:
            raise EmbeddingError("Cannot generate embedding for empty text")

        try:
            response = self.call_api(
                "embeddings",
                model=EMBEDDING_MODEL,
                prompt=text
            )
            embedding = response.get("embedding", [])
            if not embedding:
                raise EmbeddingError("Ollama returned empty embedding")
            return embedding
        except OllamaError as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a text response using the configured Ollama LLM.

        :param prompt: The prompt text to send to the LLM
        :param temperature: Sampling temperature (default: 0.7)
        :return: The generated response from the LLM
        :raises OllamaError: If response generation fails
        """
        try:
            response = self.call_api(
                "generate",
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": temperature}
            )
            generated_text = response.get("response", "")
            if not generated_text:
                raise OllamaError("Ollama returned empty response")
            return generated_text
        except OllamaError as e:
            raise OllamaError(f"Failed to generate response: {e}") from e


def create_client() -> OllamaClient:
    """Create and return an Ollama client instance.

    :return: Instance of the Ollama API client
    """
    return OllamaClient(
        base_url=OLLAMA_BASE_URL,
        timeout=API_TIMEOUT
    )
