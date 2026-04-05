"""Ollama API client for text embedding and generation."""

import json
import requests
import traceback

from .config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    API_TIMEOUT
)


class OllamaClient:
    """Client for interacting with the Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = API_TIMEOUT):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def call_api(self, endpoint: str, **kwargs) -> dict:
        """
        Make an HTTP request to the Ollama API.
        
        Args:
            endpoint: The API endpoint to call (e.g., 'embeddings', 'generate')
            **kwargs: Keyword arguments passed to the API request
            
        Returns:
            dict: JSON response from the Ollama API
            
        Raises:
            requests.HTTPError: If the API request fails
            requests.ConnectionError: If unable to connect
        """
        url = f"{self.base_url}/api/{endpoint}"
        
        # Ensure streaming is false for synchronous requests
        kwargs["stream"] = False
        
        response = requests.post(url, json=kwargs, timeout=self.timeout)
        
        if response.status_code != 200:
            # Debug: print raw response for troubleshooting
            print(f"\n=== RAW API RESPONSE (Status {response.status_code}) ===")
            print(f"Response Text:\n{response.text}")
            try:
                print(f"Response JSON:\n{response.json()}")
            except Exception:
                print("Could not parse as JSON")
            print(f"=== ========== ===\n")
            raise requests.HTTPError(f"HTTP {response.status_code}: {response.text}")
        
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
            print(f"Could not parse Ollama response as JSON: {response_text[:200]}...")
            raise
    
    def get_embedding(self, text: str) -> list[float]:
        """
        Generate a text embedding using the configured Ollama embedding model.
        
        Args:
            text: The text string to generate an embedding for
            
        Returns:
            list[float]: The embedding vector as a list of floats
            
        Note:
            The text should be reasonably short (< 8000 tokens)
        """
        if not text or len(text.strip()) == 0:
            return [0.0] * 512  # Return zero vector for empty text
        
        try:
            response = self.call_api(
                "embeddings",
                model=EMBEDDING_MODEL,
                prompt=text
            )
            return response.get("embedding", [])
        except requests.exceptions.JSONDecodeError as e:
            print(f"\n[WARNING] JSON decode error while getting embedding:")
            print(f"[DEBUG] Error: {str(e)}")
            return [0.0] * 512
        except Exception as e:
            print(f"\n[ERROR] Error getting embedding: {str(e)}")
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            return [0.0] * 512
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a text response using the configured Ollama LLM.
        
        Args:
            prompt: The prompt text to send to the LLM
            temperature: Sampling temperature (default: 0.7)
            
        Returns:
            str: The generated response from the LLM
        """
        try:
            response = self.call_api(
                "generate",
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": temperature}
            )
            return response.get("response", "")
        except requests.exceptions.JSONDecodeError as e:
            print(f"\n[WARNING] JSON decode error from Ollama:")
            print(f"[DEBUG] Error: {str(e)}")
            print(f"[DEBUG] Try reloading the model in Ollama or check the model is ready.")
            return "I encountered an error processing your request. Please try again in a moment."
        except Exception as e:
            print(f"\n[ERROR] Error getting response: {str(e)}")
            print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            return f"I'm sorry, I encountered an error: {str(e)}"


def create_client() -> OllamaClient:
    """Create and return an Ollama client instance."""
    return OllamaClient(
        base_url=OLLAMA_BASE_URL,
        timeout=API_TIMEOUT
    )