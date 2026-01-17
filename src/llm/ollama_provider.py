"""
Ollama LLM provider implementation using LangChain.

Supports local models like Llama 3, Mistral, Gemma, and others via Ollama.
"""

import time
import logging
from typing import Optional

import openlit

from .base import BaseLLMProvider, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False
    logger.warning("langchain-ollama not installed. Ollama provider unavailable.")


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider using LangChain.

    Supports local models running via Ollama server.
    No API key required - runs locally.
    """

    PROVIDER_NAME = "ollama"

    DEFAULT_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "mixtral",
        "gemma2",
        "phi3",
        "qwen2.5",
        "codellama",
    ]

    # No API key needed for Ollama
    API_KEY_ENV_VAR = ""

    # Environment variable for Ollama base URL
    BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider with optional base URL from environment."""
        # Set base URL from config or environment
        import os
        if config.base_url is None:
            config.base_url = os.getenv(self.BASE_URL_ENV_VAR, self.DEFAULT_BASE_URL)
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the LangChain Ollama client."""
        if not LANGCHAIN_OLLAMA_AVAILABLE:
            logger.error("langchain-ollama package not installed")
            self._client = None
            return

        try:
            self._client = ChatOllama(
                model=self.model,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                base_url=self.base_url,
                top_p=self.top_p
            )
            logger.info(f"Ollama provider initialized with model: {self.model} at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self._client = None

    def is_available(self) -> bool:
        """Check if Ollama provider is available."""
        if not LANGCHAIN_OLLAMA_AVAILABLE or self._client is None:
            return False

        # Try to check if Ollama server is running
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """List models available on the Ollama server."""
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
        return []

    @openlit.trace
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Ollama.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text and token usage
        """
        if not LANGCHAIN_OLLAMA_AVAILABLE:
            raise RuntimeError("langchain-ollama package not installed")

        # Build messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Override defaults with kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Create client with overrides if needed
        client = self._client
        if client is None or temperature != self.temperature or max_tokens != self.max_tokens:
            client = ChatOllama(
                model=self.model,
                temperature=temperature,
                num_predict=max_tokens,
                base_url=self.base_url,
            )

        # Generate
        start_time = time.time()
        try:
            result = client.invoke(messages)
            generation_time = (time.time() - start_time) * 1000

            # Ollama provides token counts in response metadata
            # The exact format depends on the langchain-ollama version
            usage = {}
            if hasattr(result, "response_metadata"):
                usage = result.response_metadata

            # Try to extract token counts (may not always be available)
            prompt_tokens = usage.get("prompt_eval_count", 0)
            completion_tokens = usage.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Estimate tokens if not provided
            if total_tokens == 0:
                # Rough estimate: ~4 characters per token
                prompt_tokens = len(prompt + (system_prompt or "")) // 4
                completion_tokens = len(result.content) // 4
                total_tokens = prompt_tokens + completion_tokens

            response = LLMResponse(
                text=result.content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason=usage.get("done_reason", "stop"),
                raw_response=result
            )

            self._update_stats(response)
            return response

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise


def create_ollama_provider(
    model: str = "mistral",
    temperature: float = 0.8,
    max_tokens: int = 500,
    base_url: Optional[str] = None
) -> OllamaProvider:
    """
    Factory function to create an Ollama provider.

    Args:
        model: Model name (default: mistral)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        base_url: Optional Ollama server URL (uses env var or default if not provided)

    Returns:
        Configured OllamaProvider instance
    """
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )
    return OllamaProvider(config)
