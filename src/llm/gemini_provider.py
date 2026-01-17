"""
Google Gemini LLM provider implementation using LangChain.

Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and other Google AI models.
"""

import time
import logging
from typing import Optional

import openlit

from .base import BaseLLMProvider, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_AVAILABLE = False
    logger.warning("langchain-google-genai not installed. Gemini provider unavailable.")


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider using LangChain.

    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0, and other models.
    """

    PROVIDER_NAME = "gemini"

    DEFAULT_MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-pro",
    ]

    API_KEY_ENV_VAR = "GOOGLE_API_KEY"

    def _initialize_client(self) -> None:
        """Initialize the LangChain Google Generative AI client."""
        if not LANGCHAIN_GOOGLE_AVAILABLE:
            logger.error("langchain-google-genai package not installed")
            self._client = None
            return

        if not self.api_key:
            logger.warning(f"No API key found for Gemini. Set {self.API_KEY_ENV_VAR} environment variable.")
            self._client = None
            return

        try:
            self._client = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=self.api_key,
                top_p=self.top_p
            )
            logger.info(f"Gemini provider initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self._client = None

    def is_available(self) -> bool:
        """Check if Gemini provider is available."""
        return (
            LANGCHAIN_GOOGLE_AVAILABLE
            and self._client is not None
            and self.api_key is not None
        )

    @openlit.trace
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Google Gemini.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text and token usage
        """
        if not self.is_available():
            raise RuntimeError("Gemini provider is not available. Check API key and installation.")

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
        if temperature != self.temperature or max_tokens != self.max_tokens:
            client = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=self.api_key,
            )

        # Generate
        start_time = time.time()
        try:
            result = client.invoke(messages)
            generation_time = (time.time() - start_time) * 1000

            # Extract token usage from response metadata
            # Gemini provides usage_metadata
            usage_metadata = result.response_metadata.get("usage_metadata", {})
            prompt_tokens = usage_metadata.get("prompt_token_count", 0)
            completion_tokens = usage_metadata.get("candidates_token_count", 0)
            total_tokens = usage_metadata.get("total_token_count", prompt_tokens + completion_tokens)

            response = LLMResponse(
                text=result.content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason=result.response_metadata.get("finish_reason", "STOP"),
                raw_response=result
            )

            self._update_stats(response)
            return response

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise


def create_gemini_provider(
    model: str = "gemini-1.5-flash",
    temperature: float = 0.8,
    max_tokens: int = 500,
    api_key: Optional[str] = None
) -> GeminiProvider:
    """
    Factory function to create a Gemini provider.

    Args:
        model: Model name (default: gemini-1.5-flash)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Configured GeminiProvider instance
    """
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    return GeminiProvider(config)
