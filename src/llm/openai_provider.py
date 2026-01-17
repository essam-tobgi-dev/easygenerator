"""
OpenAI LLM provider implementation using LangChain.

Supports GPT-4, GPT-4o, GPT-4o-mini, and other OpenAI models.
"""

import time
import logging
from typing import Optional

import openlit

from .base import BaseLLMProvider, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    logger.warning("langchain-openai not installed. OpenAI provider unavailable.")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider using LangChain.

    Supports all OpenAI chat models including GPT-4o, GPT-4o-mini, GPT-4, etc.
    """

    PROVIDER_NAME = "openai"

    DEFAULT_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]

    API_KEY_ENV_VAR = "OPENAI_API_KEY"

    def _initialize_client(self) -> None:
        """Initialize the LangChain OpenAI client."""
        if not LANGCHAIN_OPENAI_AVAILABLE:
            logger.error("langchain-openai package not installed")
            self._client = None
            return

        if not self.api_key:
            logger.warning(f"No API key found for OpenAI. Set {self.API_KEY_ENV_VAR} environment variable.")
            self._client = None
            return

        try:
            self._client = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                model_kwargs={"top_p": self.top_p}
            )
            logger.info(f"OpenAI provider initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._client = None

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return (
            LANGCHAIN_OPENAI_AVAILABLE
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
        Generate text using OpenAI.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text and token usage
        """
        if not self.is_available():
            raise RuntimeError("OpenAI provider is not available. Check API key and installation.")

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
            client = ChatOpenAI(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
            )

        # Generate
        start_time = time.time()
        try:
            result = client.invoke(messages)
            generation_time = (time.time() - start_time) * 1000

            # Extract token usage from response metadata
            usage = result.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            response = LLMResponse(
                text=result.content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason=result.response_metadata.get("finish_reason", "stop"),
                raw_response=result
            )

            self._update_stats(response)
            return response

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise


def create_openai_provider(
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_tokens: int = 500,
    api_key: Optional[str] = None
) -> OpenAIProvider:
    """
    Factory function to create an OpenAI provider.

    Args:
        model: Model name (default: gpt-4o-mini)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Configured OpenAIProvider instance
    """
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    return OpenAIProvider(config)
