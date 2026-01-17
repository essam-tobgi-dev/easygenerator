"""
Anthropic Claude LLM provider implementation using LangChain.

Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other Claude models.
"""

import time
import logging
from typing import Optional

import openlit

from .base import BaseLLMProvider, LLMResponse, LLMConfig

logger = logging.getLogger(__name__)

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_ANTHROPIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_ANTHROPIC_AVAILABLE = False
    logger.warning("langchain-anthropic not installed. Anthropic provider unavailable.")


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider using LangChain.

    Supports all Claude models including Claude 3.5 Sonnet, Claude 3 Opus, etc.
    """

    PROVIDER_NAME = "anthropic"

    DEFAULT_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

    def _initialize_client(self) -> None:
        """Initialize the LangChain Anthropic client."""
        if not LANGCHAIN_ANTHROPIC_AVAILABLE:
            logger.error("langchain-anthropic package not installed")
            self._client = None
            return

        if not self.api_key:
            logger.warning(f"No API key found for Anthropic. Set {self.API_KEY_ENV_VAR} environment variable.")
            self._client = None
            return

        try:
            self._client = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                top_p=self.top_p
            )
            logger.info(f"Anthropic provider initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self._client = None

    def is_available(self) -> bool:
        """Check if Anthropic provider is available."""
        return (
            LANGCHAIN_ANTHROPIC_AVAILABLE
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
        Generate text using Anthropic Claude.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated text and token usage
        """
        if not self.is_available():
            raise RuntimeError("Anthropic provider is not available. Check API key and installation.")

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
            client = ChatAnthropic(
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
            usage = result.response_metadata.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            response = LLMResponse(
                text=result.content,
                model=self.model,
                provider=self.PROVIDER_NAME,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time_ms=generation_time,
                finish_reason=result.response_metadata.get("stop_reason", "end_turn"),
                raw_response=result
            )

            self._update_stats(response)
            return response

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise


def create_anthropic_provider(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.8,
    max_tokens: int = 500,
    api_key: Optional[str] = None
) -> AnthropicProvider:
    """
    Factory function to create an Anthropic provider.

    Args:
        model: Model name (default: claude-3-5-sonnet-20241022)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Configured AnthropicProvider instance
    """
    config = LLMConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    return AnthropicProvider(config)
