"""
Base class for LLM providers.

Defines the abstract interface that all LLM provider implementations must follow.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.

    Contains the generated text along with metadata about the generation.
    """
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None

    @property
    def cost_estimate(self) -> float:
        """
        Estimate cost based on token usage.

        Note: These are approximate rates and may vary by model.
        """
        # Approximate costs per 1K tokens (varies by model)
        costs = {
            "openai": {"input": 0.0005, "output": 0.0015},
            "anthropic": {"input": 0.003, "output": 0.015},
            "gemini": {"input": 0.00025, "output": 0.0005},
            "ollama": {"input": 0.0, "output": 0.0},  # Local, no API cost
        }

        provider_costs = costs.get(self.provider.lower(), {"input": 0.001, "output": 0.002})

        input_cost = (self.prompt_tokens / 1000) * provider_costs["input"]
        output_cost = (self.completion_tokens / 1000) * provider_costs["output"]

        return input_cost + output_cost


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    model: str
    temperature: float = 0.8
    max_tokens: int = 500
    top_p: float = 1.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: dict = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class
    and implement the required abstract methods.
    """

    # Class-level provider name
    PROVIDER_NAME: str = "base"

    # Default models for this provider
    DEFAULT_MODELS: list[str] = []

    # Environment variable name for API key
    API_KEY_ENV_VAR: str = ""

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider.

        Args:
            config: LLMConfig with model settings
        """
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p

        # Get API key from config or environment
        self.api_key = config.api_key or self._get_api_key_from_env()
        self.base_url = config.base_url

        # Statistics tracking
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_time_ms = 0.0

        # Initialize the underlying client
        self._client = None
        self._initialize_client()

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        if self.API_KEY_ENV_VAR:
            return os.getenv(self.API_KEY_ENV_VAR)
        return None

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the underlying LangChain client.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the LLM.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt for context
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and configured.

        Returns:
            True if the provider can be used, False otherwise
        """
        pass

    def get_stats(self) -> dict:
        """Get usage statistics for this provider."""
        return {
            "provider": self.PROVIDER_NAME,
            "model": self.model,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / self._total_requests if self._total_requests > 0 else 0,
            "avg_tokens": self._total_tokens / self._total_requests if self._total_requests > 0 else 0,
        }

    def _update_stats(self, response: LLMResponse) -> None:
        """Update internal statistics from a response."""
        self._total_requests += 1
        self._total_tokens += response.total_tokens
        self._total_cost += response.cost_estimate
        self._total_time_ms += response.generation_time_ms

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_time_ms = 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
