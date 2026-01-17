"""
LLM Provider abstraction layer using LangChain.

Provides a unified interface for multiple LLM providers with
OpenLIT observability for token tracking and monitoring.
"""

from .base import BaseLLMProvider, LLMResponse, LLMConfig
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider


# Comprehensive list of all supported models by provider
SUPPORTED_MODELS = {
    "openai": [
        {"model": "gpt-4o", "description": "Most capable GPT-4 Omni model"},
        {"model": "gpt-4o-mini", "description": "Fast and cost-effective GPT-4 Omni"},
        {"model": "gpt-4-turbo", "description": "GPT-4 Turbo with vision"},
        {"model": "gpt-4", "description": "Standard GPT-4 model"},
        {"model": "gpt-3.5-turbo", "description": "Fast GPT-3.5 model"},
        {"model": "o1", "description": "Reasoning model for complex tasks"},
        {"model": "o1-mini", "description": "Faster reasoning model"},
        {"model": "o1-preview", "description": "Preview of o1 reasoning model"},
    ],
    "anthropic": [
        {"model": "claude-opus-4-20250514", "description": "Most capable Claude model"},
        {"model": "claude-sonnet-4-20250514", "description": "Balanced Claude model"},
        {"model": "claude-3-5-sonnet-20241022", "description": "Claude 3.5 Sonnet"},
        {"model": "claude-3-5-haiku-20241022", "description": "Fast Claude 3.5 Haiku"},
        {"model": "claude-3-opus-20240229", "description": "Claude 3 Opus"},
        {"model": "claude-3-sonnet-20240229", "description": "Claude 3 Sonnet"},
        {"model": "claude-3-haiku-20240307", "description": "Fast Claude 3 Haiku"},
    ],
    "gemini": [
        {"model": "gemini-2.0-flash-exp", "description": "Latest Gemini 2.0 Flash"},
        {"model": "gemini-2.0-flash", "description": "Gemini 2.0 Flash"},
        {"model": "gemini-1.5-pro", "description": "Gemini 1.5 Pro"},
        {"model": "gemini-1.5-flash", "description": "Fast Gemini 1.5 Flash"},
        {"model": "gemini-1.5-flash-8b", "description": "Compact Gemini Flash"},
        {"model": "gemini-pro", "description": "Standard Gemini Pro"},
    ],
    "ollama": [
        {"model": "llama3.2", "description": "Meta Llama 3.2"},
        {"model": "llama3.1", "description": "Meta Llama 3.1"},
        {"model": "llama3", "description": "Meta Llama 3"},
        {"model": "mistral", "description": "Mistral 7B"},
        {"model": "mixtral", "description": "Mixtral 8x7B MoE"},
        {"model": "gemma2", "description": "Google Gemma 2"},
        {"model": "gemma", "description": "Google Gemma"},
        {"model": "phi3", "description": "Microsoft Phi-3"},
        {"model": "phi", "description": "Microsoft Phi-2"},
        {"model": "qwen2.5", "description": "Alibaba Qwen 2.5"},
        {"model": "qwen2", "description": "Alibaba Qwen 2"},
        {"model": "codellama", "description": "Code Llama"},
        {"model": "deepseek-coder", "description": "DeepSeek Coder"},
        {"model": "starcoder2", "description": "StarCoder 2"},
        {"model": "command-r", "description": "Cohere Command-R"},
        {"model": "neural-chat", "description": "Intel Neural Chat"},
        {"model": "vicuna", "description": "Vicuna"},
        {"model": "orca-mini", "description": "Orca Mini"},
    ],
}


def get_all_supported_models() -> list[dict]:
    """
    Get a flat list of all supported models across all providers.

    Returns:
        List of dicts with provider, model, and description
    """
    models = []
    for provider, provider_models in SUPPORTED_MODELS.items():
        for model_info in provider_models:
            models.append({
                "provider": provider,
                "model": model_info["model"],
                "description": model_info["description"],
                "key": f"{provider}:{model_info['model']}"
            })
    return models


def get_provider_models(provider: str) -> list[dict]:
    """
    Get supported models for a specific provider.

    Args:
        provider: Provider name (openai, anthropic, gemini, ollama)

    Returns:
        List of model dicts for the provider
    """
    return SUPPORTED_MODELS.get(provider, [])


def get_provider_class(provider: str):
    """
    Get the provider class for a given provider name.

    Args:
        provider: Provider name

    Returns:
        Provider class
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
    }
    return providers.get(provider)


def create_provider(provider: str, model: str, **kwargs):
    """
    Factory function to create a provider instance.

    Args:
        provider: Provider name
        model: Model name
        **kwargs: Additional config options (temperature, max_tokens, api_key)

    Returns:
        Configured provider instance
    """
    provider_class = get_provider_class(provider)
    if provider_class is None:
        raise ValueError(f"Unknown provider: {provider}")

    config = LLMConfig(
        model=model,
        temperature=kwargs.get("temperature", 0.8),
        max_tokens=kwargs.get("max_tokens", 500),
        api_key=kwargs.get("api_key"),
        base_url=kwargs.get("base_url"),
    )
    return provider_class(config)


__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "LLMConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
    "SUPPORTED_MODELS",
    "get_all_supported_models",
    "get_provider_models",
    "get_provider_class",
    "create_provider",
]
