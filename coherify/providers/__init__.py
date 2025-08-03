"""
External API providers for coherence evaluation.

This module provides integrations with external API providers like OpenAI and Anthropic
for production-quality model evaluation.
"""

from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .base import ModelProvider, ModelResponse
from .manager import (
    ProviderManager, get_provider_manager, get_provider, 
    list_available_providers, setup_providers
)

__all__ = [
    "ModelProvider",
    "ModelResponse", 
    "OpenAIProvider",
    "AnthropicProvider",
    "ProviderManager",
    "get_provider_manager",
    "get_provider",
    "list_available_providers", 
    "setup_providers"
]