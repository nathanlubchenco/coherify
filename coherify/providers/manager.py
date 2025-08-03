"""
Provider manager for external API providers.
"""

import os
from typing import Dict, List, Optional, Union
from .base import ModelProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


class ProviderManager:
    """Manages multiple external API providers."""
    
    def __init__(self):
        """Initialize provider manager."""
        self._providers: Dict[str, ModelProvider] = {}
        self._default_provider: Optional[str] = None
        
        # Auto-detect and initialize available providers
        self._auto_initialize_providers()
    
    def _auto_initialize_providers(self):
        """Automatically initialize providers based on available API keys."""
        # Check for OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.add_provider("openai", OpenAIProvider())
                if not self._default_provider:
                    self._default_provider = "openai"
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI provider: {e}")
        
        # Check for Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.add_provider("anthropic", AnthropicProvider())
                if not self._default_provider:
                    self._default_provider = "anthropic"
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic provider: {e}")
    
    def add_provider(self, name: str, provider: ModelProvider):
        """Add a provider to the manager."""
        self._providers[name] = provider
        if not self._default_provider:
            self._default_provider = name
    
    def get_provider(self, name: Optional[str] = None) -> ModelProvider:
        """Get a provider by name, or the default provider."""
        if name is None:
            name = self._default_provider
        
        if name is None:
            raise ValueError("No providers available. Set up API keys for OpenAI or Anthropic.")
        
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not found. Available: {list(self._providers.keys())}")
        
        return self._providers[name]
    
    def list_providers(self) -> List[str]:
        """List all available provider names."""
        return list(self._providers.keys())
    
    def get_default_provider(self) -> Optional[str]:
        """Get the default provider name."""
        return self._default_provider
    
    def set_default_provider(self, name: str):
        """Set the default provider."""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not found. Available: {list(self._providers.keys())}")
        self._default_provider = name
    
    def create_openai_provider(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
        name: str = "openai"
    ) -> OpenAIProvider:
        """Create and add an OpenAI provider."""
        provider = OpenAIProvider(api_key=api_key, model_name=model_name)
        self.add_provider(name, provider)
        return provider
    
    def create_anthropic_provider(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        name: str = "anthropic"
    ) -> AnthropicProvider:
        """Create and add an Anthropic provider."""
        provider = AnthropicProvider(api_key=api_key, model_name=model_name)
        self.add_provider(name, provider)
        return provider
    
    def get_all_available_models(self) -> Dict[str, List[str]]:
        """Get all available models from all providers."""
        models = {}
        for name, provider in self._providers.items():
            try:
                models[name] = provider.get_available_models()
            except Exception as e:
                print(f"Warning: Could not get models for {name}: {e}")
                models[name] = []
        return models
    
    def generate_with_multiple_providers(
        self,
        prompt: str,
        provider_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, 'ModelResponse']:
        """Generate responses using multiple providers."""
        if provider_names is None:
            provider_names = list(self._providers.keys())
        
        responses = {}
        for name in provider_names:
            if name in self._providers:
                try:
                    response = self._providers[name].generate_text(prompt, **kwargs)
                    responses[name] = response
                except Exception as e:
                    print(f"Warning: Generation failed for {name}: {e}")
        
        return responses
    
    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is available."""
        return name in self._providers


# Global provider manager instance
_provider_manager = None


def get_provider_manager() -> ProviderManager:
    """Get the global provider manager instance."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager


def get_provider(name: Optional[str] = None) -> ModelProvider:
    """Get a provider from the global manager."""
    return get_provider_manager().get_provider(name)


def list_available_providers() -> List[str]:
    """List all available providers."""
    return get_provider_manager().list_providers()


def setup_providers(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_model: str = "gpt-4",
    anthropic_model: str = "claude-3-5-sonnet-20241022"
) -> ProviderManager:
    """Set up providers with specific API keys."""
    manager = get_provider_manager()
    
    if openai_api_key:
        manager.create_openai_provider(api_key=openai_api_key, model_name=openai_model)
    
    if anthropic_api_key:
        manager.create_anthropic_provider(api_key=anthropic_api_key, model_name=anthropic_model)
    
    return manager