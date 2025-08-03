"""
Base classes for external API providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ModelResponse:
    """Response from a model API call."""

    text: str
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelProvider(ABC):
    """Base class for external model API providers."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the model provider.

        Args:
            api_key: API key for authentication. If None, will try to get from environment.
            model_name: Default model name to use.
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.model_name = model_name
        self._validate_api_key()

    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""

    @abstractmethod
    def _validate_api_key(self) -> None:
        """Validate that API key is available and properly formatted."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate text using the model API.

        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            model: Model name to use (overrides default)
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse containing generated text and metadata
        """

    @abstractmethod
    def get_probability(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> float:
        """
        Get the probability/likelihood of a text sequence.

        Args:
            text: Text to evaluate
            model: Model name to use (overrides default)
            **kwargs: Additional provider-specific parameters

        Returns:
            Probability score (0.0 to 1.0)
        """

    @abstractmethod
    def embed_text(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> List[float]:
        """
        Get text embeddings.

        Args:
            text: Text to embed
            model: Embedding model name to use
            **kwargs: Additional provider-specific parameters

        Returns:
            List of embedding values
        """

    @abstractmethod
    def classify_entailment(
        self, premise: str, hypothesis: str, model: Optional[str] = None, **kwargs
    ) -> Dict[str, float]:
        """
        Classify entailment relationship between premise and hypothesis.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            model: Model name to use (overrides default)
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with entailment scores:
            {'entailment': float, 'contradiction': float, 'neutral': float}
        """

    def generate_multiple(
        self,
        prompt: str,
        num_generations: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[ModelResponse]:
        """
        Generate multiple responses for variance analysis.

        Args:
            prompt: Input prompt
            num_generations: Number of responses to generate
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            model: Model name to use
            **kwargs: Additional parameters

        Returns:
            List of ModelResponse objects
        """
        responses = []
        for i in range(num_generations):
            response = self.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                **kwargs,
            )
            response.metadata["generation_index"] = i
            responses.append(response)
        return responses

    def generate_with_temperatures(
        self,
        prompt: str,
        temperatures: List[float] = [0.3, 0.7, 1.0],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[ModelResponse]:
        """
        Generate responses with different temperature settings.

        Args:
            prompt: Input prompt
            temperatures: List of temperature values to try
            max_tokens: Maximum tokens per generation
            model: Model name to use
            **kwargs: Additional parameters

        Returns:
            List of ModelResponse objects with different temperatures
        """
        responses = []
        for temp in temperatures:
            response = self.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                model=model,
                **kwargs,
            )
            response.temperature = temp
            responses.append(response)
        return responses

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()

    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        # Default implementation - providers should override if they support model listing
        return [self.model_name] if self.model_name else []
