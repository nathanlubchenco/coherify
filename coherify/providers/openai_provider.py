"""
OpenAI API provider for coherence evaluation.
"""

import os
from typing import List, Dict, Optional
import numpy as np

from .base import ModelProvider, ModelResponse


class OpenAIProvider(ModelProvider):
    """OpenAI API provider for model interactions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Default model to use
            organization: OpenAI organization ID
            base_url: Base URL for API calls
        """
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

        super().__init__(api_key=api_key, model_name=model_name)

        # Try to import OpenAI client
        try:
            import openai

            self.client = openai.OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not found. Install with: pip install openai"
            )

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")

    def _validate_api_key(self) -> None:
        """Validate OpenAI API key."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        if not self.api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format. Should start with 'sk-'")

    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate text using OpenAI API."""
        model_to_use = model or self.model_name

        # Handle reasoning models (like o3)
        if model_to_use and model_to_use.startswith("o"):
            return self._generate_reasoning(
                prompt, model_to_use, temperature, max_tokens, **kwargs
            )

        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            choice = response.choices[0]
            return ModelResponse(
                text=choice.message.content,
                confidence=None,  # OpenAI doesn't provide confidence scores
                tokens_used=response.usage.total_tokens if response.usage else None,
                model_name=model_to_use,
                temperature=temperature,
                metadata={
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "response_id": response.id,
                },
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _generate_reasoning(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> ModelResponse:
        """Generate text using OpenAI reasoning models (o3, etc.)."""
        try:
            # Reasoning models may have different API structure
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,  # o3 uses max_completion_tokens
                temperature=temperature,
                **kwargs,
            )

            choice = response.choices[0]

            # Extract reasoning trace if available
            reasoning_trace = None
            if hasattr(choice.message, "reasoning") and choice.message.reasoning:
                reasoning_trace = choice.message.reasoning

            return ModelResponse(
                text=choice.message.content,
                confidence=None,
                tokens_used=response.usage.total_tokens if response.usage else None,
                model_name=model,
                temperature=temperature,
                metadata={
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "response_id": response.id,
                    "reasoning_trace": reasoning_trace,
                    "is_reasoning_model": True,
                },
            )

        except Exception as e:
            # Fallback to standard API if reasoning API fails
            try:
                return self.generate_text(
                    prompt, max_tokens, temperature, "gpt-4o", **kwargs
                )
            except Exception as e:
                raise RuntimeError(f"OpenAI reasoning model API call failed: {str(e)}")

    def get_probability(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> float:
        """
        Get probability using OpenAI completion API.
        Note: This is an approximation using logprobs.
        """
        model_to_use = model or self.model_name

        try:
            # Use completion API with logprobs for probability estimation
            response = self.client.completions.create(
                model=(
                    model_to_use
                    if model_to_use.startswith("text-")
                    else "gpt-4o-mini"
                ),
                prompt=text,
                max_tokens=0,  # Just want logprobs, not generation
                logprobs=1,
                echo=True,
                **kwargs,
            )

            if response.choices and response.choices[0].logprobs:
                # Average log probabilities
                logprobs = response.choices[0].logprobs.token_logprobs
                if logprobs:
                    avg_logprob = np.mean([lp for lp in logprobs if lp is not None])
                    # Convert log prob to probability (approximate)
                    return min(1.0, max(0.0, np.exp(avg_logprob)))

            # Fallback: use chat completion with low temperature for confidence estimation
            chat_response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "Rate the likelihood of this text on a scale of 0.0 to 1.0. Respond only with a number.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=5,
                temperature=0.1,
            )

            prob_text = chat_response.choices[0].message.content.strip()
            try:
                return float(prob_text)
            except (ValueError, TypeError):
                return 0.5  # Default moderate probability

        except Exception as e:
            print(f"Warning: Probability estimation failed: {e}")
            return 0.5  # Default probability

    def embed_text(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> List[float]:
        """Get text embeddings using OpenAI embeddings API."""
        embedding_model = model or "text-embedding-3-small"

        try:
            response = self.client.embeddings.create(
                model=embedding_model, input=text, **kwargs
            )

            return response.data[0].embedding

        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings API call failed: {str(e)}")

    def classify_entailment(
        self, premise: str, hypothesis: str, model: Optional[str] = None, **kwargs
    ) -> Dict[str, float]:
        """Classify entailment using OpenAI models."""
        model_to_use = model or self.model_name

        prompt = f"""
Analyze the logical relationship between these two statements:

Premise: "{premise}"
Hypothesis: "{hypothesis}"

Classify the relationship as one of:
- ENTAILMENT: The hypothesis logically follows from the premise
- CONTRADICTION: The hypothesis contradicts the premise  
- NEUTRAL: The hypothesis is neither entailed nor contradicted

Respond with only the classification and a confidence score (0.0-1.0).
Format: CLASSIFICATION confidence_score
"""

        try:
            response = self.generate_text(
                prompt=prompt, max_tokens=20, temperature=0.1, model=model_to_use
            )

            result_text = response.text.strip().upper()

            # Parse response
            if "ENTAILMENT" in result_text:
                return {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1}
            elif "CONTRADICTION" in result_text:
                return {"entailment": 0.1, "contradiction": 0.8, "neutral": 0.1}
            else:  # NEUTRAL or unknown
                return {"entailment": 0.2, "contradiction": 0.2, "neutral": 0.6}

        except Exception as e:
            print(f"Warning: Entailment classification failed: {e}")
            # Default to neutral
            return {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}

    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        try:
            models = self.client.models.list()
            return [
                model.id
                for model in models.data
                if model.id.startswith(("gpt-", "text-", "o"))
            ]
        except Exception:
            # Fallback list if API call fails
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "o3-mini",  # Reasoning models
                "o3",
            ]
