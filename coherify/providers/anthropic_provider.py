"""
Anthropic API provider for coherence evaluation.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from .base import ModelProvider, ModelResponse


class AnthropicProvider(ModelProvider):
    """Anthropic API provider for model interactions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        base_url: Optional[str] = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model_name: Default model to use
            base_url: Base URL for API calls
        """
        self.base_url = base_url or os.getenv(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )

        super().__init__(api_key=api_key, model_name=model_name)

        # Try to import Anthropic client
        try:
            import anthropic

            self.client = anthropic.Anthropic(
                api_key=self.api_key, base_url=self.base_url
            )
        except ImportError:
            raise ImportError(
                "Anthropic library not found. Install with: pip install anthropic"
            )

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get Anthropic API key from environment."""
        return os.getenv("ANTHROPIC_API_KEY")

    def _validate_api_key(self) -> None:
        """Validate Anthropic API key."""
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        if not self.api_key.startswith("sk-ant-"):
            raise ValueError(
                "Invalid Anthropic API key format. Should start with 'sk-ant-'"
            )

    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate text using Anthropic API."""
        model_to_use = model or self.model_name
        max_tokens = max_tokens or 1024

        try:
            response = self.client.messages.create(
                model=model_to_use,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Extract text content
            text_content = ""
            for content_block in response.content:
                if content_block.type == "text":
                    text_content += content_block.text

            return ModelResponse(
                text=text_content,
                confidence=None,  # Anthropic doesn't provide confidence scores directly
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                model_name=model_to_use,
                temperature=temperature,
                metadata={
                    "stop_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "response_id": response.id,
                },
            )

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")

    def get_probability(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> float:
        """
        Get probability estimation using Anthropic models.
        Since Anthropic doesn't provide direct logprobs, we use a prompt-based approach.
        """
        model_to_use = model or self.model_name

        prompt = f"""
Rate the likelihood/probability of this text being natural, coherent, and factually plausible on a scale from 0.0 to 1.0, where:
- 0.0 means extremely unlikely/incoherent/implausible
- 0.5 means neutral/uncertain
- 1.0 means extremely likely/coherent/plausible

Text to evaluate: "{text}"

Respond with only a number between 0.0 and 1.0, no explanation.
"""

        try:
            response = self.generate_text(
                prompt=prompt, max_tokens=10, temperature=0.1, model=model_to_use
            )

            prob_text = response.text.strip()
            try:
                probability = float(prob_text)
                return max(0.0, min(1.0, probability))  # Clamp to valid range
            except ValueError:
                return 0.5  # Default moderate probability

        except Exception as e:
            print(f"Warning: Probability estimation failed: {e}")
            return 0.5  # Default probability

    def embed_text(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> List[float]:
        """
        Get text embeddings using Anthropic models.
        Note: Anthropic doesn't provide dedicated embedding models,
        so we use a workaround with text analysis.
        """
        # Since Anthropic doesn't have embedding models, we'll create a semantic fingerprint
        # using the model's understanding of the text
        model_to_use = model or self.model_name

        prompt = f"""
Analyze this text and provide a semantic representation as 10 numerical values between -1.0 and 1.0 that capture its key semantic properties:

Text: "{text}"

For each dimension, consider:
1. Sentiment (negative to positive)
2. Formality (informal to formal)
3. Complexity (simple to complex)
4. Concreteness (abstract to concrete)
5. Certainty (uncertain to certain)
6. Emotional intensity (calm to intense)
7. Technical level (general to technical)
8. Time orientation (past to future)
9. Subjectivity (objective to subjective)
10. Urgency (casual to urgent)

Respond with only 10 numbers separated by commas, no explanation.
Example: -0.2, 0.8, 0.1, -0.5, 0.3, 0.0, 0.9, -0.1, 0.4, 0.2
"""

        try:
            response = self.generate_text(
                prompt=prompt, max_tokens=50, temperature=0.1, model=model_to_use
            )

            # Parse the response to extract numerical values
            values_text = response.text.strip()
            try:
                values = [float(x.strip()) for x in values_text.split(",")]
                if len(values) == 10:
                    # Normalize to create a proper embedding-like vector
                    np_values = np.array(values)
                    norm = np.linalg.norm(np_values)
                    if norm > 0:
                        np_values = np_values / norm
                    return np_values.tolist()
            except Exception:
                pass

            # Fallback: create a simple hash-based embedding
            import hashlib

            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hex to numbers
            embedding = []
            for i in range(0, len(text_hash), 3):
                hex_chunk = text_hash[i : i + 3]
                val = int(hex_chunk, 16) / 4095.0 * 2 - 1  # Normalize to [-1, 1]
                embedding.append(val)

            return embedding[:384]  # Standard embedding dimension

        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Return a zero vector as fallback
            return [0.0] * 384

    def classify_entailment(
        self, premise: str, hypothesis: str, model: Optional[str] = None, **kwargs
    ) -> Dict[str, float]:
        """Classify entailment using Anthropic models."""
        model_to_use = model or self.model_name

        prompt = f"""
Analyze the logical relationship between these two statements:

Premise: "{premise}"
Hypothesis: "{hypothesis}"

Determine if the hypothesis:
1. ENTAILS from the premise (logically follows)
2. CONTRADICTS the premise (logically conflicts)
3. Is NEUTRAL to the premise (neither follows nor conflicts)

Provide confidence scores for each relationship (0.0-1.0) that sum to 1.0.

Respond in this exact format:
entailment: X.XX
contradiction: X.XX
neutral: X.XX

Where X.XX are decimal numbers that sum to 1.0.
"""

        try:
            response = self.generate_text(
                prompt=prompt, max_tokens=50, temperature=0.1, model=model_to_use
            )

            result_text = response.text.strip().lower()

            # Parse the structured response
            scores = {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}

            lines = result_text.split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    try:
                        score = float(value.strip())
                        if key in scores:
                            scores[key] = score
                    except ValueError:
                        continue

            # Normalize scores to sum to 1.0
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

            return scores

        except Exception as e:
            print(f"Warning: Entailment classification failed: {e}")
            # Default to neutral distribution
            return {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}

    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        # Anthropic doesn't provide a models endpoint, so return known models
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
