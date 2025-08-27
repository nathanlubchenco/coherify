"""
Multi-Response Coherence Measures

This module implements coherence measures that generate multiple responses
and evaluate consistency between them, enabling detection of model uncertainty
and systematic inconsistencies.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # Fallback implementations for numpy functions
    class MockNumpy:
        @staticmethod
        def zeros(shape):
            if isinstance(shape, tuple):
                return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
            return [0.0] * shape

        @staticmethod
        def fill_diagonal(arr, val):
            for i in range(min(len(arr), len(arr[0]) if arr else 0)):
                arr[i][i] = val

        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0.0

        @staticmethod
        def std(arr):
            if not arr:
                return 0.0
            mean_val = sum(arr) / len(arr)
            return (sum((x - mean_val) ** 2 for x in arr) / len(arr)) ** 0.5

        @staticmethod
        def var(arr):
            if not arr:
                return 0.0
            mean_val = sum(arr) / len(arr)
            return sum((x - mean_val) ** 2 for x in arr) / len(arr)

        @staticmethod
        def corrcoef(x, y):
            # Simple correlation coefficient
            if len(x) != len(y) or len(x) < 2:
                return [[1.0, 0.0], [0.0, 1.0]]

            n = len(x)
            mean_x = sum(x) / n
            mean_y = sum(y) / n

            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))

            denominator = (sum_sq_x * sum_sq_y) ** 0.5

            if denominator == 0:
                corr = 0.0
            else:
                corr = numerator / denominator

            return [[1.0, corr], [corr, 1.0]]

        @staticmethod
        def linspace(start, stop, num):
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]

        @staticmethod
        def argmax(arr):
            return arr.index(max(arr))

        @staticmethod
        def argmin(arr):
            return arr.index(min(arr))

    np = MockNumpy()

from coherify.core.base import (
    CoherenceMeasure,
    CoherenceResult,
    Proposition,
    PropositionSet,
)
from coherify.measures.hybrid import HybridCoherence
from coherify.measures.semantic import SemanticCoherence


@dataclass
class MultiResponseConfig:
    """Configuration for multi-response coherence evaluation."""

    num_responses: int = 5
    temperature_range: Tuple[float, float] = (0.3, 0.9)
    temperature_strategy: str = "uniform"  # "uniform", "fixed", "adaptive"
    max_response_length: int = 512
    response_timeout: float = 30.0
    enable_reasoning_trace: bool = False
    consistency_threshold: float = 0.7


@dataclass
class MultiResponseResult:
    """Result from multi-response coherence evaluation."""

    responses: List[str]
    coherence_scores: List[float]
    pairwise_coherences: Any  # Can be np.ndarray or list of lists
    mean_coherence: float
    consistency_score: float
    confidence_score: float
    temperature_variance: float
    best_response_idx: int
    worst_response_idx: int
    metadata: Dict[str, Any]


class MultiResponseCoherenceMeasure(CoherenceMeasure):
    """Base class for multi-response coherence evaluation."""

    def __init__(
        self, base_measure: CoherenceMeasure, config: MultiResponseConfig, provider=None
    ):
        """
        Initialize multi-response coherence measure.

        Args:
            base_measure: Underlying coherence measure to use
            config: Configuration for multi-response evaluation
            provider: API provider for response generation (optional)
        """
        self.base_measure = base_measure
        self.config = config
        self.provider = provider

    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute coherence using base measure on single proposition set."""
        return self.base_measure.compute(prop_set)

    def compute_multi_response(
        self, prompt: str, context: Optional[str] = None
    ) -> MultiResponseResult:
        """
        Generate multiple responses and evaluate coherence between them.

        Args:
            prompt: The prompt to generate responses for
            context: Optional context for response generation

        Returns:
            MultiResponseResult with coherence analysis
        """
        # Generate multiple responses
        responses = self._generate_responses(prompt, context)

        # Convert responses to proposition sets
        prop_sets = self._responses_to_proposition_sets(responses, context)

        # Compute pairwise coherences
        pairwise_coherences = self._compute_pairwise_coherences(prop_sets)

        # Compute individual coherence scores
        coherence_scores = self._compute_individual_coherences(prop_sets)

        # Analyze consistency and confidence
        consistency_score = self._compute_consistency_score(pairwise_coherences)
        confidence_score = self._compute_confidence_score(
            pairwise_coherences, coherence_scores
        )
        temperature_variance = self._compute_temperature_variance(responses)

        # Identify best and worst responses
        best_idx = int(np.argmax(coherence_scores))
        worst_idx = int(np.argmin(coherence_scores))

        return MultiResponseResult(
            responses=responses,
            coherence_scores=coherence_scores,
            pairwise_coherences=pairwise_coherences,
            mean_coherence=float(np.mean(coherence_scores)),
            consistency_score=consistency_score,
            confidence_score=confidence_score,
            temperature_variance=temperature_variance,
            best_response_idx=best_idx,
            worst_response_idx=worst_idx,
            metadata={
                "config": self.config,
                "base_measure": self.base_measure.__class__.__name__,
                "provider": self.provider.provider_name if self.provider else "local",
            },
        )

    def _generate_responses(
        self, prompt: str, context: Optional[str] = None
    ) -> List[str]:
        """Generate multiple responses using different temperatures."""
        if not self.provider:
            # Fallback: create synthetic responses for testing
            return self._generate_synthetic_responses(prompt, context)

        responses = []
        temperatures = self._get_temperatures()

        for temp in temperatures:
            try:
                response = self.provider.generate(
                    prompt=prompt,
                    context=context,
                    temperature=temp,
                    max_tokens=self.config.max_response_length,
                    timeout=self.config.response_timeout,
                )
                responses.append(response)
            except Exception as e:
                # Fallback to previous response or empty string
                if responses:
                    responses.append(responses[-1])
                else:
                    responses.append(f"Error generating response: {str(e)}")

        return responses

    def _generate_synthetic_responses(
        self, prompt: str, context: Optional[str] = None
    ) -> List[str]:
        """Generate synthetic responses for testing when no provider available."""
        base_response = f"Response to: {prompt}"
        if context:
            base_response = f"Based on context: {context[:50]}... {base_response}"

        # Create variations by adding different details
        variations = [
            base_response,
            f"{base_response} Additionally, this includes more details.",
            f"{base_response} However, there are some nuances to consider.",
            f"{base_response} Furthermore, the implications are significant.",
            f"{base_response} It's important to note the broader context.",
        ]

        return variations[: self.config.num_responses]

    def _get_temperatures(self) -> List[float]:
        """Get list of temperatures based on configuration."""
        if self.config.temperature_strategy == "uniform":
            return list(
                np.linspace(
                    self.config.temperature_range[0],
                    self.config.temperature_range[1],
                    self.config.num_responses,
                )
            )
        elif self.config.temperature_strategy == "fixed":
            return [0.7] * self.config.num_responses
        elif self.config.temperature_strategy == "adaptive":
            # Start with low temperature, increase if responses are too similar
            temps = [0.3, 0.5, 0.7, 0.8, 0.9]
            return temps[: self.config.num_responses]
        else:
            raise ValueError(
                f"Unknown temperature strategy: {self.config.temperature_strategy}"
            )

    def _responses_to_proposition_sets(
        self, responses: List[str], context: Optional[str]
    ) -> List[PropositionSet]:
        """Convert responses to PropositionSets."""
        prop_sets = []
        for response in responses:
            try:
                prop_set = PropositionSet.from_qa_pair("", response)
                if context:
                    prop_set.context = context
                prop_sets.append(prop_set)
            except Exception:
                # Fallback: create simple proposition set
                props = [Proposition(text=response)]
                prop_sets.append(PropositionSet(propositions=props, context=context))

        return prop_sets

    def _compute_pairwise_coherences(self, prop_sets: List[PropositionSet]):
        """Compute pairwise coherences between all proposition sets."""
        n = len(prop_sets)
        coherences = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Combine proposition sets for coherence evaluation
                combined_props = prop_sets[i].propositions + prop_sets[j].propositions
                combined_set = PropositionSet(
                    propositions=combined_props, context=prop_sets[i].context
                )

                result = self.base_measure.compute(combined_set)
                coherences[i, j] = result.score
                coherences[j, i] = result.score

        # Set diagonal to 1.0 (perfect self-coherence)
        np.fill_diagonal(coherences, 1.0)

        return coherences

    def _compute_individual_coherences(
        self, prop_sets: List[PropositionSet]
    ) -> List[float]:
        """Compute coherence scores for individual responses."""
        scores = []
        for prop_set in prop_sets:
            if len(prop_set.propositions) > 1:
                result = self.base_measure.compute(prop_set)
                scores.append(result.score)
            else:
                # Single proposition: assign neutral coherence
                scores.append(0.5)

        return scores

    def _compute_consistency_score(self, pairwise_coherences) -> float:
        """Compute overall consistency score from pairwise coherences."""
        # Extract upper triangular values (excluding diagonal)
        n = len(pairwise_coherences)
        upper_tri = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_tri.append(pairwise_coherences[i][j])

        if not upper_tri:
            return 1.0

        # Consistency is the mean of pairwise coherences
        return float(np.mean(upper_tri))

    def _compute_confidence_score(
        self, pairwise_coherences, individual_scores: List[float]
    ) -> float:
        """
        Compute confidence score based on coherence consistency.

        High confidence when:
        - High individual coherence scores
        - High pairwise coherence scores
        - Low variance in scores
        """
        consistency = self._compute_consistency_score(pairwise_coherences)
        mean_individual = np.mean(individual_scores)
        score_variance = np.var(individual_scores)

        # Combine factors: high mean, high consistency, low variance
        confidence = (mean_individual + consistency) / 2 - score_variance

        # Normalize to [0, 1]
        return float(max(0.0, min(1.0, confidence)))

    def _compute_temperature_variance(self, responses: List[str]) -> float:
        """Compute variance in response lengths as proxy for temperature effects."""
        lengths = [len(response) for response in responses]
        if len(lengths) <= 1:
            return 0.0

        # Normalize by mean length
        mean_length = np.mean(lengths)
        if mean_length == 0:
            return 0.0

        variance = np.var(lengths) / mean_length
        return float(variance)


class TemperatureVarianceCoherence(MultiResponseCoherenceMeasure):
    """Coherence measure specifically designed for temperature variance analysis."""

    def __init__(
        self,
        base_measure: Optional[CoherenceMeasure] = None,
        config: Optional[MultiResponseConfig] = None,
        provider=None,
    ):
        """Initialize with defaults optimized for temperature variance."""
        if base_measure is None:
            base_measure = HybridCoherence()

        if config is None:
            config = MultiResponseConfig(
                num_responses=5,
                temperature_range=(0.2, 0.8),
                temperature_strategy="uniform",
                consistency_threshold=0.6,
            )

        super().__init__(base_measure, config, provider)

    def evaluate_temperature_consistency(
        self, prompt: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate how consistent responses are across temperature settings.

        Returns detailed analysis of temperature effects on coherence.
        """
        result = self.compute_multi_response(prompt, context)

        # Analyze temperature effects
        temperatures = self._get_temperatures()
        temp_analysis = {}

        for i, temp in enumerate(temperatures):
            temp_analysis[f"temp_{temp:.1f}"] = {
                "response": result.responses[i],
                "coherence": result.coherence_scores[i],
                "length": len(result.responses[i]),
            }

        # Compute temperature-coherence correlation
        temp_coherence_corr = np.corrcoef(temperatures, result.coherence_scores)[0, 1]

        return {
            "multi_response_result": result,
            "temperature_analysis": temp_analysis,
            "temperature_coherence_correlation": float(temp_coherence_corr),
            "optimal_temperature": temperatures[result.best_response_idx],
            "consistency_verdict": (
                "consistent"
                if result.consistency_score > self.config.consistency_threshold
                else "inconsistent"
            ),
        }


class SelfConsistencyCoherence(MultiResponseCoherenceMeasure):
    """Coherence measure for self-consistency evaluation across multiple generations."""

    def __init__(
        self,
        base_measure: Optional[CoherenceMeasure] = None,
        config: Optional[MultiResponseConfig] = None,
        provider=None,
    ):
        """Initialize with defaults optimized for self-consistency."""
        if base_measure is None:
            base_measure = SemanticCoherence()

        if config is None:
            config = MultiResponseConfig(
                num_responses=3,
                temperature_range=(
                    0.7,
                    0.7,
                ),  # Fixed temperature for pure self-consistency
                temperature_strategy="fixed",
                consistency_threshold=0.8,
            )

        super().__init__(base_measure, config, provider)

    def evaluate_self_consistency(
        self, prompt: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate self-consistency by generating multiple responses at same temperature.

        This is useful for detecting when models give different answers to
        identical prompts, indicating uncertainty or randomness.
        """
        result = self.compute_multi_response(prompt, context)

        # Check if all responses are essentially the same
        response_similarity = self._compute_response_similarity(result.responses)

        # Majority voting (if applicable)
        majority_response = self._find_majority_response(result.responses)

        return {
            "multi_response_result": result,
            "response_similarity": response_similarity,
            "majority_response": majority_response,
            "self_consistency_score": result.consistency_score,
            "is_self_consistent": result.consistency_score
            > self.config.consistency_threshold,
            "response_diversity": 1.0 - response_similarity,
        }

    def _compute_response_similarity(self, responses: List[str]) -> float:
        """Compute average similarity between responses."""
        if len(responses) <= 1:
            return 1.0

        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # Simple similarity based on normalized edit distance
                sim = self._string_similarity(responses[i], responses[j])
                similarities.append(sim)

        return float(np.mean(similarities))

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute normalized string similarity."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Jaccard similarity on words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _find_majority_response(self, responses: List[str]) -> Dict[str, Any]:
        """Find majority response if one exists."""
        # Group similar responses
        groups = {}
        threshold = 0.8

        for response in responses:
            matched = False
            for group_key in groups:
                if self._string_similarity(response, group_key) > threshold:
                    groups[group_key].append(response)
                    matched = True
                    break

            if not matched:
                groups[response] = [response]

        # Find largest group
        largest_group = max(groups.items(), key=lambda x: len(x[1]))

        return {
            "majority_response": largest_group[0],
            "group_size": len(largest_group[1]),
            "total_responses": len(responses),
            "majority_fraction": len(largest_group[1]) / len(responses),
            "num_unique_groups": len(groups),
        }
