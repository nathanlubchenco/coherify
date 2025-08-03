"""
Sampling-based approximation algorithms for large proposition sets.
Reduces computational complexity by sampling representative subsets.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from coherify.core.base import (
    CoherenceMeasure,
    PropositionSet,
    Proposition,
)
from coherify.measures.semantic import SemanticCoherence


@dataclass
class SamplingResult:
    """Result of sampling-based coherence approximation."""

    approximate_score: float
    sample_size: int
    total_size: int
    sampling_time: float
    computation_time: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sampling_method: str = "unknown"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PropositionSampler(ABC):
    """Base class for proposition sampling strategies."""

    @abstractmethod
    def sample(
        self, propositions: List[Proposition], sample_size: int, **kwargs
    ) -> List[Proposition]:
        """Sample propositions from the given list."""

    @abstractmethod
    def get_sampling_weights(
        self, propositions: List[Proposition]
    ) -> Optional[List[float]]:
        """Get sampling weights for propositions (if applicable)."""


class RandomSampler(PropositionSampler):
    """Simple random sampling without replacement."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample(
        self, propositions: List[Proposition], sample_size: int, **kwargs
    ) -> List[Proposition]:
        """Randomly sample propositions."""
        if sample_size >= len(propositions):
            return propositions.copy()

        return random.sample(propositions, sample_size)

    def get_sampling_weights(
        self, propositions: List[Proposition]
    ) -> Optional[List[float]]:
        """Random sampling uses uniform weights."""
        return None


class StratifiedSampler(PropositionSampler):
    """Stratified sampling based on proposition characteristics."""

    def __init__(
        self,
        strata_key: str = "length",
        num_strata: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize stratified sampler.

        Args:
            strata_key: How to stratify ('length', 'complexity', 'semantic')
            num_strata: Number of strata to create
            seed: Random seed
        """
        self.strata_key = strata_key
        self.num_strata = num_strata
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample(
        self, propositions: List[Proposition], sample_size: int, **kwargs
    ) -> List[Proposition]:
        """Sample from each stratum proportionally."""
        if sample_size >= len(propositions):
            return propositions.copy()

        # Create strata
        strata = self._create_strata(propositions)

        # Calculate samples per stratum
        total_props = len(propositions)
        sampled = []

        for stratum_props in strata:
            if not stratum_props:
                continue

            stratum_size = len(stratum_props)
            stratum_sample_size = max(
                1, int((stratum_size / total_props) * sample_size)
            )
            stratum_sample_size = min(stratum_sample_size, len(stratum_props))

            stratum_sample = random.sample(stratum_props, stratum_sample_size)
            sampled.extend(stratum_sample)

        # If we have too many, randomly remove some
        if len(sampled) > sample_size:
            sampled = random.sample(sampled, sample_size)

        # If we have too few, add random samples from remaining
        elif len(sampled) < sample_size:
            remaining = [p for p in propositions if p not in sampled]
            additional_needed = sample_size - len(sampled)
            if remaining and additional_needed > 0:
                additional = random.sample(
                    remaining, min(additional_needed, len(remaining))
                )
                sampled.extend(additional)

        return sampled

    def _create_strata(
        self, propositions: List[Proposition]
    ) -> List[List[Proposition]]:
        """Create strata based on stratification key."""
        if self.strata_key == "length":
            # Stratify by text length
            lengths = [len(p.text) for p in propositions]
            min_len, max_len = min(lengths), max(lengths)

            if min_len == max_len:
                return [propositions]  # All same length

            strata = [[] for _ in range(self.num_strata)]
            bin_size = (max_len - min_len) / self.num_strata

            for prop in propositions:
                bin_idx = min(
                    int((len(prop.text) - min_len) / bin_size), self.num_strata - 1
                )
                strata[bin_idx].append(prop)

            return strata

        elif self.strata_key == "complexity":
            # Stratify by estimated complexity (word count, punctuation, etc.)
            complexities = [self._estimate_complexity(p) for p in propositions]
            min_comp, max_comp = min(complexities), max(complexities)

            if min_comp == max_comp:
                return [propositions]

            strata = [[] for _ in range(self.num_strata)]
            bin_size = (max_comp - min_comp) / self.num_strata

            for prop, complexity in zip(propositions, complexities):
                bin_idx = min(
                    int((complexity - min_comp) / bin_size), self.num_strata - 1
                )
                strata[bin_idx].append(prop)

            return strata

        else:
            # Default to simple division
            chunk_size = len(propositions) // self.num_strata
            strata = []
            for i in range(0, len(propositions), chunk_size):
                strata.append(propositions[i : i + chunk_size])
            return strata

    def _estimate_complexity(self, proposition: Proposition) -> float:
        """Estimate text complexity."""
        text = proposition.text
        word_count = len(text.split())
        sentence_count = text.count(".") + text.count("!") + text.count("?") + 1
        avg_word_length = (
            np.mean([len(word) for word in text.split()]) if text.split() else 0
        )

        # Simple complexity score
        complexity = word_count * 0.5 + sentence_count * 2.0 + avg_word_length * 0.3
        return complexity

    def get_sampling_weights(
        self, propositions: List[Proposition]
    ) -> Optional[List[float]]:
        """Stratified sampling uses stratum-based weights."""
        return None


class DiversitySampler(PropositionSampler):
    """Diversity-based sampling to maximize representativeness."""

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        diversity_threshold: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Initialize diversity sampler.

        Args:
            coherence_measure: Measure to compute diversity
            diversity_threshold: Minimum diversity threshold
            seed: Random seed
        """
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.diversity_threshold = diversity_threshold
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample(
        self, propositions: List[Proposition], sample_size: int, **kwargs
    ) -> List[Proposition]:
        """Sample to maximize diversity."""
        if sample_size >= len(propositions):
            return propositions.copy()

        # Start with random proposition
        sampled = [random.choice(propositions)]
        remaining = [p for p in propositions if p != sampled[0]]

        # Greedily add most diverse propositions
        while len(sampled) < sample_size and remaining:
            best_prop = None
            best_diversity = -1

            for candidate in remaining:
                # Compute average diversity to already sampled
                diversities = []
                for sampled_prop in sampled:
                    # Use inverse coherence as diversity measure
                    test_set = PropositionSet([sampled_prop, candidate])
                    if hasattr(self.coherence_measure, "encoder"):
                        # For semantic measures, use embedding distance
                        try:
                            embeddings = self.coherence_measure.encoder.encode(
                                [sampled_prop.text, candidate.text]
                            )
                            from sklearn.metrics.pairwise import cosine_similarity

                            similarity = cosine_similarity(
                                [embeddings[0]], [embeddings[1]]
                            )[0][0]
                            diversity = 1.0 - similarity
                            diversities.append(diversity)
                        except Exception:
                            # Fallback to coherence-based diversity
                            coherence_result = self.coherence_measure.compute(test_set)
                            diversity = 1.0 - coherence_result.score
                            diversities.append(diversity)
                    else:
                        # Use coherence-based diversity
                        coherence_result = self.coherence_measure.compute(test_set)
                        diversity = 1.0 - coherence_result.score
                        diversities.append(diversity)

                avg_diversity = np.mean(diversities)
                if avg_diversity > best_diversity:
                    best_diversity = avg_diversity
                    best_prop = candidate

            if best_prop and best_diversity >= self.diversity_threshold:
                sampled.append(best_prop)
                remaining.remove(best_prop)
            else:
                # If no diverse enough option, add random
                if remaining:
                    random_prop = random.choice(remaining)
                    sampled.append(random_prop)
                    remaining.remove(random_prop)

        return sampled

    def get_sampling_weights(
        self, propositions: List[Proposition]
    ) -> Optional[List[float]]:
        """Diversity sampling doesn't use predetermined weights."""
        return None


class ImportanceSampler(PropositionSampler):
    """Importance-based sampling using proposition significance scores."""

    def __init__(
        self,
        importance_strategy: str = "centrality",
        coherence_measure: Optional[CoherenceMeasure] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize importance sampler.

        Args:
            importance_strategy: How to compute importance ('centrality', 'length', 'tf_idf')
            coherence_measure: Measure for computing centrality
            seed: Random seed
        """
        self.importance_strategy = importance_strategy
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample(
        self, propositions: List[Proposition], sample_size: int, **kwargs
    ) -> List[Proposition]:
        """Sample based on importance weights."""
        if sample_size >= len(propositions):
            return propositions.copy()

        # Compute importance weights
        weights = self.get_sampling_weights(propositions)
        if weights is None:
            # Fallback to random sampling
            return random.sample(propositions, sample_size)

        # Weighted sampling without replacement
        sampled_indices = np.random.choice(
            len(propositions), size=sample_size, replace=False, p=weights
        )

        return [propositions[i] for i in sampled_indices]

    def get_sampling_weights(
        self, propositions: List[Proposition]
    ) -> Optional[List[float]]:
        """Compute importance-based sampling weights."""
        if self.importance_strategy == "length":
            # Weight by text length
            lengths = [len(p.text) for p in propositions]
            weights = np.array(lengths, dtype=float)
            weights = weights / weights.sum()
            return weights.tolist()

        elif self.importance_strategy == "centrality":
            # Weight by centrality (average coherence with all others)
            weights = []

            for i, target_prop in enumerate(propositions):
                centrality_scores = []

                for j, other_prop in enumerate(propositions):
                    if i != j:
                        test_set = PropositionSet([target_prop, other_prop])
                        result = self.coherence_measure.compute(test_set)
                        centrality_scores.append(result.score)

                centrality = np.mean(centrality_scores) if centrality_scores else 0.0
                weights.append(centrality)

            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                return weights.tolist()
            else:
                return None

        elif self.importance_strategy == "tf_idf":
            # Simple TF-IDF based importance
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                texts = [p.text for p in propositions]
                vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
                tfidf_matrix = vectorizer.fit_transform(texts)

                # Use sum of TF-IDF scores as importance
                weights = np.array(tfidf_matrix.sum(axis=1)).flatten()
                weights = weights / weights.sum()
                return weights.tolist()

            except ImportError:
                # Fallback if sklearn not available
                return None

        return None


class SamplingBasedApproximator:
    """Main class for sampling-based coherence approximation."""

    def __init__(
        self,
        sampler: PropositionSampler,
        coherence_measure: CoherenceMeasure,
        num_bootstrap_samples: int = 10,
        confidence_level: float = 0.95,
    ):
        """
        Initialize sampling-based approximator.

        Args:
            sampler: Sampling strategy to use
            coherence_measure: Base coherence measure
            num_bootstrap_samples: Number of bootstrap samples for confidence interval
            confidence_level: Confidence level for interval estimation
        """
        self.sampler = sampler
        self.coherence_measure = coherence_measure
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level

    def approximate_coherence(
        self,
        prop_set: PropositionSet,
        sample_size: int,
        compute_confidence: bool = True,
    ) -> SamplingResult:
        """
        Approximate coherence using sampling.

        Args:
            prop_set: Proposition set to analyze
            sample_size: Number of propositions to sample
            compute_confidence: Whether to compute confidence interval

        Returns:
            SamplingResult with approximation
        """
        start_time = time.time()

        propositions = prop_set.propositions
        total_size = len(propositions)

        if sample_size >= total_size:
            # No need to sample
            sampling_time = time.time() - start_time
            comp_start = time.time()
            result = self.coherence_measure.compute(prop_set)
            comp_time = time.time() - comp_start

            return SamplingResult(
                approximate_score=result.score,
                sample_size=total_size,
                total_size=total_size,
                sampling_time=sampling_time,
                computation_time=comp_time,
                sampling_method=self.sampler.__class__.__name__,
                metadata={"exact_computation": True},
            )

        # Sample propositions
        sampled_props = self.sampler.sample(propositions, sample_size)
        sampling_time = time.time() - start_time

        # Compute coherence on sample
        comp_start = time.time()
        sampled_set = PropositionSet(
            propositions=sampled_props,
            context=prop_set.context,
            metadata=prop_set.metadata,
        )
        result = self.coherence_measure.compute(sampled_set)
        comp_time = time.time() - comp_start

        # Bootstrap confidence interval
        confidence_interval = None
        if compute_confidence and self.num_bootstrap_samples > 1:
            confidence_interval = self._compute_confidence_interval(
                propositions, sample_size
            )

        return SamplingResult(
            approximate_score=result.score,
            sample_size=len(sampled_props),
            total_size=total_size,
            sampling_time=sampling_time,
            computation_time=comp_time,
            confidence_interval=confidence_interval,
            sampling_method=self.sampler.__class__.__name__,
            metadata={
                "reduction_ratio": len(sampled_props) / total_size,
                "bootstrap_samples": (
                    self.num_bootstrap_samples if compute_confidence else 0
                ),
            },
        )

    def _compute_confidence_interval(
        self, propositions: List[Proposition], sample_size: int
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        bootstrap_scores = []

        for _ in range(self.num_bootstrap_samples):
            # Bootstrap sample
            bootstrap_sample = self.sampler.sample(propositions, sample_size)
            bootstrap_set = PropositionSet(propositions=bootstrap_sample)

            # Compute coherence
            result = self.coherence_measure.compute(bootstrap_set)
            bootstrap_scores.append(result.score)

        # Compute confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)

        return (lower_bound, upper_bound)

    def evaluate_sampling_quality(
        self, prop_set: PropositionSet, sample_sizes: List[int], num_trials: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate sampling quality across different sample sizes.

        Args:
            prop_set: Proposition set to evaluate
            sample_sizes: Different sample sizes to test
            num_trials: Number of trials per sample size

        Returns:
            Evaluation results
        """
        # Compute true coherence (if feasible)
        true_score = None
        if len(prop_set.propositions) <= 50:  # Only for small sets
            true_result = self.coherence_measure.compute(prop_set)
            true_score = true_result.score

        results = {}

        for sample_size in sample_sizes:
            if sample_size >= len(prop_set.propositions):
                continue

            trial_scores = []
            trial_times = []

            for _ in range(num_trials):
                result = self.approximate_coherence(
                    prop_set, sample_size, compute_confidence=False
                )
                trial_scores.append(result.approximate_score)
                trial_times.append(result.computation_time + result.sampling_time)

            # Compute statistics
            results[sample_size] = {
                "mean_score": np.mean(trial_scores),
                "std_score": np.std(trial_scores),
                "mean_time": np.mean(trial_times),
                "std_time": np.std(trial_times),
                "scores": trial_scores,
                "reduction_ratio": sample_size / len(prop_set.propositions),
            }

            if true_score is not None:
                results[sample_size]["absolute_error"] = abs(
                    np.mean(trial_scores) - true_score
                )
                results[sample_size]["relative_error"] = abs(
                    np.mean(trial_scores) - true_score
                ) / max(true_score, 1e-8)

        evaluation = {
            "sample_size_results": results,
            "total_propositions": len(prop_set.propositions),
            "true_score": true_score,
            "sampler_type": self.sampler.__class__.__name__,
            "coherence_measure": self.coherence_measure.__class__.__name__,
        }

        return evaluation
