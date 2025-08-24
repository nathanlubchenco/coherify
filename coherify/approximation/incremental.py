"""
Incremental and streaming approximation algorithms for dynamic proposition sets.
Efficiently updates coherence estimates as propositions are added or removed.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import time

from coherify.core.base import (
    CoherenceMeasure,
    PropositionSet,
    Proposition,
)


@dataclass
class IncrementalUpdate:
    """Result of an incremental coherence update."""

    new_score: float
    old_score: float
    operation: str  # 'add', 'remove', 'update'
    affected_proposition: Proposition
    update_time: float
    incremental_computation: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncrementalCoherenceTracker:
    """
    Tracks coherence incrementally as propositions are added/removed.

    Maintains partial computations to enable efficient updates without
    full recomputation.
    """

    def __init__(
        self,
        coherence_measure: CoherenceMeasure,
        max_cache_size: int = 1000,
        recompute_threshold: int = 10,
    ):
        """
        Initialize incremental tracker.

        Args:
            coherence_measure: Base coherence measure
            max_cache_size: Maximum cached pairwise computations
            recompute_threshold: Number of updates before full recomputation
        """
        self.coherence_measure = coherence_measure
        self.max_cache_size = max_cache_size
        self.recompute_threshold = recompute_threshold

        # Current state
        self.propositions: List[Proposition] = []
        self.current_score: float = 0.0
        self.pairwise_cache: Dict[Tuple[str, str], float] = {}
        self.update_count: int = 0

        # Statistics
        self.total_updates: int = 0
        self.full_recomputations: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def add_proposition(self, proposition: Proposition) -> IncrementalUpdate:
        """Add a proposition and update coherence incrementally."""
        start_time = time.time()
        old_score = self.current_score

        # Check if we need full recomputation
        if self.update_count >= self.recompute_threshold:
            self.propositions.append(proposition)
            new_score = self._full_recomputation()
            incremental = False
            self.update_count = 0
            self.full_recomputations += 1
        else:
            # Incremental update
            new_score = self._incremental_add(proposition)
            incremental = True
            self.update_count += 1

        self.current_score = new_score
        self.total_updates += 1
        update_time = time.time() - start_time

        return IncrementalUpdate(
            new_score=new_score,
            old_score=old_score,
            operation="add",
            affected_proposition=proposition,
            update_time=update_time,
            incremental_computation=incremental,
            metadata={
                "total_propositions": len(self.propositions),
                "cache_size": len(self.pairwise_cache),
                "update_count": self.update_count,
            },
        )

    def remove_proposition(self, proposition: Proposition) -> IncrementalUpdate:
        """Remove a proposition and update coherence incrementally."""
        start_time = time.time()
        old_score = self.current_score

        if proposition not in self.propositions:
            raise ValueError("Proposition not found in tracker")

        # For removal, often easier to just recompute
        self.propositions.remove(proposition)
        new_score = self._full_recomputation()

        self.current_score = new_score
        self.total_updates += 1
        self.full_recomputations += 1
        update_time = time.time() - start_time

        # Clean cache of entries involving removed proposition
        self._clean_cache_for_proposition(proposition)

        return IncrementalUpdate(
            new_score=new_score,
            old_score=old_score,
            operation="remove",
            affected_proposition=proposition,
            update_time=update_time,
            incremental_computation=False,
            metadata={
                "total_propositions": len(self.propositions),
                "cache_cleaned": True,
            },
        )

    def _incremental_add(self, new_proposition: Proposition) -> float:
        """Incrementally update coherence when adding a proposition."""
        if not self.propositions:
            # First proposition
            self.propositions.append(new_proposition)
            return 1.0

        # Compute coherence with each existing proposition
        pairwise_scores = []
        for existing_prop in self.propositions:
            score = self._get_pairwise_coherence(existing_prop, new_proposition)
            pairwise_scores.append(score)

        self.propositions.append(new_proposition)

        # Update overall coherence using weighted combination
        n = len(self.propositions)
        if n == 2:
            # Just two propositions
            return pairwise_scores[0]
        else:
            # Weighted update: combine old score with new pairwise scores
            old_weight = (n - 1) * (n - 2) / (n * (n - 1))  # Weight of existing pairs
            new_weight = 2 * (n - 1) / (n * (n - 1))  # Weight of new pairs

            avg_new_score = np.mean(pairwise_scores)
            updated_score = old_weight * self.current_score + new_weight * avg_new_score

            return updated_score

    def _get_pairwise_coherence(self, prop1: Proposition, prop2: Proposition) -> float:
        """Get pairwise coherence with caching."""
        # Create cache key (order independent)
        key1 = (prop1.text, prop2.text)
        key2 = (prop2.text, prop1.text)

        # Check cache
        if key1 in self.pairwise_cache:
            self.cache_hits += 1
            return self.pairwise_cache[key1]
        elif key2 in self.pairwise_cache:
            self.cache_hits += 1
            return self.pairwise_cache[key2]

        # Compute and cache
        self.cache_misses += 1
        prop_set = PropositionSet([prop1, prop2])
        result = self.coherence_measure.compute(prop_set)

        # Cache with size limit
        if len(self.pairwise_cache) < self.max_cache_size:
            self.pairwise_cache[key1] = result.score

        return result.score

    def _full_recomputation(self) -> float:
        """Perform full coherence recomputation."""
        if not self.propositions:
            return 0.0
        elif len(self.propositions) == 1:
            return 1.0

        prop_set = PropositionSet(self.propositions)
        result = self.coherence_measure.compute(prop_set)
        return result.score

    def _clean_cache_for_proposition(self, proposition: Proposition):
        """Remove cache entries involving a specific proposition."""
        keys_to_remove = []
        for key in self.pairwise_cache:
            if key[0] == proposition.text or key[1] == proposition.text:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.pairwise_cache[key]

    def get_current_state(self) -> Dict[str, Any]:
        """Get current tracker state."""
        return {
            "num_propositions": len(self.propositions),
            "current_score": self.current_score,
            "cache_size": len(self.pairwise_cache),
            "update_count": self.update_count,
            "total_updates": self.total_updates,
            "full_recomputations": self.full_recomputations,
            "cache_hit_rate": self.cache_hits
            / max(self.cache_hits + self.cache_misses, 1),
            "propositions": [p.text for p in self.propositions],
        }

    def reset(self):
        """Reset tracker to initial state."""
        self.propositions.clear()
        self.current_score = 0.0
        self.pairwise_cache.clear()
        self.update_count = 0
        self.total_updates = 0
        self.full_recomputations = 0
        self.cache_hits = 0
        self.cache_misses = 0


class StreamingCoherenceEstimator:
    """
    Estimates coherence for streaming proposition data.

    Uses sliding window and reservoir sampling to handle continuous streams
    of propositions while maintaining bounded memory usage.
    """

    def __init__(
        self,
        coherence_measure: CoherenceMeasure,
        window_size: int = 100,
        reservoir_size: int = 50,
        update_frequency: int = 10,
    ):
        """
        Initialize streaming estimator.

        Args:
            coherence_measure: Base coherence measure
            window_size: Size of sliding window
            reservoir_size: Size of reservoir sample
            update_frequency: How often to recompute coherence
        """
        self.coherence_measure = coherence_measure
        self.window_size = window_size
        self.reservoir_size = reservoir_size
        self.update_frequency = update_frequency

        # Sliding window
        self.window: Deque[Proposition] = deque(maxlen=window_size)

        # Reservoir sample (representative of all seen data)
        self.reservoir: List[Proposition] = []
        self.total_seen: int = 0

        # Current estimates
        self.window_coherence: float = 0.0
        self.reservoir_coherence: float = 0.0
        self.global_estimate: float = 0.0

        # Update tracking
        self.updates_since_computation: int = 0
        self.last_update_time: float = 0.0

        # Statistics
        self.total_propositions_seen: int = 0
        self.coherence_computations: int = 0

    def add_proposition(self, proposition: Proposition) -> Dict[str, Any]:
        """Add proposition to stream and update estimates."""
        start_time = time.time()

        # Add to window
        self.window.append(proposition)

        # Update reservoir sample
        self._update_reservoir(proposition)

        self.total_seen += 1
        self.total_propositions_seen += 1
        self.updates_since_computation += 1

        # Update coherence estimates if needed
        if self.updates_since_computation >= self.update_frequency:
            self._update_coherence_estimates()
            self.updates_since_computation = 0
            self.coherence_computations += 1

        update_time = time.time() - start_time
        self.last_update_time = update_time

        return {
            "window_coherence": self.window_coherence,
            "reservoir_coherence": self.reservoir_coherence,
            "global_estimate": self.global_estimate,
            "window_size": len(self.window),
            "reservoir_size": len(self.reservoir),
            "total_seen": self.total_seen,
            "update_time": update_time,
        }

    def _update_reservoir(self, proposition: Proposition):
        """Update reservoir sample using reservoir sampling algorithm."""
        if len(self.reservoir) < self.reservoir_size:
            # Reservoir not full, just add
            self.reservoir.append(proposition)
        else:
            # Reservoir full, replace with probability
            import random

            replace_idx = random.randint(1, self.total_seen)
            if replace_idx <= self.reservoir_size:
                self.reservoir[replace_idx - 1] = proposition

    def _update_coherence_estimates(self):
        """Update coherence estimates for window and reservoir."""
        # Window coherence
        if len(self.window) > 1:
            window_set = PropositionSet(list(self.window))
            window_result = self.coherence_measure.compute(window_set)
            self.window_coherence = window_result.score
        else:
            self.window_coherence = 1.0 if len(self.window) == 1 else 0.0

        # Reservoir coherence
        if len(self.reservoir) > 1:
            reservoir_set = PropositionSet(self.reservoir)
            reservoir_result = self.coherence_measure.compute(reservoir_set)
            self.reservoir_coherence = reservoir_result.score
        else:
            self.reservoir_coherence = 1.0 if len(self.reservoir) == 1 else 0.0

        # Global estimate (weighted combination)
        window_weight = 0.3
        reservoir_weight = 0.7
        self.global_estimate = (
            window_weight * self.window_coherence
            + reservoir_weight * self.reservoir_coherence
        )

    def get_coherence_trend(self, lookback: int = 10) -> Dict[str, Any]:
        """Analyze coherence trend over recent updates."""
        # This is a simplified version - in practice would maintain history
        return {
            "current_estimate": self.global_estimate,
            "window_coherence": self.window_coherence,
            "reservoir_coherence": self.reservoir_coherence,
            "trend": "stable",  # Simplified
            "confidence": min(len(self.reservoir) / self.reservoir_size, 1.0),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "total_propositions_seen": self.total_propositions_seen,
            "current_window_size": len(self.window),
            "current_reservoir_size": len(self.reservoir),
            "coherence_computations": self.coherence_computations,
            "avg_update_time": self.last_update_time,
            "compression_ratio": len(self.reservoir) / max(self.total_seen, 1),
            "current_estimates": {
                "window": self.window_coherence,
                "reservoir": self.reservoir_coherence,
                "global": self.global_estimate,
            },
        }

    def reset(self):
        """Reset estimator to initial state."""
        self.window.clear()
        self.reservoir.clear()
        self.total_seen = 0
        self.window_coherence = 0.0
        self.reservoir_coherence = 0.0
        self.global_estimate = 0.0
        self.updates_since_computation = 0
        self.total_propositions_seen = 0
        self.coherence_computations = 0


class AdaptiveApproximator:
    """
    Adaptive approximation that chooses the best strategy based on data characteristics.
    """

    def __init__(
        self,
        coherence_measure: CoherenceMeasure,
        strategies: Optional[List[str]] = None,
    ):
        """
        Initialize adaptive approximator.

        Args:
            coherence_measure: Base coherence measure
            strategies: List of strategies to consider
        """
        self.coherence_measure = coherence_measure
        self.strategies = strategies or ["sampling", "clustering", "incremental"]

        # Strategy instances
        self.strategy_instances = {}
        self._initialize_strategies()

        # Performance tracking
        from collections import defaultdict
        self.strategy_performance = defaultdict(list)
        self.strategy_usage = defaultdict(int)

    def _initialize_strategies(self):
        """Initialize strategy instances."""
        from .sampling import RandomSampler, SamplingBasedApproximator
        from .clustering import ClusterBasedApproximator

        if "sampling" in self.strategies:
            sampler = RandomSampler(seed=42)
            self.strategy_instances["sampling"] = SamplingBasedApproximator(
                sampler, self.coherence_measure
            )

        if "clustering" in self.strategies:
            self.strategy_instances["clustering"] = ClusterBasedApproximator(
                self.coherence_measure
            )

        if "incremental" in self.strategies:
            self.strategy_instances["incremental"] = IncrementalCoherenceTracker(
                self.coherence_measure
            )

    def approximate_coherence(
        self,
        prop_set: PropositionSet,
        target_accuracy: float = 0.9,
        max_computation_time: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Choose and apply best approximation strategy.

        Args:
            prop_set: Proposition set to analyze
            target_accuracy: Target approximation accuracy
            max_computation_time: Maximum computation time allowed

        Returns:
            Approximation result with strategy information
        """
        n_props = len(prop_set.propositions)

        # Choose strategy based on characteristics
        chosen_strategy = self._choose_strategy(
            n_props, target_accuracy, max_computation_time
        )

        start_time = time.time()

        # Apply chosen strategy
        if chosen_strategy == "sampling":
            sample_size = self._determine_sample_size(n_props, target_accuracy)
            result = self.strategy_instances["sampling"].approximate_coherence(
                prop_set, sample_size
            )
            approx_result = {
                "score": result.approximate_score,
                "method": "sampling",
                "sample_size": result.sample_size,
                "reduction_ratio": result.sample_size / result.total_size,
            }

        elif chosen_strategy == "clustering":
            target_clusters = self._determine_cluster_count(n_props, target_accuracy)
            result = self.strategy_instances["clustering"].approximate_coherence(
                prop_set, target_clusters
            )
            approx_result = {
                "score": result.approximate_score,
                "method": "clustering",
                "num_clusters": result.num_clusters,
                "reduction_ratio": result.num_clusters / result.total_propositions,
            }

        else:  # incremental or fallback
            # For non-incremental use, just compute directly
            result = self.coherence_measure.compute(prop_set)
            approx_result = {
                "score": result.score,
                "method": "direct",
                "reduction_ratio": 1.0,
            }

        computation_time = time.time() - start_time
        approx_result["computation_time"] = computation_time

        # Track performance
        self.strategy_usage[chosen_strategy] += 1
        self.strategy_performance[chosen_strategy].append(
            {
                "accuracy_target": target_accuracy,
                "computation_time": computation_time,
                "num_propositions": n_props,
            }
        )

        return approx_result

    def _choose_strategy(
        self, n_propositions: int, target_accuracy: float, max_time: float
    ) -> str:
        """Choose best strategy based on problem characteristics."""
        # Simple heuristics - in practice would be more sophisticated
        if n_propositions <= 20:
            return "direct"
        elif n_propositions <= 100:
            return "clustering"
        else:
            return "sampling"

    def _determine_sample_size(
        self, n_propositions: int, target_accuracy: float
    ) -> int:
        """Determine appropriate sample size."""
        # Simple heuristic based on target accuracy
        if target_accuracy >= 0.95:
            return min(n_propositions, max(50, int(n_propositions * 0.3)))
        elif target_accuracy >= 0.9:
            return min(n_propositions, max(30, int(n_propositions * 0.2)))
        else:
            return min(n_propositions, max(20, int(n_propositions * 0.1)))

    def _determine_cluster_count(
        self, n_propositions: int, target_accuracy: float
    ) -> int:
        """Determine appropriate cluster count."""
        if target_accuracy >= 0.95:
            return max(5, int(np.sqrt(n_propositions)))
        elif target_accuracy >= 0.9:
            return max(3, int(np.sqrt(n_propositions) * 0.7))
        else:
            return max(2, int(np.sqrt(n_propositions) * 0.5))
