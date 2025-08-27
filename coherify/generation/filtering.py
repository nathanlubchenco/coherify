"""
Coherence-based filtering for generated text.
Filters and ranks generated candidates based on coherence criteria.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coherify.core.base import CoherenceMeasure, PropositionSet
from coherify.measures.hybrid import HybridCoherence


@dataclass
class FilterResult:
    """Result of coherence filtering."""

    passed_candidates: List[str]
    filtered_candidates: List[str]
    coherence_scores: List[float]
    filter_time: float
    filter_statistics: Dict[str, Any]
    metadata: Dict[str, Any]


class CoherenceFilter:
    """
    Base coherence filter for generated text candidates.

    Filters candidates based on coherence thresholds and quality criteria.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        min_coherence_threshold: float = 0.3,
        max_candidates: Optional[int] = None,
        require_improvement: bool = False,
    ):
        """
        Initialize coherence filter.

        Args:
            coherence_measure: Coherence measure for evaluation
            min_coherence_threshold: Minimum coherence score to pass
            max_candidates: Maximum candidates to return
            require_improvement: Whether to require coherence improvement over baseline
        """
        self.coherence_measure = coherence_measure or HybridCoherence()
        self.min_coherence_threshold = min_coherence_threshold
        self.max_candidates = max_candidates
        self.require_improvement = require_improvement

        # Statistics
        self.total_evaluations = 0
        self.total_filtered = 0

    def filter_candidates(
        self,
        context: str,
        candidates: List[str],
        baseline_coherence: Optional[float] = None,
    ) -> FilterResult:
        """
        Filter candidates based on coherence criteria.

        Args:
            context: Context for coherence evaluation
            candidates: List of generated text candidates
            baseline_coherence: Baseline coherence to compare against

        Returns:
            FilterResult with passed and filtered candidates
        """
        start_time = time.time()

        if not candidates:
            return FilterResult(
                passed_candidates=[],
                filtered_candidates=[],
                coherence_scores=[],
                filter_time=0.0,
                filter_statistics={},
                metadata={"no_candidates": True},
            )

        # Evaluate coherence for all candidates
        candidate_scores = []
        for candidate in candidates:
            score = self._evaluate_coherence(context, candidate)
            candidate_scores.append((candidate, score))
            self.total_evaluations += 1

        # Apply filtering criteria
        passed = []
        filtered = []

        for candidate, score in candidate_scores:
            if self._passes_filter(score, baseline_coherence):
                passed.append((candidate, score))
            else:
                filtered.append((candidate, score))
                self.total_filtered += 1

        # Sort passed candidates by coherence score
        passed.sort(key=lambda x: x[1], reverse=True)

        # Apply max candidates limit
        if self.max_candidates and len(passed) > self.max_candidates:
            extra_passed = passed[self.max_candidates :]
            passed = passed[: self.max_candidates]
            # Move extra to filtered
            filtered.extend(extra_passed)

        filter_time = time.time() - start_time

        # Extract lists and scores
        passed_candidates = [c for c, _ in passed]
        filtered_candidates = [c for c, _ in filtered]
        coherence_scores = [s for _, s in passed]

        # Compute statistics
        stats = self._compute_statistics(candidate_scores, passed, filtered)

        return FilterResult(
            passed_candidates=passed_candidates,
            filtered_candidates=filtered_candidates,
            coherence_scores=coherence_scores,
            filter_time=filter_time,
            filter_statistics=stats,
            metadata={
                "total_candidates": len(candidates),
                "passed_count": len(passed_candidates),
                "filtered_count": len(filtered_candidates),
                "filter_threshold": self.min_coherence_threshold,
            },
        )

    def _evaluate_coherence(self, context: str, candidate: str) -> float:
        """Evaluate coherence between context and candidate."""
        if not candidate.strip():
            return 0.0

        prop_set = PropositionSet.from_qa_pair(context, candidate)

        if len(prop_set.propositions) <= 1:
            return 1.0

        result = self.coherence_measure.compute(prop_set)
        return result.score

    def _passes_filter(self, score: float, baseline: Optional[float]) -> bool:
        """Check if score passes filtering criteria."""
        # Basic threshold check
        if score < self.min_coherence_threshold:
            return False

        # Improvement requirement
        if self.require_improvement and baseline is not None:
            if score <= baseline:
                return False

        return True

    def _compute_statistics(
        self,
        all_scores: List[Tuple[str, float]],
        passed: List[Tuple[str, float]],
        filtered: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """Compute filtering statistics."""
        all_coherence = [s for _, s in all_scores]
        passed_coherence = [s for _, s in passed]
        filtered_coherence = [s for _, s in filtered]

        stats = {
            "total_candidates": len(all_scores),
            "passed_candidates": len(passed),
            "filtered_candidates": len(filtered),
            "pass_rate": len(passed) / len(all_scores) if all_scores else 0,
            "coherence_statistics": {
                "all": self._score_stats(all_coherence),
                "passed": (
                    self._score_stats(passed_coherence) if passed_coherence else {}
                ),
                "filtered": (
                    self._score_stats(filtered_coherence) if filtered_coherence else {}
                ),
            },
        }

        return stats

    def _score_stats(self, scores: List[float]) -> Dict[str, float]:
        """Compute statistics for a list of scores."""
        if not scores:
            return {}

        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "median": np.median(scores),
        }

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get overall filter statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "total_filtered": self.total_filtered,
            "filter_rate": self.total_filtered / max(self.total_evaluations, 1),
            "threshold": self.min_coherence_threshold,
            "require_improvement": self.require_improvement,
        }


class AdaptiveCoherenceFilter(CoherenceFilter):
    """
    Adaptive filter that adjusts thresholds based on candidate quality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_threshold = self.min_coherence_threshold
        self.adaptation_factor = 0.1
        self.quality_history = []
        self.min_threshold = 0.1
        self.max_threshold = 0.8

    def filter_candidates(self, *args, **kwargs) -> FilterResult:
        """Filter with adaptive threshold adjustment."""
        result = super().filter_candidates(*args, **kwargs)

        # Adapt threshold based on results
        self._adapt_threshold(result)

        return result

    def _adapt_threshold(self, result: FilterResult):
        """Adapt threshold based on filtering results."""
        if not result.coherence_scores:
            return

        # Track average quality
        avg_quality = np.mean(result.coherence_scores)
        self.quality_history.append(avg_quality)

        # Adjust threshold based on pass rate and quality
        pass_rate = len(result.passed_candidates) / max(
            result.metadata["total_candidates"], 1
        )

        if pass_rate < 0.3:  # Too few candidates passing
            self.min_coherence_threshold = max(
                self.min_threshold,
                self.min_coherence_threshold - self.adaptation_factor,
            )
        elif pass_rate > 0.8:  # Too many candidates passing
            self.min_coherence_threshold = min(
                self.max_threshold,
                self.min_coherence_threshold + self.adaptation_factor,
            )

        # Quality-based adjustment
        if len(self.quality_history) >= 5:
            recent_quality = np.mean(self.quality_history[-3:])
            if recent_quality < 0.4:  # Low quality
                self.min_coherence_threshold = min(
                    self.max_threshold,
                    self.min_coherence_threshold + self.adaptation_factor / 2,
                )


class MultiStageFilter:
    """
    Multi-stage filtering pipeline with different coherence criteria.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        stages: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize multi-stage filter.

        Args:
            coherence_measure: Coherence measure for evaluation
            stages: List of stage configurations
        """
        self.coherence_measure = coherence_measure or HybridCoherence()

        # Default stages if none provided
        if stages is None:
            stages = [
                {
                    "name": "basic_filter",
                    "threshold": 0.2,
                    "description": "Basic coherence filter",
                },
                {
                    "name": "quality_filter",
                    "threshold": 0.4,
                    "description": "Quality coherence filter",
                },
                {
                    "name": "excellence_filter",
                    "threshold": 0.6,
                    "description": "Excellence coherence filter",
                },
            ]

        self.stages = stages
        self.stage_statistics = defaultdict(list)

    def filter_candidates(
        self, context: str, candidates: List[str], target_stage: Optional[str] = None
    ) -> Dict[str, FilterResult]:
        """
        Apply multi-stage filtering pipeline.

        Args:
            context: Context for coherence evaluation
            candidates: List of candidates to filter
            target_stage: Stop at specific stage (optional)

        Returns:
            Dictionary mapping stage names to FilterResults
        """
        results = {}
        current_candidates = candidates.copy()

        for stage in self.stages:
            stage_name = stage["name"]
            threshold = stage["threshold"]

            # Create filter for this stage
            stage_filter = CoherenceFilter(
                coherence_measure=self.coherence_measure,
                min_coherence_threshold=threshold,
            )

            # Apply filter
            stage_result = stage_filter.filter_candidates(context, current_candidates)
            results[stage_name] = stage_result

            # Track statistics
            self.stage_statistics[stage_name].append(
                {
                    "input_count": len(current_candidates),
                    "output_count": len(stage_result.passed_candidates),
                    "filter_rate": 1
                    - (
                        len(stage_result.passed_candidates)
                        / max(len(current_candidates), 1)
                    ),
                    "avg_coherence": (
                        np.mean(stage_result.coherence_scores)
                        if stage_result.coherence_scores
                        else 0
                    ),
                }
            )

            # Update candidates for next stage
            current_candidates = stage_result.passed_candidates

            # Stop if target stage reached or no candidates left
            if stage_name == target_stage or not current_candidates:
                break

        return results

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics for the entire pipeline."""
        pipeline_stats = {}

        for stage_name, stage_data in self.stage_statistics.items():
            if stage_data:
                pipeline_stats[stage_name] = {
                    "runs": len(stage_data),
                    "avg_input_count": np.mean([d["input_count"] for d in stage_data]),
                    "avg_output_count": np.mean(
                        [d["output_count"] for d in stage_data]
                    ),
                    "avg_filter_rate": np.mean([d["filter_rate"] for d in stage_data]),
                    "avg_coherence": np.mean([d["avg_coherence"] for d in stage_data]),
                }

        return pipeline_stats

    def recommend_stage(
        self, context: str, sample_candidates: List[str], target_output_count: int = 5
    ) -> str:
        """
        Recommend optimal stage based on sample candidates.

        Args:
            context: Context for evaluation
            sample_candidates: Sample of candidates to analyze
            target_output_count: Desired number of output candidates

        Returns:
            Recommended stage name
        """
        if not sample_candidates:
            return self.stages[0]["name"]

        # Test all stages on sample
        sample_results = self.filter_candidates(context, sample_candidates)

        # Find stage that produces closest to target count
        best_stage = self.stages[0]["name"]
        best_diff = float("inf")

        for stage_name, result in sample_results.items():
            output_count = len(result.passed_candidates)
            diff = abs(output_count - target_output_count)

            if diff < best_diff:
                best_diff = diff
                best_stage = stage_name

        return best_stage


class ContextAwareFilter(CoherenceFilter):
    """
    Filter that adapts to different types of context and generation tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_profiles = {
            "technical": {"threshold": 0.5, "weight_precision": True},
            "creative": {"threshold": 0.3, "weight_diversity": True},
            "factual": {"threshold": 0.6, "weight_accuracy": True},
            "conversational": {"threshold": 0.4, "weight_naturalness": True},
        }
        self.current_profile = "general"

    def set_context_profile(self, profile_name: str):
        """Set context profile for adaptive filtering."""
        if profile_name in self.context_profiles:
            self.current_profile = profile_name
            profile_config = self.context_profiles[profile_name]
            self.min_coherence_threshold = profile_config["threshold"]

    def analyze_context(self, context: str) -> str:
        """Analyze context to determine appropriate profile."""
        context_lower = context.lower()

        # Simple keyword-based classification
        if any(
            word in context_lower
            for word in ["technical", "algorithm", "method", "process"]
        ):
            return "technical"
        elif any(
            word in context_lower
            for word in ["story", "creative", "imagine", "describe"]
        ):
            return "creative"
        elif any(
            word in context_lower for word in ["fact", "evidence", "research", "study"]
        ):
            return "factual"
        elif any(
            word in context_lower
            for word in ["chat", "talk", "conversation", "discuss"]
        ):
            return "conversational"
        else:
            return "general"

    def filter_candidates(
        self, context: str, candidates: List[str], **kwargs
    ) -> FilterResult:
        """Filter with context-aware adaptation."""
        # Auto-detect context type
        detected_profile = self.analyze_context(context)
        if detected_profile != "general":
            self.set_context_profile(detected_profile)

        result = super().filter_candidates(context, candidates, **kwargs)

        # Add profile information to metadata
        result.metadata["context_profile"] = self.current_profile
        result.metadata["detected_profile"] = detected_profile

        return result
