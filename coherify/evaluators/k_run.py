"""
K-run orchestration for benchmark evaluation.

Provides orchestration for running benchmarks multiple times with different
configurations and aggregating results.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .majority_voting import MajorityVotingEvaluator, VotingResult


@dataclass
class KRunConfiguration:
    """Configuration for K-run evaluation."""

    k_runs: int = 5
    voting_strategy: str = "simple"
    parallel_execution: bool = True
    max_workers: int = 4
    retry_failed_runs: bool = True
    max_retries: int = 2
    cache_intermediate: bool = True
    coherence_threshold: float = 0.5


@dataclass
class KRunResult:
    """Result of K-run benchmark evaluation."""

    dataset_results: Dict[str, Any]
    configuration: KRunConfiguration
    total_time: float
    successful_samples: int
    failed_samples: int
    retry_count: int

    def get_success_rate(self) -> float:
        """Get overall success rate."""
        total = self.successful_samples + self.failed_samples
        return self.successful_samples / total if total > 0 else 0.0


class KRunBenchmarkEvaluator:
    """
    Orchestrator for running benchmarks K times with majority voting.

    Provides:
    - Parallel execution of multiple runs
    - Retry logic for failed evaluations
    - Progress tracking and intermediate caching
    - Comprehensive result aggregation
    """

    def __init__(self, base_evaluator: Any, config: Optional[KRunConfiguration] = None):
        """
        Initialize K-run benchmark evaluator.

        Args:
            base_evaluator: Base benchmark evaluator
            config: K-run configuration
        """
        self.base_evaluator = base_evaluator
        self.config = config or KRunConfiguration()

        # Initialize majority voting evaluator
        self.majority_evaluator = MajorityVotingEvaluator(
            base_evaluator=base_evaluator,
            k_runs=self.config.k_runs,
            voting_strategy=self.config.voting_strategy,
            coherence_threshold=self.config.coherence_threshold,
        )

        # Runtime state
        self._intermediate_cache: Dict[str, Any] = {}
        self._retry_counts: Dict[int, int] = {}

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> KRunResult:
        """
        Evaluate dataset with K-run majority voting.

        Args:
            dataset: Dataset to evaluate
            progress_callback: Progress callback (progress_pct, status_msg)

        Returns:
            KRunResult with comprehensive evaluation results
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(
                0.0, f"Starting K-run evaluation (K={self.config.k_runs})"
            )

        if self.config.parallel_execution:
            results = self._evaluate_parallel(dataset, progress_callback)
        else:
            results = self._evaluate_sequential(dataset, progress_callback)

        total_time = time.time() - start_time

        # Count success/failure
        voting_results = results["voting_results"]
        successful = len([r for r in voting_results if r.final_answer is not None])
        failed = len(voting_results) - successful

        # Total retry count
        total_retries = sum(self._retry_counts.values())

        if progress_callback:
            progress_callback(
                1.0,
                f"K-run evaluation completed ({successful}/{len(voting_results)} successful)",
            )

        return KRunResult(
            dataset_results=results,
            configuration=self.config,
            total_time=total_time,
            successful_samples=successful,
            failed_samples=failed,
            retry_count=total_retries,
        )

    def _evaluate_sequential(
        self,
        dataset: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Evaluate dataset sequentially."""

        def dataset_progress(current, total):
            if progress_callback:
                progress = current / total if total > 0 else 0.0
                progress_callback(progress, f"Processing sample {current}/{total}")

        return self.majority_evaluator.evaluate_dataset_with_voting(
            dataset, progress_callback=dataset_progress
        )

    def _evaluate_parallel(
        self,
        dataset: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Evaluate dataset with parallel execution."""
        voting_results = []
        completed_samples = 0
        total_samples = len(dataset)

        # Use ThreadPoolExecutor for parallel sample evaluation
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all samples
            future_to_sample = {
                executor.submit(self._evaluate_single_with_retry, i, sample): (
                    i,
                    sample,
                )
                for i, sample in enumerate(dataset)
            }

            # Process completed futures
            for future in as_completed(future_to_sample):
                sample_idx, sample = future_to_sample[future]

                try:
                    voting_result = future.result()
                    voting_results.append((sample_idx, voting_result))

                except Exception:
                    # Create failed voting result
                    failed_result = VotingResult(
                        final_answer=None,
                        vote_distribution={},
                        confidence=0.0,
                        individual_runs=[],
                        voting_strategy=self.config.voting_strategy,
                        coherence_scores=[],
                        evaluation_time=0.0,
                    )
                    voting_results.append((sample_idx, failed_result))

                completed_samples += 1

                if progress_callback:
                    progress = completed_samples / total_samples
                    progress_callback(
                        progress,
                        f"Completed {completed_samples}/{total_samples} samples",
                    )

        # Sort results by original sample order
        voting_results.sort(key=lambda x: x[0])
        ordered_results = [result for _, result in voting_results]

        # Compute statistics
        stats = self.majority_evaluator._compute_dataset_statistics(ordered_results)
        stats.update(
            {
                "total_samples": total_samples,
                "k_runs": self.config.k_runs,
                "voting_strategy": self.config.voting_strategy,
                "parallel_execution": True,
                "max_workers": self.config.max_workers,
            }
        )

        return {
            "voting_results": ordered_results,
            "statistics": stats,
            "configuration": {
                "k_runs": self.config.k_runs,
                "voting_strategy": self.config.voting_strategy,
                "coherence_threshold": self.config.coherence_threshold,
                "parallel_execution": True,
            },
        }

    def _evaluate_single_with_retry(
        self, sample_idx: int, sample: Dict[str, Any]
    ) -> VotingResult:
        """Evaluate single sample with retry logic."""
        max_attempts = (
            self.config.max_retries + 1 if self.config.retry_failed_runs else 1
        )

        for attempt in range(max_attempts):
            try:
                # Check intermediate cache
                cache_key = f"sample_{sample_idx}_attempt_{attempt}"
                if (
                    self.config.cache_intermediate
                    and cache_key in self._intermediate_cache
                ):
                    return self._intermediate_cache[cache_key]

                # Evaluate with majority voting
                voting_result = self.majority_evaluator.evaluate_single_with_voting(
                    sample
                )

                # Cache result if successful
                if (
                    self.config.cache_intermediate
                    and voting_result.final_answer is not None
                ):
                    self._intermediate_cache[cache_key] = voting_result

                # Track retries
                if sample_idx not in self._retry_counts:
                    self._retry_counts[sample_idx] = 0
                self._retry_counts[sample_idx] += attempt

                return voting_result

            except Exception:
                if attempt < max_attempts - 1:
                    # Wait before retry
                    time.sleep(0.1 * (attempt + 1))
                    continue

        # All attempts failed - create failed result
        self._retry_counts[sample_idx] = max_attempts - 1

        return VotingResult(
            final_answer=None,
            vote_distribution={},
            confidence=0.0,
            individual_runs=[],
            voting_strategy=self.config.voting_strategy,
            coherence_scores=[],
            evaluation_time=0.0,
        )

    def create_evaluation_report(self, k_run_result: KRunResult) -> str:
        """Create comprehensive evaluation report."""
        config = k_run_result.configuration
        stats = k_run_result.dataset_results.get("statistics", {})

        report_lines = [
            "# K-Run Benchmark Evaluation Report",
            f"**Configuration:** K={config.k_runs}, Strategy={config.voting_strategy}",
            f"**Execution:** {'Parallel' if config.parallel_execution else 'Sequential'}",
            f"**Total Time:** {k_run_result.total_time:.2f}s",
            "",
            "## Success Metrics",
            f"- **Success Rate:** {k_run_result.get_success_rate():.1%} ({k_run_result.successful_samples}/{k_run_result.successful_samples + k_run_result.failed_samples})",
            f"- **Retry Count:** {k_run_result.retry_count} total retries",
            "",
            "## Voting Statistics",
            f"- **Mean Agreement Rate:** {stats.get('mean_agreement_rate', 0):.3f} ± {stats.get('std_agreement_rate', 0):.3f}",
            f"- **Unanimous Rate:** {stats.get('unanimous_rate', 0):.1%}",
            f"- **Mean Confidence:** {stats.get('mean_confidence', 0):.3f} ± {stats.get('std_confidence', 0):.3f}",
            "",
            "## Coherence Statistics",
            f"- **Mean Coherence:** {stats.get('mean_coherence', 0):.3f} ± {stats.get('std_coherence', 0):.3f}",
            f"- **Total Individual Runs:** {stats.get('total_individual_runs', 0)}",
            f"- **Successful Runs:** {stats.get('successful_runs', 0)}",
            "",
            "## Performance",
            f"- **Time per Sample:** {stats.get('avg_time_per_sample', 0):.3f}s",
            f"- **Time per Run:** {stats.get('avg_time_per_run', 0):.3f}s",
            (
                f"- **Parallelization:** {config.max_workers} workers"
                if config.parallel_execution
                else "- **Execution:** Sequential"
            ),
        ]

        return "\n".join(report_lines)


def create_k_run_evaluator(
    base_evaluator: Any,
    k_runs: int = 5,
    voting_strategy: str = "simple",
    parallel: bool = True,
) -> KRunBenchmarkEvaluator:
    """Convenience factory for creating K-run evaluators."""
    config = KRunConfiguration(
        k_runs=k_runs, voting_strategy=voting_strategy, parallel_execution=parallel
    )

    return KRunBenchmarkEvaluator(base_evaluator, config)
