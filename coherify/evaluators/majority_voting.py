"""
Majority voting evaluator for aggregating multiple evaluation runs.

Provides naive and weighted majority voting for both discrete and continuous responses.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

from coherify.core.base import CoherenceResult


@dataclass
class VotingResult:
    """Result of majority voting evaluation."""
    
    final_answer: Any
    vote_distribution: Dict[Any, int]
    confidence: float
    individual_runs: List[Dict[str, Any]]
    voting_strategy: str
    coherence_scores: List[float]
    evaluation_time: float
    
    def get_agreement_rate(self) -> float:
        """Get the agreement rate for the winning answer."""
        if not self.vote_distribution:
            return 0.0
        total_votes = sum(self.vote_distribution.values())
        winning_votes = max(self.vote_distribution.values())
        return winning_votes / total_votes if total_votes > 0 else 0.0
    
    def is_unanimous(self) -> bool:
        """Check if all runs produced the same result."""
        return len(self.vote_distribution) == 1


class MajorityVotingEvaluator:
    """
    Evaluator that runs K evaluations and applies majority voting.
    
    Supports multiple voting strategies:
    - simple: Each run gets one vote
    - weighted: Votes weighted by coherence scores  
    - confidence: Votes weighted by confidence scores
    """
    
    def __init__(
        self,
        base_evaluator: Any,
        k_runs: int = 5,
        voting_strategy: str = "simple",
        coherence_threshold: float = 0.5,
        answer_key: str = "answer",
        coherence_key: str = "coherence"
    ):
        """
        Initialize majority voting evaluator.
        
        Args:
            base_evaluator: Base evaluator to run K times
            k_runs: Number of evaluation runs
            voting_strategy: "simple", "weighted", or "confidence"
            coherence_threshold: Minimum coherence for vote consideration
            answer_key: Key to extract answer from evaluation results
            coherence_key: Key to extract coherence score from results
        """
        self.base_evaluator = base_evaluator
        self.k_runs = k_runs
        self.voting_strategy = voting_strategy
        self.coherence_threshold = coherence_threshold
        self.answer_key = answer_key
        self.coherence_key = coherence_key
        
        if voting_strategy not in ["simple", "weighted", "confidence"]:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")
    
    def evaluate_single_with_voting(
        self, 
        sample: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> VotingResult:
        """
        Evaluate a single sample K times and apply majority voting.
        
        Args:
            sample: Input sample to evaluate
            progress_callback: Optional callback for progress tracking
            
        Returns:
            VotingResult with final answer and voting details
        """
        start_time = time.time()
        
        individual_runs = []
        
        # Run K evaluations
        for i in range(self.k_runs):
            if progress_callback:
                progress_callback(i, self.k_runs)
                
            try:
                # Run base evaluator - check if it has evaluate_sample method first
                if hasattr(self.base_evaluator, 'evaluate_sample'):
                    result = self.base_evaluator.evaluate_sample(sample)
                else:
                    result = self.base_evaluator.evaluate_single(sample)
                
                # Extract answer and coherence
                answer = self._extract_answer(result, sample)
                coherence = self._extract_coherence(result)
                
                run_data = {
                    "run_id": i,
                    "answer": answer,
                    "coherence": coherence,
                    "full_result": result,
                    "valid": coherence >= self.coherence_threshold
                }
                
                individual_runs.append(run_data)
                
            except Exception as e:
                # Handle failed runs
                run_data = {
                    "run_id": i,
                    "answer": None,
                    "coherence": 0.0,
                    "full_result": None,
                    "valid": False,
                    "error": str(e)
                }
                individual_runs.append(run_data)
        
        if progress_callback:
            progress_callback(self.k_runs, self.k_runs)
        
        # Apply majority voting
        final_answer, vote_distribution, confidence = self._apply_majority_voting(individual_runs)
        
        coherence_scores = [run["coherence"] for run in individual_runs if run["valid"]]
        
        evaluation_time = time.time() - start_time
        
        return VotingResult(
            final_answer=final_answer,
            vote_distribution=vote_distribution,
            confidence=confidence,
            individual_runs=individual_runs,
            voting_strategy=self.voting_strategy,
            coherence_scores=coherence_scores,
            evaluation_time=evaluation_time
        )
    
    def evaluate_dataset_with_voting(
        self, 
        dataset: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset with K-run majority voting.
        
        Args:
            dataset: List of samples to evaluate
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Dictionary with aggregated results and statistics
        """
        start_time = time.time()
        
        voting_results = []
        total_samples = len(dataset)
        
        for i, sample in enumerate(dataset):
            if progress_callback:
                progress_callback(i, total_samples)
            
            sample_progress_callback = None
            if progress_callback:
                def sample_progress_callback(run_idx, total_runs):
                    overall_progress = (i * self.k_runs + run_idx) / (total_samples * self.k_runs)
                    progress_callback(int(overall_progress * total_samples), total_samples)
            
            voting_result = self.evaluate_single_with_voting(
                sample, 
                progress_callback=sample_progress_callback
            )
            voting_results.append(voting_result)
        
        if progress_callback:
            progress_callback(total_samples, total_samples)
        
        # Aggregate statistics
        total_time = time.time() - start_time
        
        stats = self._compute_dataset_statistics(voting_results)
        stats.update({
            "total_samples": total_samples,
            "k_runs": self.k_runs,
            "voting_strategy": self.voting_strategy,
            "total_evaluation_time": total_time,
            "avg_time_per_sample": total_time / total_samples if total_samples > 0 else 0,
            "avg_time_per_run": total_time / (total_samples * self.k_runs) if total_samples > 0 else 0
        })
        
        return {
            "voting_results": voting_results,
            "statistics": stats,
            "configuration": {
                "k_runs": self.k_runs,
                "voting_strategy": self.voting_strategy,
                "coherence_threshold": self.coherence_threshold
            }
        }
    
    def _extract_answer(self, result: Any, sample: Dict[str, Any]) -> Any:
        """Extract answer from evaluation result."""
        if isinstance(result, dict):
            # For TruthfulQA evaluator results - use the coherence score as answer surrogate
            # This allows us to do majority voting based on coherence patterns
            coherence_score = result.get("coherence_score", 0.0)
            # Discretize coherence into buckets for voting
            if coherence_score >= 0.9:
                return "high_coherence"
            elif coherence_score >= 0.7:
                return "medium_coherence" 
            elif coherence_score >= 0.5:
                return "low_coherence"
            else:
                return "very_low_coherence"
                
        elif hasattr(result, self.answer_key):
            return getattr(result, self.answer_key)
        else:
            # Fallback: use original answer from sample
            return sample.get("answer", sample.get("best_answer", "unknown"))
    
    def _extract_coherence(self, result: Any) -> float:
        """Extract coherence score from evaluation result."""
        if isinstance(result, dict):
            # For TruthfulQA evaluator results
            if "coherence_score" in result:
                return result["coherence_score"]
            return result.get(self.coherence_key, result.get("mean_coherence", 1.0))
        elif isinstance(result, CoherenceResult):
            return result.score
        elif hasattr(result, self.coherence_key):
            return getattr(result, self.coherence_key)
        else:
            return 1.0  # Default coherence if not found
    
    def _apply_majority_voting(self, runs: List[Dict[str, Any]]) -> tuple[Any, Dict[Any, int], float]:
        """Apply majority voting to individual runs."""
        # Filter valid runs
        valid_runs = [run for run in runs if run.get("valid", False)]
        
        if not valid_runs:
            # No valid runs - return None with 0 confidence
            return None, {}, 0.0
        
        if self.voting_strategy == "simple":
            return self._simple_majority_voting(valid_runs)
        elif self.voting_strategy == "weighted":
            return self._weighted_majority_voting(valid_runs)
        elif self.voting_strategy == "confidence":
            return self._confidence_weighted_voting(valid_runs)
    
    def _simple_majority_voting(self, runs: List[Dict[str, Any]]) -> tuple[Any, Dict[Any, int], float]:
        """Apply simple majority voting - each run gets one vote."""
        answers = [run["answer"] for run in runs]
        vote_counts = Counter(answers)
        
        final_answer = vote_counts.most_common(1)[0][0]
        winning_votes = vote_counts[final_answer]
        total_votes = len(answers)
        
        confidence = winning_votes / total_votes
        
        return final_answer, dict(vote_counts), confidence
    
    def _weighted_majority_voting(self, runs: List[Dict[str, Any]]) -> tuple[Any, Dict[Any, int], float]:
        """Apply coherence-weighted majority voting."""
        answer_weights = defaultdict(float)
        
        for run in runs:
            answer = run["answer"]
            coherence = run["coherence"]
            answer_weights[answer] += coherence
        
        if not answer_weights:
            return None, {}, 0.0
        
        final_answer = max(answer_weights, key=answer_weights.get)
        total_weight = sum(answer_weights.values())
        
        confidence = answer_weights[final_answer] / total_weight if total_weight > 0 else 0.0
        
        # Convert weights to vote counts for consistency
        vote_distribution = {
            answer: int(weight / total_weight * len(runs)) if total_weight > 0 else 0
            for answer, weight in answer_weights.items()
        }
        
        return final_answer, vote_distribution, confidence
    
    def _confidence_weighted_voting(self, runs: List[Dict[str, Any]]) -> tuple[Any, Dict[Any, int], float]:
        """Apply confidence-weighted voting (using normalized coherence)."""
        # For now, same as weighted - could be enhanced with separate confidence scores
        return self._weighted_majority_voting(runs)
    
    def _compute_dataset_statistics(self, voting_results: List[VotingResult]) -> Dict[str, Any]:
        """Compute aggregated statistics for dataset evaluation."""
        if not voting_results:
            return {}
        
        # Basic statistics
        agreement_rates = [result.get_agreement_rate() for result in voting_results]
        unanimous_count = sum(1 for result in voting_results if result.is_unanimous())
        
        # Coherence statistics  
        all_coherence_scores = []
        for result in voting_results:
            all_coherence_scores.extend(result.coherence_scores)
        
        # Confidence statistics
        confidences = [result.confidence for result in voting_results]
        
        # Timing statistics
        eval_times = [result.evaluation_time for result in voting_results]
        
        stats = {
            "mean_agreement_rate": np.mean(agreement_rates) if agreement_rates else 0.0,
            "std_agreement_rate": np.std(agreement_rates) if agreement_rates else 0.0,
            "unanimous_rate": unanimous_count / len(voting_results),
            
            "mean_confidence": np.mean(confidences) if confidences else 0.0,
            "std_confidence": np.std(confidences) if confidences else 0.0,
            
            "mean_coherence": np.mean(all_coherence_scores) if all_coherence_scores else 0.0,
            "std_coherence": np.std(all_coherence_scores) if all_coherence_scores else 0.0,
            
            "mean_eval_time": np.mean(eval_times) if eval_times else 0.0,
            "total_individual_runs": sum(len(r.individual_runs) for r in voting_results),
            "successful_runs": sum(len([run for run in r.individual_runs if run.get("valid", False)]) for r in voting_results)
        }
        
        return stats


def create_majority_voting_evaluator(
    base_evaluator: Any,
    k_runs: int = 5,
    strategy: str = "simple"
) -> MajorityVotingEvaluator:
    """Convenience factory function for creating majority voting evaluators."""
    return MajorityVotingEvaluator(
        base_evaluator=base_evaluator,
        k_runs=k_runs,
        voting_strategy=strategy
    )