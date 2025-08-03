"""
Coherence-guided beam search for text generation.
Integrates coherence scoring into beam search to maintain logical consistency.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time

from coherify.core.base import (
    CoherenceMeasure,
    PropositionSet,
)
from coherify.measures.hybrid import HybridCoherence


@dataclass
class GenerationCandidate:
    """Represents a candidate during beam search generation."""

    text: str
    tokens: List[str]
    log_prob: float
    coherence_score: float
    combined_score: float
    parent_id: Optional[int] = None
    generation_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeamSearchResult:
    """Result of coherence-guided beam search."""

    best_candidate: GenerationCandidate
    all_candidates: List[GenerationCandidate]
    search_time: float
    total_steps: int
    coherence_evaluations: int
    beam_statistics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoherenceGuidedBeamSearch:
    """
    Beam search that incorporates coherence scoring for better generation quality.

    Balances language model probability with coherence to existing context.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        coherence_weight: float = 0.3,
        lm_weight: float = 0.7,
        beam_size: int = 5,
        max_length: int = 100,
        min_coherence_threshold: float = 0.1,
        diversity_penalty: float = 0.0,
    ):
        """
        Initialize coherence-guided beam search.

        Args:
            coherence_measure: Coherence measure for evaluation
            coherence_weight: Weight for coherence scores (0-1)
            lm_weight: Weight for language model scores (0-1)
            beam_size: Number of beams to maintain
            max_length: Maximum generation length
            min_coherence_threshold: Minimum coherence to keep candidate
            diversity_penalty: Penalty for repetitive content
        """
        self.coherence_measure = coherence_measure or HybridCoherence()
        self.coherence_weight = coherence_weight
        self.lm_weight = lm_weight
        self.beam_size = beam_size
        self.max_length = max_length
        self.min_coherence_threshold = min_coherence_threshold
        self.diversity_penalty = diversity_penalty

        # Validate weights
        if abs(coherence_weight + lm_weight - 1.0) > 1e-6:
            raise ValueError("coherence_weight + lm_weight must equal 1.0")

        # Statistics
        self.coherence_evaluations = 0
        self.candidates_pruned = 0

    def search(
        self,
        context: str,
        generation_function: Callable[[str, int], List[Tuple[str, float]]],
        prompt: Optional[str] = None,
    ) -> BeamSearchResult:
        """
        Perform coherence-guided beam search.

        Args:
            context: Context for coherence evaluation
            generation_function: Function that generates next token candidates
                                Returns list of (token, log_prob) tuples
            prompt: Optional prompt to start generation

        Returns:
            BeamSearchResult with best candidate and statistics
        """
        start_time = time.time()

        # Initialize beam
        initial_text = prompt or ""
        initial_candidate = GenerationCandidate(
            text=initial_text,
            tokens=initial_text.split() if initial_text else [],
            log_prob=0.0,
            coherence_score=(
                1.0
                if not initial_text
                else self._evaluate_coherence(context, initial_text)
            ),
            combined_score=0.0,
            generation_step=0,
        )
        initial_candidate.combined_score = self._compute_combined_score(
            initial_candidate
        )

        beam = [initial_candidate]
        all_candidates = [initial_candidate]
        step = 0

        # Beam search loop
        while step < self.max_length and beam:
            step += 1
            new_candidates = []

            # Expand each beam candidate
            for candidate in beam:
                if self._is_complete(candidate):
                    new_candidates.append(candidate)
                    continue

                # Generate next token candidates
                next_tokens = generation_function(candidate.text, self.beam_size * 2)

                # Create new candidates
                for token, log_prob in next_tokens:
                    new_text = self._append_token(candidate.text, token)
                    new_tokens = candidate.tokens + [token]
                    new_log_prob = candidate.log_prob + log_prob

                    # Evaluate coherence
                    coherence_score = self._evaluate_coherence(context, new_text)
                    self.coherence_evaluations += 1

                    # Apply coherence threshold
                    if coherence_score < self.min_coherence_threshold:
                        self.candidates_pruned += 1
                        continue

                    # Create new candidate
                    new_candidate = GenerationCandidate(
                        text=new_text,
                        tokens=new_tokens,
                        log_prob=new_log_prob,
                        coherence_score=coherence_score,
                        combined_score=0.0,
                        parent_id=id(candidate),
                        generation_step=step,
                    )

                    # Apply diversity penalty
                    if self.diversity_penalty > 0:
                        diversity_penalty = self._compute_diversity_penalty(
                            new_candidate, beam
                        )
                        new_candidate.log_prob -= diversity_penalty

                    # Compute combined score
                    new_candidate.combined_score = self._compute_combined_score(
                        new_candidate
                    )

                    new_candidates.append(new_candidate)
                    all_candidates.append(new_candidate)

            # Select top candidates for next beam
            if new_candidates:
                # Sort by combined score
                new_candidates.sort(key=lambda x: x.combined_score, reverse=True)
                beam = new_candidates[: self.beam_size]
            else:
                break

        # Select best candidate
        best_candidate = (
            max(beam, key=lambda x: x.combined_score) if beam else initial_candidate
        )

        search_time = time.time() - start_time

        # Compute statistics
        beam_stats = self._compute_beam_statistics(all_candidates, beam)

        return BeamSearchResult(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            search_time=search_time,
            total_steps=step,
            coherence_evaluations=self.coherence_evaluations,
            beam_statistics=beam_stats,
            metadata={
                "candidates_pruned": self.candidates_pruned,
                "final_beam_size": len(beam),
                "coherence_weight": self.coherence_weight,
                "lm_weight": self.lm_weight,
            },
        )

    def _evaluate_coherence(self, context: str, generated_text: str) -> float:
        """Evaluate coherence between context and generated text."""
        if not generated_text.strip():
            return 1.0

        # Create proposition set
        prop_set = PropositionSet.from_qa_pair(context, generated_text)

        if len(prop_set.propositions) <= 1:
            return 1.0

        result = self.coherence_measure.compute(prop_set)
        return result.score

    def _compute_combined_score(self, candidate: GenerationCandidate) -> float:
        """Compute combined score from language model and coherence scores."""
        # Normalize log probability (rough approximation)
        normalized_lm_score = np.exp(candidate.log_prob / max(len(candidate.tokens), 1))
        normalized_lm_score = min(normalized_lm_score, 1.0)

        combined = (
            self.lm_weight * normalized_lm_score
            + self.coherence_weight * candidate.coherence_score
        )

        return combined

    def _append_token(self, text: str, token: str) -> str:
        """Append token to text with appropriate spacing."""
        if not text:
            return token

        # Simple heuristic for spacing
        if token.startswith("'") or token in ".,!?;:":
            return text + token
        else:
            return text + " " + token

    def _is_complete(self, candidate: GenerationCandidate) -> bool:
        """Check if candidate is complete (e.g., ends with period)."""
        if not candidate.text:
            return False

        # Simple completion heuristics
        return (
            candidate.text.endswith(".")
            or candidate.text.endswith("!")
            or candidate.text.endswith("?")
            or len(candidate.tokens) >= self.max_length
        )

    def _compute_diversity_penalty(
        self, candidate: GenerationCandidate, beam: List[GenerationCandidate]
    ) -> float:
        """Compute penalty for lack of diversity."""
        if not beam:
            return 0.0

        # Count token overlaps with beam candidates
        candidate_tokens = set(candidate.tokens)
        overlap_counts = []

        for beam_candidate in beam:
            beam_tokens = set(beam_candidate.tokens)
            overlap = len(candidate_tokens & beam_tokens)
            overlap_ratio = overlap / max(len(candidate_tokens), 1)
            overlap_counts.append(overlap_ratio)

        # Penalty based on maximum overlap
        max_overlap = max(overlap_counts) if overlap_counts else 0.0
        penalty = self.diversity_penalty * max_overlap

        return penalty

    def _compute_beam_statistics(
        self,
        all_candidates: List[GenerationCandidate],
        final_beam: List[GenerationCandidate],
    ) -> Dict[str, Any]:
        """Compute beam search statistics."""
        if not all_candidates:
            return {}

        coherence_scores = [c.coherence_score for c in all_candidates]
        lm_scores = [c.log_prob for c in all_candidates]
        combined_scores = [c.combined_score for c in all_candidates]

        stats = {
            "total_candidates": len(all_candidates),
            "final_beam_size": len(final_beam),
            "coherence_stats": {
                "mean": np.mean(coherence_scores),
                "std": np.std(coherence_scores),
                "min": np.min(coherence_scores),
                "max": np.max(coherence_scores),
            },
            "lm_stats": {
                "mean": np.mean(lm_scores),
                "std": np.std(lm_scores),
                "min": np.min(lm_scores),
                "max": np.max(lm_scores),
            },
            "combined_stats": {
                "mean": np.mean(combined_scores),
                "std": np.std(combined_scores),
                "min": np.min(combined_scores),
                "max": np.max(combined_scores),
            },
        }

        # Final beam analysis
        if final_beam:
            final_coherence = [c.coherence_score for c in final_beam]
            final_combined = [c.combined_score for c in final_beam]

            stats["final_beam_stats"] = {
                "coherence_range": [np.min(final_coherence), np.max(final_coherence)],
                "combined_range": [np.min(final_combined), np.max(final_combined)],
                "best_coherence": np.max(final_coherence),
                "best_combined": np.max(final_combined),
            }

        return stats


class CoherenceBeamSearchDecoder:
    """
    Higher-level decoder that integrates with language models.

    Provides easy interface for coherence-guided generation.
    """

    def __init__(
        self,
        beam_search: CoherenceGuidedBeamSearch,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        """
        Initialize decoder.

        Args:
            beam_search: Configured beam search instance
            tokenizer: Tokenizer for text processing
            model: Language model for token generation
        """
        self.beam_search = beam_search
        self.tokenizer = tokenizer
        self.model = model

    def generate(
        self,
        context: str,
        prompt: str = "",
        num_return_sequences: int = 1,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Generate text using coherence-guided beam search.

        Args:
            context: Context for coherence evaluation
            prompt: Starting prompt for generation
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        # Create mock generation function if model not available
        generation_fn = self._create_generation_function(temperature)

        # Perform beam search
        result = self.beam_search.search(
            context=context, generation_function=generation_fn, prompt=prompt
        )

        # Return top sequences
        candidates = sorted(
            result.all_candidates, key=lambda x: x.combined_score, reverse=True
        )

        # Filter complete sequences
        complete_candidates = [
            c for c in candidates if self.beam_search._is_complete(c)
        ]

        if not complete_candidates:
            complete_candidates = candidates[:num_return_sequences]

        return [c.text for c in complete_candidates[:num_return_sequences]]

    def _create_generation_function(self, temperature: float) -> Callable:
        """Create generation function for beam search."""

        def mock_generation_function(
            current_text: str, num_candidates: int
        ) -> List[Tuple[str, float]]:
            """Mock generation function for demonstration."""
            # In practice, this would use a real language model

            # Simple word bank based on context
            word_bank = [
                ("learning", -0.5),
                ("machine", -0.7),
                ("data", -0.6),
                ("algorithm", -0.8),
                ("model", -0.6),
                ("training", -0.7),
                ("neural", -0.8),
                ("network", -0.8),
                ("deep", -0.9),
                ("artificial", -1.0),
                ("intelligence", -0.9),
                ("patterns", -0.8),
                ("analysis", -0.7),
                ("prediction", -0.8),
                ("accuracy", -0.9),
                ("performance", -0.8),
                (".", -0.3),
                ("and", -0.4),
                ("the", -0.3),
                ("is", -0.4),
                ("can", -0.5),
                ("will", -0.5),
                ("uses", -0.6),
                ("helps", -0.6),
            ]

            # Add some randomness with temperature
            import random

            random.seed(42)  # For reproducible results

            # Adjust probabilities with temperature
            adjusted_probs = []
            for word, log_prob in word_bank:
                adjusted_log_prob = log_prob / temperature
                adjusted_probs.append((word, adjusted_log_prob))

            # Sample candidates
            selected = random.sample(
                adjusted_probs, min(num_candidates, len(adjusted_probs))
            )

            return selected

        return mock_generation_function

    def analyze_generation(self, result: BeamSearchResult) -> Dict[str, Any]:
        """Analyze generation result for insights."""
        analysis = {
            "generation_quality": {
                "best_coherence": result.best_candidate.coherence_score,
                "best_combined": result.best_candidate.combined_score,
                "generation_length": len(result.best_candidate.tokens),
                "search_efficiency": result.coherence_evaluations
                / max(result.total_steps, 1),
            },
            "search_statistics": result.beam_statistics,
            "timing": {
                "total_time": result.search_time,
                "time_per_step": result.search_time / max(result.total_steps, 1),
                "evaluations_per_second": result.coherence_evaluations
                / max(result.search_time, 1e-6),
            },
            "coherence_impact": self._analyze_coherence_impact(result),
        }

        return analysis

    def _analyze_coherence_impact(self, result: BeamSearchResult) -> Dict[str, Any]:
        """Analyze impact of coherence on generation quality."""
        candidates = result.all_candidates

        if len(candidates) < 2:
            return {"insufficient_data": True}

        # Compare coherence vs LM-only ranking
        lm_ranking = sorted(candidates, key=lambda x: x.log_prob, reverse=True)
        coherence_ranking = sorted(
            candidates, key=lambda x: x.coherence_score, reverse=True
        )
        combined_ranking = sorted(
            candidates, key=lambda x: x.combined_score, reverse=True
        )

        # Measure ranking correlation
        def ranking_correlation(list1, list2):
            """Simple ranking correlation measure."""
            if len(list1) != len(list2):
                return 0.0

            rank_diff = 0
            for i, item in enumerate(list1):
                try:
                    j = list2.index(item)
                    rank_diff += abs(i - j)
                except ValueError:
                    rank_diff += len(list2)  # Maximum penalty

            max_diff = len(list1) * (len(list1) - 1) / 2
            return 1.0 - (rank_diff / max_diff) if max_diff > 0 else 1.0

        impact = {
            "lm_coherence_correlation": ranking_correlation(
                lm_ranking, coherence_ranking
            ),
            "coherence_improvement": {
                "best_lm_coherence": lm_ranking[0].coherence_score if lm_ranking else 0,
                "best_combined_coherence": (
                    combined_ranking[0].coherence_score if combined_ranking else 0
                ),
                "coherence_gain": (
                    (
                        combined_ranking[0].coherence_score
                        - lm_ranking[0].coherence_score
                    )
                    if lm_ranking and combined_ranking
                    else 0
                ),
            },
            "diversity_analysis": {
                "unique_tokens": len(
                    set([token for c in candidates for token in c.tokens])
                ),
                "avg_length": np.mean([len(c.tokens) for c in candidates]),
                "length_variance": np.var([len(c.tokens) for c in candidates]),
            },
        }

        return impact


class AdaptiveCoherenceBeamSearch(CoherenceGuidedBeamSearch):
    """
    Adaptive beam search that adjusts coherence weight based on generation quality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_coherence_weight = self.coherence_weight
        self.adaptation_rate = 0.1
        self.quality_history = []

    def search(self, *args, **kwargs) -> BeamSearchResult:
        """Perform adaptive beam search with dynamic weight adjustment."""
        result = super().search(*args, **kwargs)

        # Adapt weights based on result quality
        self._adapt_weights(result)

        return result

    def _adapt_weights(self, result: BeamSearchResult):
        """Adapt coherence weight based on generation quality."""
        # Simple adaptation strategy
        best_coherence = result.best_candidate.coherence_score
        avg_coherence = np.mean([c.coherence_score for c in result.all_candidates])

        quality_metric = (best_coherence + avg_coherence) / 2
        self.quality_history.append(quality_metric)

        # Adjust weight if quality is declining
        if len(self.quality_history) >= 3:
            recent_trend = (
                np.mean(self.quality_history[-3:])
                - np.mean(self.quality_history[-6:-3])
                if len(self.quality_history) >= 6
                else 0
            )

            if recent_trend < -0.1:  # Quality declining
                self.coherence_weight = min(
                    0.8, self.coherence_weight + self.adaptation_rate
                )
                self.lm_weight = 1.0 - self.coherence_weight
            elif recent_trend > 0.1:  # Quality improving
                self.coherence_weight = max(
                    0.1, self.coherence_weight - self.adaptation_rate / 2
                )
                self.lm_weight = 1.0 - self.coherence_weight
