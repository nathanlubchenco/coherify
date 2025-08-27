"""
Response selection mechanisms for K-pass evaluation.

This module implements Stage 2 (majority voting) and Stage 3 (coherence-based)
selection mechanisms for choosing the best response from K generated candidates.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from coherify.core.base import CoherenceMeasure
from coherify.generation.model_runner import GenerationResult
from coherify.measures import SemanticCoherence


@dataclass
class SelectionResult:
    """Result from response selection."""

    selected_response: str
    selected_index: int
    method: str  # "majority_vote", "coherence", "single"
    confidence: float
    metadata: Dict[str, Any]


class MajorityVotingSelector:
    """
    Stage 2: Majority voting selector for K-pass evaluation.

    Selects the most common response among K candidates.
    This is the baseline K-pass approach.
    """

    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize majority voting selector.

        Args:
            similarity_threshold: Threshold for considering responses "same"
        """
        self.similarity_threshold = similarity_threshold

        # Use semantic similarity to group similar responses
        try:
            self.semantic_measure = SemanticCoherence()
        except:
            self.semantic_measure = None

    def select(self, responses: List[GenerationResult]) -> SelectionResult:
        """
        Select response by majority voting.

        Args:
            responses: List of K generated responses

        Returns:
            SelectionResult with the selected response
        """
        if not responses:
            return SelectionResult(
                selected_response="",
                selected_index=-1,
                method="majority_vote",
                confidence=0.0,
                metadata={"error": "No responses provided"},
            )

        if len(responses) == 1:
            return SelectionResult(
                selected_response=responses[0].text,
                selected_index=0,
                method="single",
                confidence=1.0,
                metadata={},
            )

        # Extract text from responses
        texts = [r.text for r in responses]

        # Group similar responses
        groups = self._group_similar_responses(texts)

        # Find the largest group (majority)
        largest_group = max(groups, key=len)
        group_size = len(largest_group)

        # Select the first response from the majority group
        selected_idx = largest_group[0]
        selected_text = texts[selected_idx]

        # Calculate confidence as proportion of votes
        confidence = group_size / len(responses)

        return SelectionResult(
            selected_response=selected_text,
            selected_index=selected_idx,
            method="majority_vote",
            confidence=confidence,
            metadata={
                "group_size": group_size,
                "num_groups": len(groups),
                "total_responses": len(responses),
            },
        )

    def _group_similar_responses(self, texts: List[str]) -> List[List[int]]:
        """
        Group similar responses together.

        Args:
            texts: List of response texts

        Returns:
            List of groups, where each group is a list of indices
        """
        if self.semantic_measure is None:
            # Fallback to exact matching
            Counter(texts)
            groups = []
            seen = set()

            for i, text in enumerate(texts):
                if i not in seen:
                    group = [j for j, t in enumerate(texts) if t == text]
                    groups.append(group)
                    seen.update(group)

            return groups

        # Use semantic similarity
        groups = []
        ungrouped = set(range(len(texts)))

        while ungrouped:
            # Start a new group with an ungrouped response
            seed_idx = ungrouped.pop()
            group = [seed_idx]
            seed_text = texts[seed_idx]

            # Find similar responses
            for idx in list(ungrouped):
                similarity = self._compute_similarity(seed_text, texts[idx])
                if similarity >= self.similarity_threshold:
                    group.append(idx)
                    ungrouped.remove(idx)

            groups.append(group)

        return groups

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            from coherify.core.base import Proposition, PropositionSet

            # Create proposition sets
            prop1 = PropositionSet(
                propositions=[Proposition(text=text1)], context={"type": "response"}
            )
            prop2 = PropositionSet(
                propositions=[Proposition(text=text2)], context={"type": "response"}
            )

            # Compute pairwise similarity
            embeddings1 = self.semantic_measure._get_embeddings([text1])
            embeddings2 = self.semantic_measure._get_embeddings([text2])

            # Cosine similarity
            similarity = np.dot(embeddings1[0], embeddings2[0]) / (
                np.linalg.norm(embeddings1[0]) * np.linalg.norm(embeddings2[0])
            )

            return float(similarity)

        except:
            # Fallback to exact match
            return 1.0 if text1 == text2 else 0.0


class CoherenceSelector:
    """
    Stage 3: Coherence-based selector for K-pass evaluation.

    Selects the response with highest coherence score.
    This is our research contribution - using coherence for selection.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        question: Optional[str] = None,
    ):
        """
        Initialize coherence selector.

        Args:
            coherence_measure: Coherence measure to use
            question: Original question for context
        """
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.question = question

    def select(self, responses: List[GenerationResult]) -> SelectionResult:
        """
        Select response with highest coherence.

        Args:
            responses: List of K generated responses

        Returns:
            SelectionResult with the selected response
        """
        if not responses:
            return SelectionResult(
                selected_response="",
                selected_index=-1,
                method="coherence",
                confidence=0.0,
                metadata={"error": "No responses provided"},
            )

        if len(responses) == 1:
            return SelectionResult(
                selected_response=responses[0].text,
                selected_index=0,
                method="single",
                confidence=1.0,
                metadata={},
            )

        # Calculate coherence for each response
        coherence_scores = []
        for response in responses:
            score = self._compute_coherence(response.text)
            coherence_scores.append(score)

        # Select response with highest coherence
        best_idx = np.argmax(coherence_scores)
        best_score = coherence_scores[best_idx]
        best_response = responses[best_idx].text

        # Calculate confidence based on score distribution
        scores_array = np.array(coherence_scores)
        if len(scores_array) > 1:
            # Confidence is how much better the best is than the mean
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            if std_score > 0:
                z_score = (best_score - mean_score) / std_score
                confidence = min(1.0, abs(z_score) / 3.0)  # Normalize z-score
            else:
                confidence = 0.5  # All same score
        else:
            confidence = best_score

        return SelectionResult(
            selected_response=best_response,
            selected_index=best_idx,
            method="coherence",
            confidence=confidence,
            metadata={
                "coherence_scores": coherence_scores,
                "best_score": best_score,
                "mean_score": float(np.mean(scores_array)),
                "std_score": float(np.std(scores_array)),
            },
        )

    def _compute_coherence(self, text: str) -> float:
        """
        Compute coherence score for a response.

        Args:
            text: Response text

        Returns:
            Coherence score
        """
        try:
            from coherify.core.base import Proposition, PropositionSet

            # Create proposition set
            propositions = []

            # Add question as context if available
            if self.question:
                propositions.append(
                    Proposition(text=self.question, metadata={"type": "question"})
                )

            # Add response
            propositions.append(Proposition(text=text, metadata={"type": "response"}))

            prop_set = PropositionSet(
                propositions=propositions, context={"evaluation": "response_coherence"}
            )

            # Compute coherence
            result = self.coherence_measure.compute(prop_set)
            return result.score

        except Exception as e:
            print(f"⚠️  Coherence computation failed: {e}")
            return 0.0


class StageComparator:
    """
    Compares performance across the three stages of evaluation.

    Stage 1: Single response (baseline)
    Stage 2: K-pass with majority voting
    Stage 3: K-pass with coherence selection
    """

    def __init__(self, evaluator):
        """
        Initialize comparator.

        Args:
            evaluator: Benchmark evaluator to use
        """
        self.evaluator = evaluator

    def compare_stages(
        self,
        samples: List[Dict[str, Any]],
        stage1_predictions: List[str],
        stage2_predictions: List[str],
        stage3_predictions: List[str],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare performance across all three stages.

        Args:
            samples: Benchmark samples
            stage1_predictions: Single response predictions
            stage2_predictions: Majority voting predictions
            stage3_predictions: Coherence selection predictions
            verbose: Print progress

        Returns:
            Comparison results
        """
        if verbose:
            print("\n📊 Comparing Performance Across Stages")
            print("=" * 50)

        # Evaluate each stage
        if verbose:
            print("\n⏳ Evaluating Stage 1 (Baseline)...")
        stage1_results = self.evaluator.evaluate_dataset(
            stage1_predictions, samples, verbose=verbose
        )

        if verbose:
            print("\n⏳ Evaluating Stage 2 (Majority Voting)...")
        stage2_results = self.evaluator.evaluate_dataset(
            stage2_predictions, samples, verbose=verbose
        )

        if verbose:
            print("\n⏳ Evaluating Stage 3 (Coherence Selection)...")
        stage3_results = self.evaluator.evaluate_dataset(
            stage3_predictions, samples, verbose=verbose
        )

        # Extract primary metrics
        stage1_score = stage1_results.truthful_score
        stage2_score = stage2_results.truthful_score
        stage3_score = stage3_results.truthful_score

        # Calculate improvements
        stage2_improvement = stage2_score - stage1_score
        stage3_improvement = stage3_score - stage1_score
        coherence_advantage = stage3_score - stage2_score

        if verbose:
            print(f"\n📈 Results:")
            print(f"  Stage 1 (Baseline):     {stage1_score:.1%}")
            print(
                f"  Stage 2 (Majority Vote): {stage2_score:.1%} ({stage2_improvement:+.1%})"
            )
            print(
                f"  Stage 3 (Coherence):     {stage3_score:.1%} ({stage3_improvement:+.1%})"
            )
            print(f"\n  Coherence Advantage:     {coherence_advantage:+.1%}")

            # Check if results match expectations
            if stage3_score > stage2_score and stage2_score > stage1_score:
                print("\n✅ Success! Coherence > Majority > Baseline")
            elif stage3_score > stage1_score:
                print("\n⚠️  Partial success: Coherence > Baseline")
            else:
                print("\n❌ Unexpected: Coherence did not improve over baseline")

        return {
            "stage1": {"score": stage1_score, "results": stage1_results},
            "stage2": {
                "score": stage2_score,
                "results": stage2_results,
                "improvement": stage2_improvement,
            },
            "stage3": {
                "score": stage3_score,
                "results": stage3_results,
                "improvement": stage3_improvement,
            },
            "coherence_advantage": coherence_advantage,
            "success": stage3_score > stage2_score and stage2_score > stage1_score,
        }
