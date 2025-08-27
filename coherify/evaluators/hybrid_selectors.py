"""
Hybrid selection methods combining coherence and consistency checking.

This module implements advanced selection strategies that combine
our coherence measures with consistency checking methods like SelfCheckGPT.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from coherify.core.base import Proposition, PropositionSet
from coherify.generation.model_runner import GenerationResult
from coherify.measures.semantic import SemanticCoherence


@dataclass
class HybridSelectionResult:
    """Result from hybrid selection with detailed scores."""

    selected_response: str
    selected_index: int
    confidence: float
    metadata: Dict[str, Any]
    coherence_scores: List[float]
    consistency_scores: List[float]
    hybrid_scores: List[float]
    alpha: float  # Weight for coherence vs consistency
    method: str = "hybrid_coherence_consistency"  # Default method name


class HybridCoherenceConsistencySelector:
    """
    Combines coherence measures with consistency checking for better selection.

    This selector uses both:
    1. Coherence scores (how well ideas connect)
    2. Consistency scores (how similar responses are to each other)

    The intuition is that truthful responses should be both coherent
    AND consistent across multiple generations.
    """

    def __init__(
        self,
        coherence_measure=None,
        consistency_method: str = "semantic",
        alpha: float = 0.5,
        use_selfcheck: bool = False,
    ):
        """
        Initialize hybrid selector.

        Args:
            coherence_measure: Coherence measure to use (default: SemanticCoherence)
            consistency_method: Method for consistency checking
            alpha: Weight for coherence (0=pure consistency, 1=pure coherence)
            use_selfcheck: Whether to use SelfCheckGPT (requires installation)
        """
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.consistency_method = consistency_method
        self.alpha = alpha
        self.use_selfcheck = use_selfcheck

        if use_selfcheck:
            try:
                import torch
                from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.selfcheck_nli = SelfCheckNLI(device=device)
            except ImportError:
                print(
                    "⚠️ SelfCheckGPT not installed. Install with: pip install selfcheckgpt"
                )
                self.use_selfcheck = False
                self.selfcheck_nli = None
        else:
            self.selfcheck_nli = None

    def select(
        self, responses: List[GenerationResult], question: Optional[str] = None
    ) -> HybridSelectionResult:
        """
        Select best response using hybrid coherence-consistency scoring.

        Args:
            responses: List of generated responses
            question: Optional question for context

        Returns:
            HybridSelectionResult with selected response and scores
        """
        if not responses:
            raise ValueError("No responses to select from")

        if len(responses) == 1:
            return HybridSelectionResult(
                selected_response=responses[0].text,
                selected_index=0,
                confidence=1.0,
                metadata={"method": "single_response"},
                coherence_scores=[1.0],
                consistency_scores=[1.0],
                hybrid_scores=[1.0],
                alpha=self.alpha,
            )

        # Compute coherence scores
        coherence_scores = self._compute_coherence_scores(responses, question)

        # Compute consistency scores
        if self.use_selfcheck and self.selfcheck_nli:
            consistency_scores = self._compute_selfcheck_consistency(responses)
        else:
            consistency_scores = self._compute_semantic_consistency(responses)

        # Combine scores
        hybrid_scores = [
            self.alpha * coh
            + (1 - self.alpha) * (1 - cons)  # Note: consistency is inverted
            for coh, cons in zip(coherence_scores, consistency_scores)
        ]

        # Select best response
        best_idx = np.argmax(hybrid_scores)

        return HybridSelectionResult(
            selected_response=responses[best_idx].text,
            selected_index=best_idx,
            confidence=hybrid_scores[best_idx],
            metadata={
                "method": "hybrid_coherence_consistency",
                "consistency_method": (
                    "selfcheck_nli" if self.use_selfcheck else "semantic"
                ),
                "alpha": self.alpha,
                "num_responses": len(responses),
            },
            coherence_scores=coherence_scores,
            consistency_scores=consistency_scores,
            hybrid_scores=hybrid_scores,
            alpha=self.alpha,
        )

    def _compute_coherence_scores(
        self, responses: List[GenerationResult], question: Optional[str] = None
    ) -> List[float]:
        """Compute coherence score for each response."""
        scores = []

        for response in responses:
            # Create proposition set
            props = []
            if question:
                props.append(Proposition(text=question, metadata={"type": "question"}))
            props.append(Proposition(text=response.text, metadata={"type": "response"}))

            prop_set = PropositionSet(
                propositions=props, context={"evaluation": "response_coherence"}
            )

            # Compute coherence
            try:
                result = self.coherence_measure.compute(prop_set)
                scores.append(result.score)
            except Exception as e:
                print(f"⚠️ Coherence computation failed: {e}")
                scores.append(0.5)  # Default middle score

        return scores

    def _compute_semantic_consistency(
        self, responses: List[GenerationResult]
    ) -> List[float]:
        """
        Compute semantic consistency scores.

        Lower scores mean more consistent (less likely to be hallucination).
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # Get embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [r.text for r in responses]
        embeddings = model.encode(texts)

        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)

        # For each response, compute average similarity to others
        consistency_scores = []
        for i in range(len(responses)):
            # Average similarity to all other responses
            similarities = [
                similarity_matrix[i, j] for j in range(len(responses)) if i != j
            ]
            avg_similarity = np.mean(similarities) if similarities else 0.5

            # Convert to inconsistency score (0 = consistent, 1 = inconsistent)
            inconsistency = 1 - avg_similarity
            consistency_scores.append(inconsistency)

        return consistency_scores

    def _compute_selfcheck_consistency(
        self, responses: List[GenerationResult]
    ) -> List[float]:
        """
        Compute consistency using SelfCheckGPT NLI method.

        Returns inconsistency scores (0 = consistent, 1 = likely hallucination).
        """
        if not self.selfcheck_nli:
            return self._compute_semantic_consistency(responses)

        texts = [r.text for r in responses]
        scores = []

        # For each response, check consistency against others
        for i, text in enumerate(texts):
            # Use other responses as samples
            samples = [texts[j] for j in range(len(texts)) if i != j]

            if samples:
                # Split into sentences (simple version)
                sentences = text.split(". ")
                sentences = [s.strip() for s in sentences if s.strip()]

                if sentences:
                    # Get consistency scores from SelfCheckGPT
                    sent_scores = self.selfcheck_nli.predict(
                        sentences=sentences, sampled_passages=samples
                    )

                    # Average across sentences
                    avg_score = np.mean(sent_scores) if len(sent_scores) > 0 else 0.5
                    scores.append(avg_score)
                else:
                    scores.append(0.5)  # Default
            else:
                scores.append(0.5)  # Default if no samples

        return scores


class AdaptiveTemperatureSelector:
    """
    Selector that adapts temperature based on response characteristics.

    This selector analyzes the diversity and quality of responses
    generated at different temperatures to find optimal selection.
    """

    def __init__(self, base_selector=None, analyze_entropy: bool = True):
        """
        Initialize adaptive temperature selector.

        Args:
            base_selector: Base selector to use (default: HybridSelector)
            analyze_entropy: Whether to analyze response entropy
        """
        self.base_selector = base_selector or HybridCoherenceConsistencySelector()
        self.analyze_entropy = analyze_entropy

    def select_with_temperature_analysis(
        self, responses: List[GenerationResult], question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select response and analyze temperature effects.

        Returns:
            Dictionary with selection result and temperature analysis
        """
        # Group responses by temperature
        temp_groups = {}
        for r in responses:
            temp = r.temperature
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(r)

        # Analyze each temperature group
        temp_analysis = {}
        for temp, group in temp_groups.items():
            if self.analyze_entropy:
                entropy = self._compute_entropy(group)
            else:
                entropy = 0.0

            temp_analysis[temp] = {
                "count": len(group),
                "entropy": entropy,
                "avg_length": np.mean([len(r.text) for r in group]),
            }

        # Select best response
        selection_result = self.base_selector.select(responses, question)

        # Find optimal temperature range
        best_response_temp = responses[selection_result.selected_index].temperature

        return {
            "selection": selection_result,
            "temperature_analysis": temp_analysis,
            "optimal_temperature": best_response_temp,
            "recommendation": self._get_temperature_recommendation(
                temp_analysis, best_response_temp
            ),
        }

    def _compute_entropy(self, responses: List[GenerationResult]) -> float:
        """Compute entropy of response set."""
        from collections import Counter

        # Simple word-level entropy
        all_words = []
        for r in responses:
            words = r.text.lower().split()
            all_words.extend(words)

        word_counts = Counter(all_words)
        total = len(all_words)

        entropy = 0.0
        for count in word_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)

        return entropy

    def _get_temperature_recommendation(
        self, temp_analysis: Dict[float, Dict], best_temp: float
    ) -> str:
        """Get recommendation for temperature settings."""

        # Analyze patterns
        temps = sorted(temp_analysis.keys())
        [temp_analysis[t]["entropy"] for t in temps]

        if best_temp <= min(temps) + 0.1:
            return f"Lower temperatures ({best_temp:.1f}) producing best results. Consider range {max(0.1, best_temp-0.2):.1f}-{best_temp+0.1:.1f}"
        elif best_temp >= max(temps) - 0.1:
            return f"Higher temperatures ({best_temp:.1f}) producing best results. Consider range {best_temp-0.1:.1f}-{min(2.0, best_temp+0.2):.1f}"
        else:
            return f"Mid-range temperature ({best_temp:.1f}) optimal. Current range {min(temps):.1f}-{max(temps):.1f} is appropriate"


# Convenience function
def create_best_selector(
    use_selfcheck: bool = False, alpha: float = 0.6
) -> HybridCoherenceConsistencySelector:
    """
    Create the best performing selector based on available packages.

    Args:
        use_selfcheck: Whether to attempt using SelfCheckGPT
        alpha: Weight for coherence vs consistency

    Returns:
        Configured selector
    """
    return HybridCoherenceConsistencySelector(
        coherence_measure=SemanticCoherence(),
        consistency_method="selfcheck_nli" if use_selfcheck else "semantic",
        alpha=alpha,
        use_selfcheck=use_selfcheck,
    )
