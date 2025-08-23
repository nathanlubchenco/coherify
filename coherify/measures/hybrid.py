"""
Hybrid coherence measure combining multiple approaches.
This provides a more robust coherence evaluation than any single method.
"""

import time
from typing import Optional, Dict, Any
import numpy as np

from coherify.core.base import CoherenceMeasure, CoherenceResult, PropositionSet
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.entailment import EntailmentCoherence
from coherify.core.types import Encoder, NLIModel


class HybridCoherence(CoherenceMeasure):
    """
    Hybrid coherence combining semantic similarity and logical entailment.

    This measure provides a comprehensive evaluation by considering:
    1. Semantic coherence (topical consistency via embeddings)
    2. Logical coherence (entailment relationships via NLI)
    3. Optional consistency (agreement across multiple samples)
    """

    def __init__(
        self,
        semantic_weight: float = 0.4,
        entailment_weight: float = 0.6,
        encoder: Optional[Encoder] = None,
        nli_model: Optional[NLIModel] = None,
        semantic_config: Optional[Dict[str, Any]] = None,
        entailment_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid coherence measure.

        Args:
            semantic_weight: Weight for semantic coherence component
            entailment_weight: Weight for entailment coherence component
            encoder: Text encoder for semantic coherence
            nli_model: NLI model for entailment coherence
            semantic_config: Configuration for semantic coherence
            entailment_config: Configuration for entailment coherence
        """
        if abs(semantic_weight + entailment_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.semantic_weight = semantic_weight
        self.entailment_weight = entailment_weight

        # Initialize component measures
        semantic_config = semantic_config or {}
        entailment_config = entailment_config or {}

        self.semantic_measure = SemanticCoherence(encoder=encoder, **semantic_config)

        self.entailment_measure = EntailmentCoherence(
            nli_model=nli_model, **entailment_config
        )

    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute hybrid coherence for a proposition set."""
        start_time = time.time()

        if len(prop_set) < 2:
            return CoherenceResult(
                score=1.0,
                measure_name="HybridCoherence",
                details={
                    "num_propositions": len(prop_set),
                    "reason": "insufficient_propositions",
                },
                computation_time=time.time() - start_time,
            )

        # Compute component coherences
        semantic_result = self.semantic_measure.compute(prop_set)
        entailment_result = self.entailment_measure.compute(prop_set)

        # Combine scores
        combined_score = (
            self.semantic_weight * semantic_result.score
            + self.entailment_weight * entailment_result.score
        )

        # Calculate agreement between measures
        score_difference = abs(semantic_result.score - entailment_result.score)
        agreement = 1.0 - score_difference  # Higher when scores are similar

        # Determine dominant component
        if semantic_result.score > entailment_result.score:
            dominant_component = "semantic"
        elif entailment_result.score > semantic_result.score:
            dominant_component = "entailment"
        else:
            dominant_component = "balanced"

        computation_time = time.time() - start_time

        return CoherenceResult(
            score=combined_score,
            measure_name="HybridCoherence",
            details={
                "component_scores": {
                    "semantic": semantic_result.score,
                    "entailment": entailment_result.score,
                },
                "component_weights": {
                    "semantic": self.semantic_weight,
                    "entailment": self.entailment_weight,
                },
                "component_details": {
                    "semantic": semantic_result.details,
                    "entailment": entailment_result.details,
                },
                "component_times": {
                    "semantic": semantic_result.computation_time,
                    "entailment": entailment_result.computation_time,
                },
                "agreement": agreement,
                "score_difference": score_difference,
                "dominant_component": dominant_component,
                "num_propositions": len(prop_set),
            },
            computation_time=computation_time,
        )

    def compute_pairwise(self, prop1, prop2) -> float:
        """Compute hybrid coherence between two propositions."""
        semantic_score = self.semantic_measure.compute_pairwise(prop1, prop2)
        entailment_score = self.entailment_measure.compute_pairwise(prop1, prop2)

        return (
            self.semantic_weight * semantic_score
            + self.entailment_weight * entailment_score
        )

    def analyze_components(self, prop_set: PropositionSet) -> Dict[str, Any]:
        """Detailed analysis of how different components contribute to coherence."""
        result = self.compute(prop_set)
        details = result.details

        analysis = {
            "overall_score": result.score,
            "semantic_contribution": self.semantic_weight
            * details["component_scores"]["semantic"],
            "entailment_contribution": self.entailment_weight
            * details["component_scores"]["entailment"],
            "component_agreement": details["agreement"],
            "recommendations": [],
        }

        # Generate recommendations based on component analysis
        semantic_score = details["component_scores"]["semantic"]
        entailment_score = details["component_scores"]["entailment"]

        if semantic_score < 0.3:
            analysis["recommendations"].append(
                "Low semantic coherence: Propositions may be off-topic or unrelated"
            )

        if entailment_score < 0.3:
            analysis["recommendations"].append(
                "Low logical coherence: Propositions may contradict each other"
            )

        if details["agreement"] < 0.5:
            analysis["recommendations"].append(
                "Low agreement between measures: Content may have mixed semantic/logical issues"
            )

        if semantic_score > 0.7 and entailment_score > 0.7:
            analysis["recommendations"].append(
                "High coherence: Content is both topically consistent and logically sound"
            )

        return analysis


class AdaptiveHybridCoherence(HybridCoherence):
    """
    Adaptive hybrid coherence that adjusts weights based on content characteristics.

    This variant automatically adjusts the importance of semantic vs. entailment
    coherence based on the type and characteristics of the propositions.
    """

    def __init__(
        self,
        base_semantic_weight: float = 0.4,
        base_entailment_weight: float = 0.6,
        **kwargs,
    ):
        """
        Initialize adaptive hybrid coherence.

        Args:
            base_semantic_weight: Base weight for semantic coherence
            base_entailment_weight: Base weight for entailment coherence
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            semantic_weight=base_semantic_weight,
            entailment_weight=base_entailment_weight,
            **kwargs,
        )
        self.base_semantic_weight = base_semantic_weight
        self.base_entailment_weight = base_entailment_weight

    def _adapt_weights(self, prop_set: PropositionSet) -> tuple[float, float]:
        """Adapt weights based on proposition characteristics."""
        propositions = [p.text for p in prop_set.propositions]

        # Handle empty proposition sets
        if not propositions:
            return self.base_semantic_weight, self.base_entailment_weight

        # Calculate text characteristics
        avg_length = np.mean([len(p.split()) for p in propositions])
        all_words = " ".join(propositions).split()
        if not all_words:
            vocabulary_diversity = 0.0
        else:
            vocabulary_diversity = len(set(" ".join(propositions).lower().split())) / len(all_words)

        # Adjust weights based on characteristics
        semantic_weight = self.base_semantic_weight
        entailment_weight = self.base_entailment_weight

        # For short, diverse propositions, favor entailment
        if avg_length < 10 and vocabulary_diversity > 0.7:
            entailment_weight += 0.1
            semantic_weight -= 0.1

        # For long, repetitive propositions, favor semantic
        elif avg_length > 20 and vocabulary_diversity < 0.3:
            semantic_weight += 0.1
            entailment_weight -= 0.1

        # Ensure weights sum to 1
        total = semantic_weight + entailment_weight
        return semantic_weight / total, entailment_weight / total

    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute adaptive hybrid coherence."""
        # Adapt weights for this specific proposition set
        orig_semantic_weight = self.semantic_weight
        orig_entailment_weight = self.entailment_weight

        self.semantic_weight, self.entailment_weight = self._adapt_weights(prop_set)

        # Compute coherence with adapted weights
        result = super().compute(prop_set)

        # Add adaptation info to details
        result.details["adapted_weights"] = {
            "semantic": self.semantic_weight,
            "entailment": self.entailment_weight,
        }
        result.details["original_weights"] = {
            "semantic": orig_semantic_weight,
            "entailment": orig_entailment_weight,
        }
        result.details["weight_adaptation"] = {
            "semantic_change": self.semantic_weight - orig_semantic_weight,
            "entailment_change": self.entailment_weight - orig_entailment_weight,
        }

        # Restore original weights
        self.semantic_weight = orig_semantic_weight
        self.entailment_weight = orig_entailment_weight

        return result
