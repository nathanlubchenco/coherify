"""
Semantic coherence measure using embedding similarity.
This is a probability-free alternative to traditional coherence measures.
"""

import time
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from coherify.core.base import (
    CoherenceMeasure,
    CoherenceResult,
    PropositionSet,
    Proposition,
)
from coherify.core.types import Encoder


class SemanticCoherence(CoherenceMeasure):
    """
    Semantic coherence based on embedding similarity.

    Computes coherence as the average pairwise cosine similarity
    between proposition embeddings. Higher values indicate more
    topical coherence.
    """

    def __init__(
        self,
        encoder: Optional[Encoder] = None,
        similarity_threshold: float = 0.0,
        aggregation: str = "mean",
    ):
        """
        Initialize semantic coherence measure.

        Args:
            encoder: Text encoder model. If None, uses sentence-transformers default.
            similarity_threshold: Minimum similarity to consider coherent
            aggregation: How to aggregate pairwise similarities ("mean", "min", "median")
        """
        self.encoder = encoder or self._get_default_encoder()
        self.similarity_threshold = similarity_threshold
        self.aggregation = aggregation

        if aggregation not in ["mean", "min", "median"]:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def _get_default_encoder(self) -> Encoder:
        """Get default sentence transformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            from coherify.utils.transformers_utils import suppress_transformer_warnings

            with suppress_transformer_warnings():
                return SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for default encoder. "
                "Install with: pip install sentence-transformers"
            )

    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute semantic coherence for a proposition set."""
        start_time = time.time()

        if len(prop_set) < 2:
            return CoherenceResult(
                score=1.0,  # Single proposition is perfectly coherent with itself
                measure_name="SemanticCoherence",
                details={
                    "num_propositions": len(prop_set),
                    "reason": "insufficient_propositions",
                },
                computation_time=time.time() - start_time,
            )

        # Extract texts
        texts = [prop.text for prop in prop_set.propositions]

        # Get embeddings
        from coherify.utils.transformers_utils import suppress_transformer_warnings

        with suppress_transformer_warnings():
            embeddings = self.encoder.encode(texts)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Extract upper triangle (excluding diagonal)
        n = len(texts)
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_similarities = similarity_matrix[upper_triangle_indices]

        # Aggregate similarities
        if self.aggregation == "mean":
            score = float(np.mean(pairwise_similarities))
        elif self.aggregation == "min":
            score = float(np.min(pairwise_similarities))
        elif self.aggregation == "median":
            score = float(np.median(pairwise_similarities))

        # Apply threshold
        coherent_pairs = np.sum(pairwise_similarities > self.similarity_threshold)

        computation_time = time.time() - start_time

        return CoherenceResult(
            score=score,
            measure_name="SemanticCoherence",
            details={
                "similarity_matrix": similarity_matrix.tolist(),
                "pairwise_similarities": pairwise_similarities.tolist(),
                "aggregation": self.aggregation,
                "similarity_threshold": self.similarity_threshold,
                "coherent_pairs": int(coherent_pairs),
                "total_pairs": len(pairwise_similarities),
                "num_propositions": n,
            },
            computation_time=computation_time,
        )

    def compute_pairwise(self, prop1: Proposition, prop2: Proposition) -> float:
        """Compute semantic similarity between two propositions."""
        texts = [prop1.text, prop2.text]
        from coherify.utils.transformers_utils import suppress_transformer_warnings

        with suppress_transformer_warnings():
            embeddings = self.encoder.encode(texts)

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
