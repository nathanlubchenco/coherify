"""
Coherence-guided reranking for RAG systems.
Uses coherence measures to rerank retrieved passages for better context selection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time

from coherify.core.base import CoherenceMeasure, PropositionSet
from coherify.measures.hybrid import HybridCoherence


@dataclass
class PassageCandidate:
    """Represents a retrieved passage candidate for reranking."""

    text: str
    score: float  # Original retrieval score
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RerankedResult:
    """Result of coherence-guided reranking."""

    passages: List[PassageCandidate]
    coherence_scores: List[float]
    original_scores: List[float]
    reranking_metadata: Dict[str, Any]


class CoherenceReranker:
    """
    Reranks retrieved passages using coherence measures.

    Combines retrieval scores with coherence scores to improve
    passage selection for RAG systems.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        coherence_weight: float = 0.5,
        retrieval_weight: float = 0.5,
        normalize_scores: bool = True,
        min_coherence_threshold: float = 0.0,
    ):
        """
        Initialize coherence reranker.

        Args:
            coherence_measure: Coherence measure to use (default: HybridCoherence)
            coherence_weight: Weight for coherence scores (0-1)
            retrieval_weight: Weight for retrieval scores (0-1)
            normalize_scores: Whether to normalize scores before combining
            min_coherence_threshold: Minimum coherence score to include passage
        """
        self.coherence_measure = coherence_measure or HybridCoherence()
        self.coherence_weight = coherence_weight
        self.retrieval_weight = retrieval_weight
        self.normalize_scores = normalize_scores
        self.min_coherence_threshold = min_coherence_threshold

        # Validate weights
        if abs(coherence_weight + retrieval_weight - 1.0) > 1e-6:
            raise ValueError("coherence_weight + retrieval_weight must equal 1.0")

    def rerank(
        self,
        query: str,
        passages: List[PassageCandidate],
        top_k: Optional[int] = None,
        context: Optional[str] = None,
    ) -> RerankedResult:
        """
        Rerank passages using coherence scores.

        Args:
            query: The original query/question
            passages: List of passage candidates to rerank
            top_k: Number of top passages to return (default: all)
            context: Additional context for coherence evaluation

        Returns:
            RerankedResult with reranked passages
        """
        if not passages:
            return RerankedResult([], [], [], {})

        start_time = time.time()

        # Compute coherence scores
        coherence_scores = self._compute_coherence_scores(query, passages, context)

        # Get original retrieval scores
        original_scores = [p.score for p in passages]

        # Combine scores
        combined_scores = self._combine_scores(original_scores, coherence_scores)

        # Apply coherence threshold filter
        valid_indices = [
            i
            for i, score in enumerate(coherence_scores)
            if score >= self.min_coherence_threshold
        ]

        if not valid_indices:
            # If no passages meet threshold, return empty result
            return RerankedResult(
                [],
                [],
                [],
                {
                    "filtered_out": len(passages),
                    "reason": "no_passages_meet_coherence_threshold",
                },
            )

        # Sort by combined scores (descending)
        passage_score_pairs = [(i, combined_scores[i]) for i in valid_indices]
        passage_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Select top_k
        if top_k is not None:
            passage_score_pairs = passage_score_pairs[:top_k]

        # Build result
        reranked_passages = [passages[i] for i, _ in passage_score_pairs]
        reranked_coherence_scores = [
            coherence_scores[i] for i, _ in passage_score_pairs
        ]
        reranked_original_scores = [original_scores[i] for i, _ in passage_score_pairs]

        metadata = {
            "reranking_time": time.time() - start_time,
            "original_count": len(passages),
            "reranked_count": len(reranked_passages),
            "coherence_weight": self.coherence_weight,
            "retrieval_weight": self.retrieval_weight,
            "filtered_count": len(passages) - len(valid_indices),
            "score_statistics": {
                "coherence_mean": np.mean(coherence_scores),
                "coherence_std": np.std(coherence_scores),
                "original_mean": np.mean(original_scores),
                "original_std": np.std(original_scores),
            },
        }

        return RerankedResult(
            passages=reranked_passages,
            coherence_scores=reranked_coherence_scores,
            original_scores=reranked_original_scores,
            reranking_metadata=metadata,
        )

    def _compute_coherence_scores(
        self,
        query: str,
        passages: List[PassageCandidate],
        context: Optional[str] = None,
    ) -> List[float]:
        """Compute coherence scores for passages with respect to query."""
        coherence_scores = []

        for passage in passages:
            # Create proposition set from query and passage
            prop_set = PropositionSet.from_qa_pair(query, passage.text)
            if context:
                prop_set.context = context

            # Compute coherence
            result = self.coherence_measure.compute(prop_set)
            coherence_scores.append(result.score)

        return coherence_scores

    def _combine_scores(
        self, retrieval_scores: List[float], coherence_scores: List[float]
    ) -> List[float]:
        """Combine retrieval and coherence scores."""
        if self.normalize_scores:
            # Normalize both score types to [0, 1]
            retrieval_scores = self._normalize_scores(retrieval_scores)
            coherence_scores = self._normalize_scores(coherence_scores)

        # Weighted combination
        combined = [
            self.retrieval_weight * r_score + self.coherence_weight * c_score
            for r_score, c_score in zip(retrieval_scores, coherence_scores)
        ]

        return combined

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)  # All equal scores

        return [(score - min_score) / (max_score - min_score) for score in scores]


class CoherenceRAG:
    """
    Complete RAG system with coherence-guided reranking.

    Integrates retrieval, coherence-based reranking, and context selection
    for improved RAG performance.
    """

    def __init__(
        self,
        reranker: Optional[CoherenceReranker] = None,
        max_context_length: int = 4000,
        passage_separator: str = "\n\n---\n\n",
    ):
        """
        Initialize CoherenceRAG system.

        Args:
            reranker: Coherence reranker to use
            max_context_length: Maximum length of combined context
            passage_separator: Separator between passages in context
        """
        self.reranker = reranker or CoherenceReranker()
        self.max_context_length = max_context_length
        self.passage_separator = passage_separator

    def retrieve_and_rerank(
        self,
        query: str,
        retrieval_function: Callable[[str], List[PassageCandidate]],
        top_k: int = 5,
        context: Optional[str] = None,
    ) -> RerankedResult:
        """
        Retrieve passages and rerank using coherence.

        Args:
            query: Query to retrieve passages for
            retrieval_function: Function that takes query and returns PassageCandidates
            top_k: Number of top passages to return
            context: Additional context for coherence evaluation

        Returns:
            RerankedResult with coherence-reranked passages
        """
        # Retrieve initial candidates
        candidates = retrieval_function(query)

        # Rerank using coherence
        reranked_result = self.reranker.rerank(
            query=query, passages=candidates, top_k=top_k, context=context
        )

        return reranked_result

    def build_context(
        self, reranked_result: RerankedResult, include_scores: bool = False
    ) -> str:
        """
        Build context string from reranked passages.

        Args:
            reranked_result: Result from reranking
            include_scores: Whether to include scores in context

        Returns:
            Combined context string
        """
        if not reranked_result.passages:
            return ""

        context_parts = []
        current_length = 0

        for i, passage in enumerate(reranked_result.passages):
            # Build passage text
            passage_text = passage.text

            if include_scores:
                coherence_score = reranked_result.coherence_scores[i]
                original_score = reranked_result.original_scores[i]
                passage_text = (
                    f"[Coherence: {coherence_score:.3f}, Retrieval: {original_score:.3f}]\n"
                    f"{passage_text}"
                )

            # Check if adding this passage would exceed max length
            additional_length = len(passage_text) + len(self.passage_separator)
            if (
                current_length + additional_length > self.max_context_length
                and context_parts
            ):
                break

            context_parts.append(passage_text)
            current_length += additional_length

        return self.passage_separator.join(context_parts)

    def evaluate_coherence_improvement(
        self,
        query: str,
        original_passages: List[PassageCandidate],
        reranked_result: RerankedResult,
    ) -> Dict[str, Any]:
        """
        Evaluate how much coherence improved through reranking.

        Args:
            query: Original query
            original_passages: Original passage order
            reranked_result: Result after reranking

        Returns:
            Evaluation metrics
        """
        # Compute coherence of original top-k vs reranked top-k
        top_k = len(reranked_result.passages)
        original_top_k = original_passages[:top_k]

        # Original coherence
        original_prop_set = PropositionSet.from_qa_pair(
            query, self.passage_separator.join([p.text for p in original_top_k])
        )
        original_coherence = self.reranker.coherence_measure.compute(original_prop_set)

        # Reranked coherence
        reranked_prop_set = PropositionSet.from_qa_pair(
            query,
            self.passage_separator.join([p.text for p in reranked_result.passages]),
        )
        reranked_coherence = self.reranker.coherence_measure.compute(reranked_prop_set)

        improvement = reranked_coherence.score - original_coherence.score

        return {
            "original_coherence": original_coherence.score,
            "reranked_coherence": reranked_coherence.score,
            "improvement": improvement,
            "relative_improvement": improvement / max(original_coherence.score, 1e-8),
            "passages_reordered": sum(
                1
                for i, p in enumerate(reranked_result.passages)
                if i >= len(original_passages) or p.text != original_passages[i].text
            ),
            "top_k_size": top_k,
        }


class BatchCoherenceReranker:
    """
    Batch processing version of coherence reranker for efficiency.
    """

    def __init__(self, base_reranker: CoherenceReranker):
        self.base_reranker = base_reranker

    def rerank_batch(
        self,
        queries: List[str],
        passage_lists: List[List[PassageCandidate]],
        top_k: Optional[int] = None,
        contexts: Optional[List[str]] = None,
    ) -> List[RerankedResult]:
        """
        Rerank multiple query-passage sets in batch.

        Args:
            queries: List of queries
            passage_lists: List of passage candidate lists
            top_k: Number of top passages per query
            contexts: Optional contexts for each query

        Returns:
            List of reranked results
        """
        if contexts is None:
            contexts = [None] * len(queries)

        results = []
        for query, passages, context in zip(queries, passage_lists, contexts):
            result = self.base_reranker.rerank(
                query=query, passages=passages, top_k=top_k, context=context
            )
            results.append(result)

        return results
