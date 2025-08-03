"""
Coherence-guided retrieval strategies.
Implements retrieval methods that consider coherence during the retrieval process.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from coherify.core.base import CoherenceMeasure, PropositionSet
from coherify.measures.semantic import SemanticCoherence
from coherify.rag.reranking import PassageCandidate


@dataclass
class CoherenceGuidedQuery:
    """Enhanced query with coherence context."""

    original_query: str
    expanded_query: str
    coherence_keywords: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CoherenceGuidedRetriever:
    """
    Retriever that uses coherence to guide retrieval strategy.

    Expands queries and filters results based on coherence patterns
    to improve retrieval quality for RAG systems.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        query_expansion_strategy: str = "semantic",
        coherence_filter_threshold: float = 0.3,
        max_expansion_terms: int = 5,
    ):
        """
        Initialize coherence-guided retriever.

        Args:
            coherence_measure: Coherence measure for evaluation
            query_expansion_strategy: Strategy for query expansion ('semantic', 'keyword', 'hybrid')
            coherence_filter_threshold: Minimum coherence score for retrieved passages
            max_expansion_terms: Maximum number of expansion terms to add
        """
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.query_expansion_strategy = query_expansion_strategy
        self.coherence_filter_threshold = coherence_filter_threshold
        self.max_expansion_terms = max_expansion_terms

    def expand_query(
        self,
        query: str,
        context: Optional[str] = None,
        domain_knowledge: Optional[List[str]] = None,
    ) -> CoherenceGuidedQuery:
        """
        Expand query using coherence-guided strategies.

        Args:
            query: Original query
            context: Additional context for expansion
            domain_knowledge: Domain-specific terms/concepts

        Returns:
            CoherenceGuidedQuery with expanded version
        """
        if self.query_expansion_strategy == "semantic":
            return self._semantic_expansion(query, context, domain_knowledge)
        elif self.query_expansion_strategy == "keyword":
            return self._keyword_expansion(query, context, domain_knowledge)
        elif self.query_expansion_strategy == "hybrid":
            return self._hybrid_expansion(query, context, domain_knowledge)
        else:
            raise ValueError(
                f"Unknown expansion strategy: {self.query_expansion_strategy}"
            )

    def _semantic_expansion(
        self,
        query: str,
        context: Optional[str] = None,
        domain_knowledge: Optional[List[str]] = None,
    ) -> CoherenceGuidedQuery:
        """Expand query using semantic similarity."""
        # Simple semantic expansion - in practice would use embeddings
        expansion_terms = []

        # Extract key concepts from query
        query_words = query.lower().split()

        # Add related terms based on semantic patterns
        semantic_expansions = {
            "machine learning": [
                "AI",
                "artificial intelligence",
                "algorithms",
                "neural networks",
            ],
            "climate change": [
                "global warming",
                "greenhouse gases",
                "carbon emissions",
            ],
            "quantum computing": ["qubits", "superposition", "quantum mechanics"],
            "blockchain": ["cryptocurrency", "distributed ledger", "smart contracts"],
        }

        for concept, expansions in semantic_expansions.items():
            if any(word in query.lower() for word in concept.split()):
                expansion_terms.extend(expansions[: self.max_expansion_terms])

        # Add domain knowledge if provided
        if domain_knowledge:
            # Use coherence to find most relevant domain terms
            domain_scores = []
            for term in domain_knowledge:
                test_prop_set = PropositionSet.from_qa_pair(query, term)
                if len(test_prop_set.propositions) > 1:
                    result = self.coherence_measure.compute(test_prop_set)
                    domain_scores.append((term, result.score))

            # Add top coherent domain terms
            domain_scores.sort(key=lambda x: x[1], reverse=True)
            expansion_terms.extend(
                [term for term, _ in domain_scores[: self.max_expansion_terms]]
            )

        # Build expanded query
        unique_expansions = list(set(expansion_terms))[: self.max_expansion_terms]
        expanded_query = query
        if unique_expansions:
            expanded_query = f"{query} {' '.join(unique_expansions)}"

        return CoherenceGuidedQuery(
            original_query=query,
            expanded_query=expanded_query,
            coherence_keywords=unique_expansions,
            metadata={
                "expansion_method": "semantic",
                "context_used": context is not None,
                "domain_terms_used": len(domain_knowledge) if domain_knowledge else 0,
            },
        )

    def _keyword_expansion(
        self,
        query: str,
        context: Optional[str] = None,
        domain_knowledge: Optional[List[str]] = None,
    ) -> CoherenceGuidedQuery:
        """Expand query using keyword-based strategies."""
        # Extract important terms and add synonyms/related concepts
        import re

        # Simple keyword extraction (in practice would use NLP)
        words = re.findall(r"\b\w+\b", query.lower())

        # Common synonyms and related terms
        keyword_expansions = {
            "learn": ["study", "understand", "analyze"],
            "find": ["discover", "locate", "identify"],
            "explain": ["describe", "clarify", "elaborate"],
            "compare": ["contrast", "evaluate", "assess"],
            "impact": ["effect", "influence", "consequence"],
        }

        expansion_terms = []
        for word in words:
            if word in keyword_expansions:
                expansion_terms.extend(keyword_expansions[word])

        # Add context terms if provided
        if context:
            context_words = re.findall(r"\b\w+\b", context.lower())
            # Add important context words (longer than 4 chars)
            context_terms = [w for w in context_words if len(w) > 4][:3]
            expansion_terms.extend(context_terms)

        unique_expansions = list(set(expansion_terms))[: self.max_expansion_terms]
        expanded_query = (
            f"{query} {' '.join(unique_expansions)}" if unique_expansions else query
        )

        return CoherenceGuidedQuery(
            original_query=query,
            expanded_query=expanded_query,
            coherence_keywords=unique_expansions,
            metadata={"expansion_method": "keyword"},
        )

    def _hybrid_expansion(
        self,
        query: str,
        context: Optional[str] = None,
        domain_knowledge: Optional[List[str]] = None,
    ) -> CoherenceGuidedQuery:
        """Combine semantic and keyword expansion strategies."""
        # Get both expansions
        semantic_expanded = self._semantic_expansion(query, context, domain_knowledge)
        keyword_expanded = self._keyword_expansion(query, context, domain_knowledge)

        # Combine expansion terms
        all_terms = (
            semantic_expanded.coherence_keywords + keyword_expanded.coherence_keywords
        )
        unique_terms = list(set(all_terms))[: self.max_expansion_terms]

        expanded_query = f"{query} {' '.join(unique_terms)}" if unique_terms else query

        return CoherenceGuidedQuery(
            original_query=query,
            expanded_query=expanded_query,
            coherence_keywords=unique_terms,
            metadata={"expansion_method": "hybrid"},
        )

    def retrieve_with_coherence_filtering(
        self,
        query: str,
        retrieval_function: Callable[[str], List[PassageCandidate]],
        max_candidates: int = 20,
        context: Optional[str] = None,
    ) -> List[PassageCandidate]:
        """
        Retrieve passages with coherence-based filtering.

        Args:
            query: Query to retrieve for
            retrieval_function: Base retrieval function
            max_candidates: Maximum candidates to consider
            context: Additional context

        Returns:
            Filtered list of coherent passage candidates
        """
        # Expand query
        expanded_query = self.expand_query(query, context)

        # Retrieve with expanded query
        candidates = retrieval_function(expanded_query.expanded_query)

        # Limit initial candidates
        candidates = candidates[:max_candidates]

        # Filter by coherence
        filtered_candidates = []
        for candidate in candidates:
            # Compute coherence between query and passage
            prop_set = PropositionSet.from_qa_pair(query, candidate.text)

            if len(prop_set.propositions) > 1:
                coherence_result = self.coherence_measure.compute(prop_set)
                if coherence_result.score >= self.coherence_filter_threshold:
                    # Update candidate with coherence score
                    candidate.metadata["coherence_score"] = coherence_result.score
                    candidate.metadata["original_retrieval_score"] = candidate.score
                    filtered_candidates.append(candidate)
            else:
                # Single proposition - keep with neutral coherence
                candidate.metadata["coherence_score"] = 0.5
                candidate.metadata["original_retrieval_score"] = candidate.score
                filtered_candidates.append(candidate)

        return filtered_candidates

    def iterative_coherence_retrieval(
        self,
        query: str,
        retrieval_function: Callable[[str], List[PassageCandidate]],
        max_iterations: int = 3,
        convergence_threshold: float = 0.05,
    ) -> Tuple[List[PassageCandidate], Dict[str, Any]]:
        """
        Iteratively refine retrieval using coherence feedback.

        Args:
            query: Original query
            retrieval_function: Base retrieval function
            max_iterations: Maximum number of refinement iterations
            convergence_threshold: Coherence improvement threshold for convergence

        Returns:
            Tuple of (final_candidates, iteration_metadata)
        """
        iteration_data = []
        current_query = query
        best_coherence = 0.0

        for iteration in range(max_iterations):
            # Retrieve candidates
            candidates = self.retrieve_with_coherence_filtering(
                current_query, retrieval_function
            )

            if not candidates:
                break

            # Compute overall coherence of top candidates
            top_candidates = candidates[:5]  # Top 5 for evaluation
            combined_text = " ".join([c.text for c in top_candidates])

            prop_set = PropositionSet.from_qa_pair(query, combined_text)
            if len(prop_set.propositions) > 1:
                coherence_result = self.coherence_measure.compute(prop_set)
                current_coherence = coherence_result.score
            else:
                current_coherence = 0.5

            # Track iteration data
            iteration_data.append(
                {
                    "iteration": iteration,
                    "coherence_score": current_coherence,
                    "num_candidates": len(candidates),
                    "query_used": current_query,
                }
            )

            # Check for convergence
            improvement = current_coherence - best_coherence
            if improvement < convergence_threshold and iteration > 0:
                break

            best_coherence = max(best_coherence, current_coherence)

            # Refine query for next iteration using top candidates
            if iteration < max_iterations - 1:
                current_query = self._refine_query_from_results(query, top_candidates)

        metadata = {
            "iterations": iteration_data,
            "final_coherence": best_coherence,
            "converged": improvement < convergence_threshold,
            "total_iterations": len(iteration_data),
        }

        return candidates, metadata

    def _refine_query_from_results(
        self, original_query: str, top_candidates: List[PassageCandidate]
    ) -> str:
        """Refine query based on top coherent results."""
        # Extract key terms from top candidates
        candidate_texts = [c.text for c in top_candidates]
        combined_text = " ".join(candidate_texts)

        # Simple term extraction (in practice would use more sophisticated NLP)
        import re
        from collections import Counter

        words = re.findall(r"\b\w+\b", combined_text.lower())
        # Filter out common words and keep longer terms
        meaningful_words = [
            w
            for w in words
            if len(w) > 4
            and w
            not in {
                "this",
                "that",
                "with",
                "from",
                "they",
                "have",
                "been",
                "will",
                "would",
                "could",
                "should",
                "there",
                "where",
                "when",
                "what",
                "which",
            }
        ]

        # Get most common meaningful terms
        term_counts = Counter(meaningful_words)
        top_terms = [term for term, count in term_counts.most_common(3)]

        # Add top terms to original query
        refined_query = f"{original_query} {' '.join(top_terms)}"
        return refined_query

    def evaluate_retrieval_coherence(
        self, query: str, candidates: List[PassageCandidate], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate coherence quality of retrieved candidates.

        Args:
            query: Original query
            candidates: Retrieved candidates
            top_k: Number of top candidates to evaluate

        Returns:
            Evaluation metrics
        """
        if not candidates:
            return {"error": "no_candidates"}

        top_candidates = candidates[:top_k]

        # Individual coherence scores
        individual_scores = []
        for candidate in top_candidates:
            prop_set = PropositionSet.from_qa_pair(query, candidate.text)
            if len(prop_set.propositions) > 1:
                result = self.coherence_measure.compute(prop_set)
                individual_scores.append(result.score)
            else:
                individual_scores.append(0.5)

        # Combined coherence (all top candidates together)
        combined_text = " ".join([c.text for c in top_candidates])
        combined_prop_set = PropositionSet.from_qa_pair(query, combined_text)
        if len(combined_prop_set.propositions) > 1:
            combined_result = self.coherence_measure.compute(combined_prop_set)
            combined_coherence = combined_result.score
        else:
            combined_coherence = 0.5

        # Diversity measure (how different are the top candidates)
        if hasattr(self.coherence_measure, "encoder"):
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                candidate_texts = [c.text for c in top_candidates]
                embeddings = self.coherence_measure.encoder.encode(candidate_texts)
                similarity_matrix = cosine_similarity(embeddings)

                # Average pairwise similarity (higher = less diverse)
                upper_triangle = similarity_matrix[
                    np.triu_indices_from(similarity_matrix, k=1)
                ]
                diversity = (
                    1.0 - np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
                )
            except Exception:
                diversity = None
        else:
            diversity = None

        return {
            "individual_coherence_scores": individual_scores,
            "mean_individual_coherence": np.mean(individual_scores),
            "std_individual_coherence": np.std(individual_scores),
            "combined_coherence": combined_coherence,
            "diversity_score": diversity,
            "num_candidates_evaluated": len(top_candidates),
            "coherence_measure": self.coherence_measure.__class__.__name__,
        }
