#!/usr/bin/env python3
"""
RAG Integration demonstration of the coherify library.

This example demonstrates:
1. Coherence-guided passage reranking for RAG systems
2. Coherence-guided retrieval strategies
3. Complete RAG pipeline with coherence optimization
4. Evaluation of coherence improvements in RAG context
"""

from typing import List

import numpy as np

from coherify import PropositionSet
from coherify.measures import HybridCoherence, SemanticCoherence
from coherify.rag import CoherenceGuidedRetriever, CoherenceRAG, CoherenceReranker
from coherify.rag.reranking import PassageCandidate


def create_mock_retrieval_function():
    """Create a mock retrieval function for demonstration."""

    # Mock corpus of passages about machine learning
    passages_db = [
        (
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            0.95,
        ),
        (
            "Neural networks are computing systems inspired by biological neural networks and are used in machine learning.",
            0.90,
        ),
        (
            "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            0.88,
        ),
        (
            "Supervised learning algorithms learn from labeled training data to make predictions on new data.",
            0.85,
        ),
        (
            "Unsupervised learning finds hidden patterns in data without using labeled examples.",
            0.82,
        ),
        (
            "Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.",
            0.80,
        ),
        (
            "The weather today is sunny with a high of 75 degrees Fahrenheit.",
            0.15,
        ),  # Irrelevant
        (
            "Overfitting occurs when a model learns the training data too well and performs poorly on new data.",
            0.78,
        ),
        (
            "Feature engineering involves selecting and transforming variables for machine learning models.",
            0.75,
        ),
        (
            "Cross-validation is a technique for assessing how well a model generalizes to unseen data.",
            0.73,
        ),
        (
            "Python is a programming language commonly used for machine learning and data science.",
            0.70,
        ),
        (
            "Gradient descent is an optimization algorithm used to minimize loss functions in machine learning.",
            0.72,
        ),
        (
            "Data preprocessing includes cleaning, transforming, and preparing data for machine learning algorithms.",
            0.68,
        ),
        (
            "Basketball is a sport played by two teams of five players each on a rectangular court.",
            0.10,
        ),  # Irrelevant
        (
            "Bias in machine learning refers to systematic errors that can affect model predictions.",
            0.65,
        ),
    ]

    def mock_retrieval(query: str) -> List[PassageCandidate]:
        """Mock retrieval function that returns passages with similarity scores."""
        # Simple keyword matching for demo
        query_lower = query.lower()
        candidates = []

        for text, base_score in passages_db:
            # Simple relevance scoring based on keyword overlap
            text_lower = text.lower()
            words = query_lower.split()
            matches = sum(1 for word in words if word in text_lower)
            relevance_boost = matches * 0.1

            final_score = min(base_score + relevance_boost, 1.0)

            candidates.append(
                PassageCandidate(
                    text=text,
                    score=final_score,
                    metadata={"source": "mock_db", "base_score": base_score},
                )
            )

        # Sort by score and return top results
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:10]  # Return top 10

    return mock_retrieval


def basic_reranking_demo():
    """Demonstrate basic coherence-guided reranking."""
    print("=== Basic Coherence Reranking Demo ===")

    # Create mock retrieval function
    retrieval_fn = create_mock_retrieval_function()

    # Setup coherence reranker
    reranker = CoherenceReranker(
        coherence_measure=HybridCoherence(),
        coherence_weight=0.6,
        retrieval_weight=0.4,
        min_coherence_threshold=0.2,
    )

    # Test query
    query = "How does machine learning work?"

    # Get initial retrieval results
    print(f"Query: {query}")
    initial_candidates = retrieval_fn(query)

    print(f"\n📊 Initial Retrieval Results (Top 5):")
    for i, candidate in enumerate(initial_candidates[:5]):
        print(f"  {i+1}. Score: {candidate.score:.3f}")
        print(f"     Text: {candidate.text[:80]}...")

    # Rerank using coherence
    reranked_result = reranker.rerank(query=query, passages=initial_candidates, top_k=5)

    print(f"\n🔄 Coherence-Reranked Results (Top 5):")
    for i, (passage, coh_score, orig_score) in enumerate(
        zip(
            reranked_result.passages,
            reranked_result.coherence_scores,
            reranked_result.original_scores,
        )
    ):
        print(f"  {i+1}. Coherence: {coh_score:.3f}, Original: {orig_score:.3f}")
        print(f"     Text: {passage.text[:80]}...")

    # Show reranking statistics
    metadata = reranked_result.reranking_metadata
    print(f"\n📈 Reranking Statistics:")
    print(f"  Processing time: {metadata['reranking_time']:.3f}s")
    print(f"  Passages processed: {metadata['original_count']}")
    print(f"  Passages returned: {metadata['reranked_count']}")
    print(f"  Passages filtered: {metadata['filtered_count']}")
    print(f"  Mean coherence: {metadata['score_statistics']['coherence_mean']:.3f}")


def complete_rag_pipeline_demo():
    """Demonstrate complete RAG pipeline with coherence."""
    print("\n=== Complete RAG Pipeline Demo ===")

    # Setup RAG system
    reranker = CoherenceReranker(
        coherence_measure=HybridCoherence(), coherence_weight=0.7, retrieval_weight=0.3
    )

    rag_system = CoherenceRAG(
        reranker=reranker, max_context_length=1000, passage_separator="\n\n"
    )

    # Test query
    query = "What is deep learning and how does it differ from traditional machine learning?"

    print(f"Query: {query}")

    # Retrieve and rerank
    retrieval_fn = create_mock_retrieval_function()
    reranked_result = rag_system.retrieve_and_rerank(
        query=query, retrieval_function=retrieval_fn, top_k=3
    )

    print(f"\n🔍 Retrieved and Reranked Passages:")
    for i, passage in enumerate(reranked_result.passages):
        coherence_score = reranked_result.coherence_scores[i]
        print(f"  Passage {i+1} (Coherence: {coherence_score:.3f}):")
        print(f"    {passage.text}")
        print()

    # Build context for generation
    context = rag_system.build_context(reranked_result, include_scores=True)

    print(f"📝 Generated Context for LLM:")
    print(f"Context length: {len(context)} characters")
    print(f"Context preview: {context[:200]}...")

    # Evaluate improvement
    original_passages = retrieval_fn(query)
    improvement = rag_system.evaluate_coherence_improvement(
        query=query,
        original_passages=original_passages,
        reranked_result=reranked_result,
    )

    print(f"\n📊 Coherence Improvement Analysis:")
    print(f"  Original coherence: {improvement['original_coherence']:.3f}")
    print(f"  Reranked coherence: {improvement['reranked_coherence']:.3f}")
    print(f"  Improvement: {improvement['improvement']:.3f}")
    print(f"  Relative improvement: {improvement['relative_improvement']:.1%}")


def coherence_guided_retrieval_demo():
    """Demonstrate coherence-guided retrieval strategies."""
    print("\n=== Coherence-Guided Retrieval Demo ===")

    # Setup coherence-guided retriever
    retriever = CoherenceGuidedRetriever(
        coherence_measure=SemanticCoherence(),
        query_expansion_strategy="hybrid",
        coherence_filter_threshold=0.3,
        max_expansion_terms=3,
    )

    # Test query
    query = "neural networks"

    print(f"Original Query: {query}")

    # Expand query
    expanded_query = retriever.expand_query(
        query=query,
        domain_knowledge=[
            "artificial intelligence",
            "deep learning",
            "backpropagation",
            "neurons",
        ],
    )

    print(f"Expanded Query: {expanded_query.expanded_query}")
    print(f"Coherence Keywords: {expanded_query.coherence_keywords}")
    print(f"Expansion Method: {expanded_query.metadata['expansion_method']}")

    # Retrieve with coherence filtering
    retrieval_fn = create_mock_retrieval_function()

    print(f"\n🔍 Standard Retrieval (Top 5):")
    standard_results = retrieval_fn(query)[:5]
    for i, candidate in enumerate(standard_results):
        print(f"  {i+1}. Score: {candidate.score:.3f}")
        print(f"     Text: {candidate.text[:60]}...")

    print(f"\n🎯 Coherence-Guided Retrieval (Top 5):")
    coherence_results = retriever.retrieve_with_coherence_filtering(
        query=query, retrieval_function=retrieval_fn, max_candidates=10
    )[:5]

    for i, candidate in enumerate(coherence_results):
        coherence_score = candidate.metadata.get("coherence_score", 0)
        print(
            f"  {i+1}. Retrieval: {candidate.score:.3f}, Coherence: {coherence_score:.3f}"
        )
        print(f"     Text: {candidate.text[:60]}...")

    # Evaluate retrieval quality
    evaluation = retriever.evaluate_retrieval_coherence(
        query=query, candidates=coherence_results, top_k=5
    )

    print(f"\n📈 Retrieval Quality Evaluation:")
    print(f"  Mean coherence: {evaluation['mean_individual_coherence']:.3f}")
    print(f"  Combined coherence: {evaluation['combined_coherence']:.3f}")
    if evaluation["diversity_score"] is not None:
        print(f"  Diversity score: {evaluation['diversity_score']:.3f}")


def iterative_refinement_demo():
    """Demonstrate iterative coherence-guided retrieval refinement."""
    print("\n=== Iterative Retrieval Refinement Demo ===")

    retriever = CoherenceGuidedRetriever(
        coherence_measure=HybridCoherence(), coherence_filter_threshold=0.2
    )

    retrieval_fn = create_mock_retrieval_function()

    query = "machine learning algorithms"

    print(f"Query: {query}")

    # Run iterative refinement
    final_candidates, metadata = retriever.iterative_coherence_retrieval(
        query=query,
        retrieval_function=retrieval_fn,
        max_iterations=3,
        convergence_threshold=0.05,
    )

    print(f"\n🔄 Iterative Refinement Results:")
    print(f"  Total iterations: {metadata['total_iterations']}")
    print(f"  Final coherence: {metadata['final_coherence']:.3f}")
    print(f"  Converged: {metadata['converged']}")

    print(f"\n📊 Iteration History:")
    for iteration_data in metadata["iterations"]:
        print(f"  Iteration {iteration_data['iteration'] + 1}:")
        print(f"    Coherence: {iteration_data['coherence_score']:.3f}")
        print(f"    Candidates: {iteration_data['num_candidates']}")
        print(f"    Query: {iteration_data['query_used'][:50]}...")

    print(f"\n🎯 Final Top Results:")
    for i, candidate in enumerate(final_candidates[:3]):
        coherence_score = candidate.metadata.get("coherence_score", 0)
        print(f"  {i+1}. Coherence: {coherence_score:.3f}")
        print(f"     Text: {candidate.text[:80]}...")


def comparative_analysis_demo():
    """Compare different coherence strategies for RAG."""
    print("\n=== Comparative RAG Strategy Analysis ===")

    query = "What are the main types of machine learning?"
    retrieval_fn = create_mock_retrieval_function()

    # Different reranking strategies
    strategies = {
        "High Coherence Weight": CoherenceReranker(
            coherence_weight=0.8, retrieval_weight=0.2
        ),
        "Balanced": CoherenceReranker(coherence_weight=0.5, retrieval_weight=0.5),
        "High Retrieval Weight": CoherenceReranker(
            coherence_weight=0.2, retrieval_weight=0.8
        ),
        "No Reranking": None,
    }

    print(f"Query: {query}")

    # Get baseline results
    initial_candidates = retrieval_fn(query)

    print(f"\n📊 Strategy Comparison:")

    for strategy_name, reranker in strategies.items():
        if reranker is None:
            # No reranking baseline
            top_passages = initial_candidates[:3]
            combined_text = " ".join([p.text for p in top_passages])
            prop_set = PropositionSet.from_qa_pair(query, combined_text)

            if len(prop_set.propositions) > 1:
                coherence_measure = HybridCoherence()
                result = coherence_measure.compute(prop_set)
                coherence_score = result.score
            else:
                coherence_score = 0.5

            avg_retrieval_score = np.mean([p.score for p in top_passages])
        else:
            # With reranking
            reranked_result = reranker.rerank(
                query=query, passages=initial_candidates, top_k=3
            )

            if reranked_result.passages:
                combined_text = " ".join([p.text for p in reranked_result.passages])
                prop_set = PropositionSet.from_qa_pair(query, combined_text)

                if len(prop_set.propositions) > 1:
                    result = reranker.coherence_measure.compute(prop_set)
                    coherence_score = result.score
                else:
                    coherence_score = 0.5

                avg_retrieval_score = np.mean(reranked_result.original_scores)
            else:
                coherence_score = 0.0
                avg_retrieval_score = 0.0

        print(f"  {strategy_name}:")
        print(f"    Combined Coherence: {coherence_score:.3f}")
        print(f"    Avg Retrieval Score: {avg_retrieval_score:.3f}")


def main():
    """Run all RAG integration demonstrations."""
    print("🚀 Coherify RAG Integration Demonstration")
    print("=" * 60)

    try:
        basic_reranking_demo()
        complete_rag_pipeline_demo()
        coherence_guided_retrieval_demo()
        iterative_refinement_demo()
        comparative_analysis_demo()

        print("\n" + "=" * 60)
        print("✅ All RAG integration features demonstrated successfully!")
        print("🔍 Coherify now provides comprehensive RAG optimization with coherence!")

    except Exception as e:
        print(f"❌ Error running RAG demos: {e}")
        print("Make sure you have all required dependencies installed.")
        raise


if __name__ == "__main__":
    main()
