#!/usr/bin/env python3
"""
Basic usage example of the coherify library.

This example demonstrates:
1. Creating proposition sets from QA pairs
2. Computing semantic coherence
3. Working with benchmark adapters
"""

from coherify import PropositionSet, SemanticCoherence
from coherify.benchmarks import QABenchmarkAdapter


def basic_coherence_example():
    """Basic example of coherence computation."""
    print("=== Basic Coherence Example ===")

    # Create a proposition set from a question-answer pair
    question = "What is the capital of France?"
    answer = "The capital of France is Paris. Paris is located in northern France. It's a beautiful city with many landmarks."

    prop_set = PropositionSet.from_qa_pair(question, answer)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Propositions: {[p.text for p in prop_set.propositions]}")

    # Compute semantic coherence
    coherence = SemanticCoherence()
    result = coherence.compute(prop_set)

    print(f"\nCoherence Result: {result}")
    print(f"Score: {result.score:.3f}")
    print(f"Computation time: {result.computation_time:.3f}s")
    print(f"Number of propositions: {result.details['num_propositions']}")
    print(
        f"Coherent pairs: {result.details['coherent_pairs']}/{result.details['total_pairs']}"
    )


def incoherent_example():
    """Example with incoherent propositions."""
    print("\n=== Incoherent Example ===")

    question = "What is the weather like today?"
    answer = "The weather is sunny and warm. My favorite color is blue. I had pizza for lunch."

    prop_set = PropositionSet.from_qa_pair(question, answer)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Propositions: {[p.text for p in prop_set.propositions]}")

    coherence = SemanticCoherence()
    result = coherence.compute(prop_set)

    print(f"\nCoherence Result: {result}")
    print(f"Score: {result.score:.3f}")


def benchmark_adapter_example():
    """Example using benchmark adapters."""
    print("\n=== Benchmark Adapter Example ===")

    # Simulate a benchmark sample
    qa_sample = {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare wrote Romeo and Juliet. He was an English playwright. The play was written in the early 1590s.",
    }

    # Use QA adapter
    adapter = QABenchmarkAdapter("example_benchmark")
    prop_set = adapter.adapt_single(qa_sample)

    print(f"Benchmark sample: {qa_sample}")
    print(f"Adapted propositions: {[p.text for p in prop_set.propositions]}")
    print(f"Context: {prop_set.context}")

    # Compute coherence
    coherence = SemanticCoherence()
    result = coherence.compute(prop_set)

    print(f"\nCoherence Score: {result.score:.3f}")


def multi_answer_example():
    """Example with multiple answers (self-consistency style)."""
    print("\n=== Multiple Answer Example ===")

    question = "What is 2 + 2?"
    answers = [
        "2 + 2 equals 4",
        "The sum of 2 and 2 is 4",
        "When you add 2 plus 2, you get 4",
        "Two plus two makes four",
    ]

    prop_set = PropositionSet.from_multi_answer(question, answers)
    print(f"Question: {question}")
    print(f"Multiple answers: {answers}")

    coherence = SemanticCoherence()
    result = coherence.compute(prop_set)

    print(f"\nCoherence Score: {result.score:.3f}")
    print("(Higher score expected due to semantic similarity)")


def aggregation_comparison():
    """Compare different aggregation methods."""
    print("\n=== Aggregation Comparison ===")

    question = "Tell me about machine learning."
    answer = "Machine learning is a subset of AI. Dogs are great pets. Neural networks process data. I like ice cream."

    prop_set = PropositionSet.from_qa_pair(question, answer)
    print(f"Mixed coherence answer: {answer}")

    aggregations = ["mean", "min", "median"]

    for agg in aggregations:
        coherence = SemanticCoherence(aggregation=agg)
        result = coherence.compute(prop_set)
        print(f"{agg.capitalize()} aggregation: {result.score:.3f}")


def main():
    """Run all examples."""
    try:
        basic_coherence_example()
        incoherent_example()
        benchmark_adapter_example()
        multi_answer_example()
        aggregation_comparison()

        print("\n=== Summary ===")
        print("✅ All examples completed successfully!")
        print("📖 Coherify library is working correctly.")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install sentence-transformers scikit-learn")
        raise


if __name__ == "__main__":
    main()
