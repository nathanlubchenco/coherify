#!/usr/bin/env python3
"""
Basic test of multi-format benchmark adapters without API dependencies.

This script demonstrates the core functionality of our new benchmark formats
using local processing only (no API keys required).
"""

import time

from coherify import (
    GSM8KAdapter,
    HellaSwagAdapter,
    HybridCoherence,
    MMLUAdapter,
    MultiResponseBenchmarkConfig,
    SemanticCoherence,
    TemperatureVarianceCoherence,
)


def test_gsm8k_adapter():
    """Test GSM8K mathematical reasoning adapter."""
    print("🧮 Testing GSM8K Adapter")
    print("-" * 40)

    # Mock GSM8K data
    sample = {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She sells them for $2 each, so she makes 9 * 2 = $18. #### 18",
    }

    # Setup adapter (no API for basic test)
    config = MultiResponseBenchmarkConfig(enable_multi_response=False)
    adapter = GSM8KAdapter(config=config, provider=None)

    # Test adaptation
    prop_set = adapter.adapt_single(sample)
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Propositions extracted: {len(prop_set.propositions)}")

    for i, prop in enumerate(prop_set.propositions):
        print(f"    {i+1}. {prop.text[:80]}...")

    # Test coherence evaluation
    measure = HybridCoherence()
    result = measure.compute(prop_set)
    print(f"  Coherence score: {result.score:.3f}")

    return result.score


def test_hellaswag_adapter():
    """Test HellaSwag commonsense reasoning adapter."""
    print("\n🤔 Testing HellaSwag Adapter")
    print("-" * 40)

    # Mock HellaSwag data
    sample = {
        "ctx": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
        "endings": [
            "rinses the bucket off with soap and blow dry the dog.",
            "uses a hose to keep washing the dog.",
            "gets the dog wet, then it runs away again.",
            "gets into a bathtub with the dog.",
        ],
        "label": 2,
    }

    # Setup adapter
    config = MultiResponseBenchmarkConfig(enable_multi_response=False)
    adapter = HellaSwagAdapter(config=config, provider=None)

    # Test adaptation
    prop_set = adapter.adapt_single(sample)
    print(f"  Context: {sample['ctx']}")
    print(f"  Propositions extracted: {len(prop_set.propositions)}")

    for i, prop in enumerate(prop_set.propositions):
        print(f"    {i+1}. {prop.text[:80]}...")

    # Test coherence evaluation
    measure = SemanticCoherence()
    result = measure.compute(prop_set)
    print(f"  Coherence score: {result.score:.3f}")

    return result.score


def test_mmlu_adapter():
    """Test MMLU knowledge consistency adapter."""
    print("\n📚 Testing MMLU Adapter")
    print("-" * 40)

    # Mock MMLU data
    sample = {
        "question": "Which of the following is the basic unit of life?",
        "choices": ["Atom", "Molecule", "Cell", "Tissue"],
        "answer": 2,
        "subject": "biology",
    }

    # Setup adapter
    config = MultiResponseBenchmarkConfig(enable_multi_response=False)
    adapter = MMLUAdapter(config=config, provider=None)

    # Test adaptation
    prop_set = adapter.adapt_single(sample)
    print(f"  Question: {sample['question']}")
    print(f"  Subject: {sample['subject']}")
    print(f"  Propositions extracted: {len(prop_set.propositions)}")

    for i, prop in enumerate(prop_set.propositions):
        print(f"    {i+1}. {prop.text}")

    # Test coherence evaluation
    measure = HybridCoherence()
    result = measure.compute(prop_set)
    print(f"  Coherence score: {result.score:.3f}")

    return result.score


def test_temperature_variance_measure():
    """Test temperature variance coherence measure (without API)."""
    print("\n🌡️ Testing Temperature Variance Measure")
    print("-" * 40)

    # Test with mock responses
    try:
        measure = TemperatureVarianceCoherence(provider=None)

        # This will use synthetic responses since no provider
        result = measure.compute_multi_response(
            prompt="What is the capital of France?", context="Geography question"
        )

        print(f"  Generated {len(result.responses)} responses")
        print(f"  Mean coherence: {result.mean_coherence:.3f}")
        print(f"  Consistency score: {result.consistency_score:.3f}")
        print(f"  Confidence score: {result.confidence_score:.3f}")

        return result.mean_coherence

    except Exception as e:
        print(f"  ⚠️ Temperature variance test failed: {e}")
        return 0.0


def run_performance_benchmark():
    """Run a simple performance benchmark across formats."""
    print("\n⚡ Performance Benchmark")
    print("-" * 40)

    # Create test data
    test_samples = {
        "GSM8K": [
            {
                "question": "Tom has 5 apples. He gives 2 to his friend. How many does he have left?",
                "answer": "Tom starts with 5 apples. He gives away 2. So he has 5 - 2 = 3 apples left. #### 3",
            },
            {
                "question": "A store sells books for $12 each. How much do 3 books cost?",
                "answer": "Each book costs $12. For 3 books, the cost is 3 × $12 = $36. #### 36",
            },
        ],
        "HellaSwag": [
            {
                "ctx": "A man is cooking in the kitchen. He",
                "endings": [
                    "puts on a hat",
                    "stirs the pot",
                    "leaves the house",
                    "watches TV",
                ],
                "label": 1,
            },
            {
                "ctx": "Children are playing in the park. They",
                "endings": [
                    "go to work",
                    "swing on swings",
                    "drive cars",
                    "read newspapers",
                ],
                "label": 1,
            },
        ],
        "MMLU": [
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "subject": "mathematics",
            },
            {
                "question": "What gas do plants absorb from the air?",
                "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
                "answer": 2,
                "subject": "biology",
            },
        ],
    }

    # Setup adapters
    config = MultiResponseBenchmarkConfig(enable_multi_response=False)
    adapters = {
        "GSM8K": GSM8KAdapter(config=config, provider=None),
        "HellaSwag": HellaSwagAdapter(config=config, provider=None),
        "MMLU": MMLUAdapter(config=config, provider=None),
    }

    measure = HybridCoherence()

    # Run benchmark
    results = {}

    for format_name, samples in test_samples.items():
        print(f"  Testing {format_name}...")

        start_time = time.time()
        scores = []

        for sample in samples:
            prop_set = adapters[format_name].adapt_single(sample)
            result = measure.compute(prop_set)
            scores.append(result.score)

        elapsed = time.time() - start_time
        avg_score = sum(scores) / len(scores)

        results[format_name] = {
            "avg_coherence": avg_score,
            "time_per_sample": elapsed / len(samples),
            "num_samples": len(samples),
        }

        print(f"    Avg coherence: {avg_score:.3f}")
        print(f"    Time per sample: {elapsed/len(samples):.3f}s")

    return results


def main():
    """Run all basic tests."""
    print("🚀 Multi-Format Benchmark Basic Test")
    print("=" * 50)

    scores = {}

    # Test individual adapters
    scores["GSM8K"] = test_gsm8k_adapter()
    scores["HellaSwag"] = test_hellaswag_adapter()
    scores["MMLU"] = test_mmlu_adapter()
    scores["TempVariance"] = test_temperature_variance_measure()

    # Run performance benchmark
    perf_results = run_performance_benchmark()

    # Summary
    print("\n📊 Summary")
    print("-" * 40)
    print("Coherence Scores:")
    for format_name, score in scores.items():
        print(f"  {format_name}: {score:.3f}")

    print("\nPerformance Results:")
    for format_name, stats in perf_results.items():
        print(
            f"  {format_name}: {stats['avg_coherence']:.3f} avg, {stats['time_per_sample']:.3f}s/sample"
        )

    print("\n✅ Basic multi-format testing completed!")
    print("\n💡 Next steps:")
    print("  - Try with API providers for multi-response evaluation")
    print("  - Test with larger datasets")
    print("  - Experiment with different coherence measures")
    print("  - Compare results across benchmark formats")


if __name__ == "__main__":
    main()
