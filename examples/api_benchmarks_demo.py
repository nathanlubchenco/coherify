#!/usr/bin/env python3
"""
API-Enhanced Benchmarks demonstration of the coherify library.

This example demonstrates:
1. API-enhanced benchmark adapters
2. External model integration for benchmark evaluation
3. Temperature variance in benchmark responses
4. Confidence scoring and answer expansion
5. Comprehensive benchmark evaluation with API providers
"""

import os
import time
from coherify import (
    PropositionSet,
    SemanticCoherence,
    HybridCoherence,
    setup_providers,
    get_provider,
)
from coherify.measures.api_enhanced import (
    APIEnhancedHybridCoherence,
    APICoherenceConfig,
)
from coherify.benchmarks.api_enhanced import (
    APIBenchmarkConfig,
    APIEnhancedQAAdapter,
    APIBenchmarkEvaluator,
)


def create_mock_qa_dataset():
    """Create a mock QA dataset for demonstration."""
    return [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables computers to learn patterns from data without explicit programming.",
            "category": "technology",
            "difficulty": "easy",
        },
        {
            "question": "How do neural networks work?",
            "answer": "Neural networks process information through interconnected nodes that simulate biological neurons.",
            "category": "technology",
            "difficulty": "medium",
        },
        {
            "question": "What causes climate change?",
            "answer": "Climate change is primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere.",
            "category": "science",
            "difficulty": "easy",
        },
        {
            "question": "Explain quantum entanglement.",
            "answer": "Quantum entanglement is a phenomenon where particles become correlated and instantly affect each other regardless of distance.",
            "category": "science",
            "difficulty": "hard",
        },
        {
            "question": "What is the purpose of photosynthesis?",
            "answer": "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen, providing energy for plants.",
            "category": "biology",
            "difficulty": "easy",
        },
    ]


def setup_api_providers_demo():
    """Set up API providers for benchmark demonstration."""
    print("=== Setting Up API Providers for Benchmarks ===")

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"OpenAI API Key: {'✅' if has_openai else '❌'}")
    print(f"Anthropic API Key: {'✅' if has_anthropic else '❌'}")

    if not has_openai and not has_anthropic:
        print("\n⚠️  No API keys available for demonstration")
        print("This demo will show the structure but skip API calls")
        return None

    # Setup providers
    manager = setup_providers(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    return manager


def basic_api_adapter_demo():
    """Demonstrate basic API-enhanced adapter functionality."""
    print("\n=== Basic API-Enhanced Adapter ===")

    # Mock dataset
    qa_dataset = create_mock_qa_dataset()
    print(f"📚 Test Dataset: {len(qa_dataset)} QA samples")

    try:
        # Configure API enhancement
        api_config = APIBenchmarkConfig(
            use_model_generation=True,
            num_generations_per_prompt=2,
            temperature_range=[0.3, 0.7],
            enable_answer_expansion=True,
            enable_confidence_scoring=True,
        )

        # Get provider
        provider = get_provider()
        print(f"🔧 Using provider: {provider.provider_name}")

        # Create API-enhanced adapter
        adapter = APIEnhancedQAAdapter(
            benchmark_name="MockQA",
            config=api_config,
            provider=provider,
            question_key="question",
            answer_key="answer",
        )

        # Test single sample adaptation
        sample = qa_dataset[0]
        print(f"\n📄 Testing sample: '{sample['question']}'")
        print(f"Original answer: {sample['answer'][:80]}...")

        # Adapt with API enhancement
        prop_set = adapter.adapt_single(sample)

        print(f"\n📊 Enhanced Proposition Set:")
        print(f"  Total propositions: {len(prop_set.propositions)}")
        print(f"  API enhanced: {prop_set.metadata.get('api_enhanced', False)}")
        print(f"  API generations: {prop_set.metadata.get('api_generations', 0)}")
        print(f"  Provider: {prop_set.metadata.get('provider', 'none')}")

        # Show different types of propositions
        for i, prop in enumerate(prop_set.propositions):
            source = prop.metadata.get("source", "original")
            temp = prop.metadata.get("temperature", "N/A")
            confidence = prop.metadata.get("api_confidence", "N/A")

            print(f"\n  Proposition {i+1} ({source}):")
            print(f"    Text: {prop.text[:100]}...")
            print(f"    Temperature: {temp}")
            print(f"    Confidence: {confidence}")

    except Exception as e:
        print(f"❌ API adapter demo failed: {e}")
        print("This is expected if no API providers are available")


def temperature_variance_benchmark_demo():
    """Demonstrate temperature variance in benchmark evaluation."""
    print("\n=== Temperature Variance in Benchmarks ===")

    try:
        provider = get_provider()

        # Configure for temperature variance analysis
        api_config = APIBenchmarkConfig(
            use_model_generation=True,
            temperature_range=[0.1, 0.5, 0.9, 1.3],  # Wider range for analysis
            num_generations_per_prompt=1,  # One per temperature
            enable_confidence_scoring=True,
        )

        adapter = APIEnhancedQAAdapter(
            benchmark_name="VarianceAnalysis", config=api_config, provider=provider
        )

        # Test question that might have varying responses
        test_sample = {
            "question": "What are the potential benefits and risks of artificial intelligence?",
            "answer": "AI offers benefits like improved efficiency and automation, but poses risks including job displacement and ethical concerns.",
        }

        print(f"🌡️  Testing temperature variance on: '{test_sample['question']}'")

        # Generate enhanced proposition set
        prop_set = adapter.adapt_single(test_sample)

        # Analyze temperature effects
        api_generated_props = [
            p
            for p in prop_set.propositions
            if p.metadata.get("source") == "api_generated"
        ]

        print(f"\n📈 Temperature Variance Analysis:")
        print(f"  Total API generations: {len(api_generated_props)}")

        # Group by temperature
        temp_groups = {}
        for prop in api_generated_props:
            temp = prop.metadata.get("temperature", "unknown")
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(prop)

        for temp, props in sorted(temp_groups.items()):
            if temp != "unknown":
                print(f"\n  Temperature {temp}:")
                for i, prop in enumerate(props):
                    confidence = prop.metadata.get("api_confidence", "N/A")
                    print(f"    Response {i+1}: {prop.text[:80]}...")
                    print(f"    Confidence: {confidence}")

        # Evaluate coherence across temperature variants
        if len(api_generated_props) > 1:
            variance_prop_set = PropositionSet(
                propositions=api_generated_props, context=test_sample["question"]
            )

            coherence_measure = HybridCoherence()
            variance_result = coherence_measure.compute(variance_prop_set)

            print(
                f"\n🎯 Coherence across temperature variants: {variance_result.score:.3f}"
            )

    except Exception as e:
        print(f"❌ Temperature variance demo failed: {e}")


def comprehensive_benchmark_evaluation():
    """Demonstrate comprehensive benchmark evaluation with API enhancement."""
    print("\n=== Comprehensive API-Enhanced Benchmark Evaluation ===")

    try:
        # Setup evaluation components
        qa_dataset = create_mock_qa_dataset()

        # Configure API enhancement
        api_config = APIBenchmarkConfig(
            use_model_generation=True,
            num_generations_per_prompt=2,
            temperature_range=[0.3, 0.7],
            enable_answer_expansion=True,
            enable_confidence_scoring=True,
        )

        # Create adapter
        adapter = APIEnhancedQAAdapter(
            benchmark_name="ComprehensiveQA", config=api_config, provider=get_provider()
        )

        # Setup coherence measures
        coherence_measures = [
            SemanticCoherence(),
            HybridCoherence(),
            APIEnhancedHybridCoherence(
                config=APICoherenceConfig(
                    use_temperature_variance=True,
                    enable_reasoning_trace=False,  # For speed
                )
            ),
        ]

        # Create evaluator
        evaluator = APIBenchmarkEvaluator(
            adapter=adapter, coherence_measures=coherence_measures, config=api_config
        )

        print(
            f"🔬 Evaluating {len(qa_dataset)} samples with {len(coherence_measures)} measures"
        )

        # Progress callback
        def progress_callback(progress, current, total):
            print(f"  Progress: {progress:.1%} ({current}/{total})")

        # Run evaluation
        start_time = time.time()
        evaluation_results = evaluator.evaluate_dataset(
            qa_dataset,
            sample_limit=3,  # Limit for demo
            progress_callback=progress_callback,
        )
        eval_time = time.time() - start_time

        print(f"\n📊 Evaluation completed in {eval_time:.2f}s")

        # Display results
        aggregate_scores = evaluation_results["aggregate_scores"]
        api_stats = evaluation_results["api_statistics"]

        print(f"\n--- API Enhancement Statistics ---")
        print(f"  Enhanced samples: {api_stats['samples_with_api_enhancement']}")
        print(f"  Total API generations: {api_stats['total_api_generations']}")
        print(
            f"  Avg generations/sample: {api_stats['average_generations_per_sample']:.1f}"
        )
        print(f"  Providers used: {', '.join(api_stats['providers_used'])}")

        print(f"\n--- Coherence Scores ---")
        for measure_name, scores in aggregate_scores.items():
            print(f"  {measure_name}:")
            print(f"    Mean: {scores['mean']:.3f} ± {scores['std']:.3f}")
            print(f"    Range: {scores['min']:.3f} - {scores['max']:.3f}")

        # Sample-level analysis
        print(f"\n--- Sample-Level Analysis ---")
        for i, result in enumerate(evaluation_results["sample_results"][:2]):
            sample = result["sample"]
            prop_set = result["proposition_set"]
            scores = result["coherence_scores"]

            print(f"\n  Sample {i+1}: '{sample['question'][:50]}...'")
            print(f"    Category: {sample.get('category', 'unknown')}")
            print(f"    Difficulty: {sample.get('difficulty', 'unknown')}")
            print(f"    Propositions: {len(prop_set.propositions)}")
            print(f"    API Enhanced: {prop_set.metadata.get('api_enhanced', False)}")

            for measure_name, score_data in scores.items():
                if "score" in score_data:
                    print(f"    {measure_name}: {score_data['score']:.3f}")

        # Generate comprehensive report
        report = evaluator.create_evaluation_report(evaluation_results)
        print(f"\n--- Full Evaluation Report ---")
        print(report)

    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")


def api_vs_local_comparison():
    """Compare API-enhanced vs local model benchmark evaluation."""
    print("\n=== API vs Local Model Comparison ===")

    try:
        # Test sample
        test_sample = {
            "question": "How does machine learning contribute to artificial intelligence?",
            "answer": "Machine learning provides AI systems with the ability to automatically improve performance through experience and data analysis.",
        }

        print(f"🔍 Comparing evaluation approaches for:")
        print(f"  Question: {test_sample['question']}")

        # Local evaluation (baseline)
        print(f"\n--- Local Model Evaluation ---")

        local_prop_set = PropositionSet.from_qa_pair(
            test_sample["question"], test_sample["answer"]
        )

        local_measure = HybridCoherence()
        local_start = time.time()
        local_result = local_measure.compute(local_prop_set)
        local_time = time.time() - local_start

        print(f"  Coherence Score: {local_result.score:.3f}")
        print(f"  Evaluation Time: {local_time:.3f}s")
        print(f"  Propositions: {len(local_prop_set.propositions)}")

        # API-enhanced evaluation
        print(f"\n--- API-Enhanced Evaluation ---")

        provider = get_provider()
        api_config = APIBenchmarkConfig(
            use_model_generation=True,
            num_generations_per_prompt=2,
            temperature_range=[0.5, 0.8],
            enable_answer_expansion=True,
        )

        adapter = APIEnhancedQAAdapter(
            benchmark_name="Comparison", config=api_config, provider=provider
        )

        api_start = time.time()
        api_prop_set = adapter.adapt_single(test_sample)

        api_measure = APIEnhancedHybridCoherence(
            config=APICoherenceConfig(use_temperature_variance=True), provider=provider
        )
        api_result = api_measure.compute(api_prop_set)
        api_time = time.time() - api_start

        print(f"  Coherence Score: {api_result.score:.3f}")
        print(f"  Evaluation Time: {api_time:.3f}s")
        print(f"  Propositions: {len(api_prop_set.propositions)}")
        print(f"  API Generations: {api_prop_set.metadata.get('api_generations', 0)}")

        # Comparison summary
        print(f"\n--- Comparison Summary ---")
        score_improvement = api_result.score - local_result.score
        time_overhead = api_time - local_time

        print(f"  Score Difference: {score_improvement:+.3f}")
        print(f"  Time Overhead: {time_overhead:+.3f}s")
        print(
            f"  Content Expansion: {len(api_prop_set.propositions) - len(local_prop_set.propositions)} propositions"
        )

        # Show value-added content
        api_generated_props = [
            p
            for p in api_prop_set.propositions
            if p.metadata.get("source") in ["api_generated", "api_expanded"]
        ]

        if api_generated_props:
            print(f"\n--- API-Generated Content ---")
            for i, prop in enumerate(api_generated_props[:2]):
                source = prop.metadata.get("source", "unknown")
                print(f"  {source.title()} {i+1}: {prop.text[:100]}...")

    except Exception as e:
        print(f"❌ API vs local comparison failed: {e}")


def production_benchmark_pipeline():
    """Demonstrate production-ready benchmark evaluation pipeline."""
    print("\n=== Production Benchmark Pipeline ===")

    print("🏭 Production Pipeline Configuration:")

    # Production configuration
    production_config = APIBenchmarkConfig(
        use_model_generation=True,
        num_generations_per_prompt=1,  # Reduced for speed
        temperature_range=[0.3, 0.7],  # Conservative range
        enable_answer_expansion=False,  # Disable for speed
        enable_confidence_scoring=True,
    )

    print(f"  Model generations: {production_config.num_generations_per_prompt}")
    print(f"  Temperature range: {production_config.temperature_range}")
    print(f"  Answer expansion: {production_config.enable_answer_expansion}")
    print(f"  Confidence scoring: {production_config.enable_confidence_scoring}")

    # Error handling and recovery
    print(f"\n🛡️  Error Handling Strategy:")
    print(f"  - Fallback to local models on API failure")
    print(f"  - Graceful degradation for rate limiting")
    print(f"  - Batch processing with retry logic")
    print(f"  - Comprehensive logging and monitoring")

    # Performance optimization
    print(f"\n⚡ Performance Optimizations:")
    print(f"  - Cached embeddings and frequent computations")
    print(f"  - Parallel processing where possible")
    print(f"  - Smart batching to minimize API calls")
    print(f"  - Temperature-based model selection")

    # Cost management
    print(f"\n💰 Cost Management:")
    print(f"  - Token usage monitoring and limits")
    print(f"  - Model selection based on task complexity")
    print(f"  - Caching to reduce redundant API calls")
    print(f"  - Batch processing for efficiency")

    # Quality assurance
    print(f"\n✅ Quality Assurance:")
    print(f"  - Confidence score thresholds")
    print(f"  - Cross-provider validation")
    print(f"  - Human review for edge cases")
    print(f"  - Continuous monitoring of evaluation quality")


def main():
    """Run all API-enhanced benchmark demonstrations."""
    print("🚀 Coherify API-Enhanced Benchmarks Demonstration")
    print("=" * 60)
    print("External API integration for enhanced benchmark evaluation")
    print("=" * 60)

    try:
        # Setup providers
        manager = setup_api_providers_demo()

        if manager:
            # Run API demonstrations
            basic_api_adapter_demo()
            temperature_variance_benchmark_demo()
            comprehensive_benchmark_evaluation()
            api_vs_local_comparison()
        else:
            print("\n⚠️  API demos skipped due to missing API keys")

        # Always show production pipeline (informational)
        production_benchmark_pipeline()

        print("\n" + "=" * 60)
        print("✅ API-enhanced benchmark demonstration completed!")
        print("🎯 Ready for production-quality benchmark evaluation!")

    except Exception as e:
        print(f"❌ Error running API benchmark demos: {e}")
        print("Ensure API keys are set and dependencies are installed.")
        raise


if __name__ == "__main__":
    main()
