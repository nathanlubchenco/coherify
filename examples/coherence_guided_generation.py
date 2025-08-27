#!/usr/bin/env python3
"""
Coherence-Guided Generation demonstration of the coherify library.

This example demonstrates:
1. Coherence-guided beam search for better text generation
2. Multi-stage filtering for generated candidates
3. Real-time coherence guidance during generation
4. Streaming coherence monitoring
5. Adaptive generation strategies
"""

import random
from typing import List, Tuple

import numpy as np

from coherify.generation import (
    AdaptiveCoherenceFilter,
    CoherenceFilter,
    CoherenceGuidedBeamSearch,
    CoherenceGuidedGenerator,
    MultiStageFilter,
    StreamingCoherenceGuide,
)
from coherify.measures import HybridCoherence, SemanticCoherence


def create_mock_language_model():
    """Create a mock language model for demonstration."""

    # Topic-specific vocabulary
    vocabularies = {
        "machine_learning": {
            "high_prob": [
                "learning",
                "algorithm",
                "data",
                "model",
                "training",
                "neural",
                "network",
            ],
            "medium_prob": [
                "analysis",
                "prediction",
                "accuracy",
                "optimization",
                "feature",
                "pattern",
            ],
            "low_prob": [
                "science",
                "technology",
                "system",
                "process",
                "method",
                "research",
            ],
        },
        "climate_science": {
            "high_prob": [
                "climate",
                "temperature",
                "carbon",
                "emissions",
                "warming",
                "atmosphere",
            ],
            "medium_prob": [
                "greenhouse",
                "effects",
                "ocean",
                "weather",
                "environmental",
                "pollution",
            ],
            "low_prob": ["science", "research", "study", "data", "analysis", "global"],
        },
        "general": {
            "high_prob": [
                "the",
                "and",
                "of",
                "to",
                "a",
                "in",
                "that",
                "is",
                "for",
                "with",
            ],
            "medium_prob": [
                "this",
                "can",
                "will",
                "are",
                "be",
                "have",
                "from",
                "they",
                "which",
            ],
            "low_prob": [
                ".",
                ",",
                "!",
                "?",
                "however",
                "therefore",
                "moreover",
                "furthermore",
            ],
        },
    }

    def generate_candidates(
        context: str, current_text: str, num_candidates: int
    ) -> List[Tuple[str, float]]:
        """Generate token candidates with probabilities."""
        # Determine topic from context
        context_lower = context.lower()
        if "machine learning" in context_lower or "ai" in context_lower:
            topic = "machine_learning"
        elif "climate" in context_lower or "environment" in context_lower:
            topic = "climate_science"
        else:
            topic = "general"

        vocab = vocabularies[topic]

        # Create candidate pool
        candidates = []

        # Add high probability words
        for word in vocab["high_prob"]:
            log_prob = random.uniform(-0.5, -0.1)
            candidates.append((word, log_prob))

        # Add medium probability words
        for word in vocab["medium_prob"]:
            log_prob = random.uniform(-1.0, -0.5)
            candidates.append((word, log_prob))

        # Add low probability words
        for word in vocab["low_prob"]:
            log_prob = random.uniform(-2.0, -1.0)
            candidates.append((word, log_prob))

        # Sample requested number
        selected = random.sample(candidates, min(num_candidates, len(candidates)))
        return selected

    return generate_candidates


def beam_search_demo():
    """Demonstrate coherence-guided beam search."""
    print("=== Coherence-Guided Beam Search Demo ===")

    # Setup beam search
    beam_search = CoherenceGuidedBeamSearch(
        coherence_measure=HybridCoherence(),
        coherence_weight=0.4,
        lm_weight=0.6,
        beam_size=5,
        max_length=20,
        min_coherence_threshold=0.2,
    )

    # Test contexts and prompts
    test_cases = [
        {
            "context": "Machine learning algorithms analyze data to find patterns",
            "prompt": "Deep learning",
            "description": "ML context with neural network prompt",
        },
        {
            "context": "Climate change affects global weather patterns",
            "prompt": "Rising temperatures",
            "description": "Climate context with warming prompt",
        },
    ]

    mock_lm = create_mock_language_model()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"Context: {test_case['context']}")
        print(f"Prompt: {test_case['prompt']}")

        # Create generation function
        def generation_fn(current_text: str, beam_size: int) -> List[Tuple[str, float]]:
            return mock_lm(test_case["context"], current_text, beam_size)

        # Perform beam search
        result = beam_search.search(
            context=test_case["context"],
            generation_function=generation_fn,
            prompt=test_case["prompt"],
        )

        print(f"\n📝 Results:")
        print(f"  Best candidate: '{result.best_candidate.text}'")
        print(f"  Coherence score: {result.best_candidate.coherence_score:.3f}")
        print(f"  Combined score: {result.best_candidate.combined_score:.3f}")
        print(f"  Generation steps: {result.total_steps}")
        print(f"  Search time: {result.search_time:.3f}s")
        print(f"  Coherence evaluations: {result.coherence_evaluations}")

        # Show top alternatives
        top_candidates = sorted(
            result.all_candidates, key=lambda x: x.combined_score, reverse=True
        )[:3]
        print(f"  Top alternatives:")
        for j, candidate in enumerate(top_candidates[:3]):
            if candidate != result.best_candidate:
                print(
                    f"    {j+1}. '{candidate.text}' (coherence: {candidate.coherence_score:.3f})"
                )


def filtering_demo():
    """Demonstrate coherence-based filtering."""
    print("\n=== Coherence-Based Filtering Demo ===")

    # Test filtering with generated candidates
    context = "Artificial intelligence systems can learn from data to make predictions"

    # Mock generated candidates (mix of coherent and incoherent)
    candidates = [
        "Machine learning algorithms process large datasets to identify patterns and relationships.",
        "Neural networks use layers of interconnected nodes to model complex data relationships.",
        "Pizza is a delicious Italian food with cheese and tomato sauce.",  # Incoherent
        "Deep learning models require substantial computational resources for training.",
        "The weather today is sunny with temperatures reaching 75 degrees.",  # Incoherent
        "Supervised learning methods use labeled examples to train predictive models.",
        "Basketball is played by two teams with five players each on a court.",  # Incoherent
        "Reinforcement learning agents learn through interaction with their environment.",
    ]

    print(f"Context: {context}")
    print(f"Total candidates: {len(candidates)}")

    # Basic filtering
    print(f"\n🔧 Basic Coherence Filter:")
    basic_filter = CoherenceFilter(
        coherence_measure=SemanticCoherence(),
        min_coherence_threshold=0.3,
        max_candidates=5,
    )

    basic_result = basic_filter.filter_candidates(context, candidates)

    print(f"  Passed: {len(basic_result.passed_candidates)}/{len(candidates)}")
    print(f"  Pass rate: {basic_result.filter_statistics['pass_rate']:.1%}")
    print(f"  Average coherence: {np.mean(basic_result.coherence_scores):.3f}")

    print(f"  Top filtered candidates:")
    for i, (candidate, score) in enumerate(
        zip(basic_result.passed_candidates, basic_result.coherence_scores)
    ):
        print(f"    {i+1}. Score {score:.3f}: {candidate[:60]}...")

    # Multi-stage filtering
    print(f"\n🏭 Multi-Stage Filter Pipeline:")
    multi_filter = MultiStageFilter(
        coherence_measure=SemanticCoherence(),
        stages=[
            {"name": "initial", "threshold": 0.2, "description": "Initial filter"},
            {"name": "quality", "threshold": 0.4, "description": "Quality filter"},
            {
                "name": "excellence",
                "threshold": 0.6,
                "description": "Excellence filter",
            },
        ],
    )

    multi_results = multi_filter.filter_candidates(context, candidates)

    for stage_name, stage_result in multi_results.items():
        print(
            f"  {stage_name.capitalize()}: {len(stage_result.passed_candidates)} candidates"
        )
        if stage_result.coherence_scores:
            print(f"    Avg coherence: {np.mean(stage_result.coherence_scores):.3f}")

    # Adaptive filtering
    print(f"\n🎯 Adaptive Coherence Filter:")
    adaptive_filter = AdaptiveCoherenceFilter(
        coherence_measure=SemanticCoherence(), min_coherence_threshold=0.3
    )

    # Simulate multiple filtering rounds
    for round_num in range(3):
        adaptive_result = adaptive_filter.filter_candidates(context, candidates)
        print(
            f"  Round {round_num + 1}: Threshold {adaptive_filter.min_coherence_threshold:.2f}, "
            f"Passed {len(adaptive_result.passed_candidates)}"
        )


def guided_generation_demo():
    """Demonstrate high-level coherence-guided generation."""
    print("\n=== Coherence-Guided Generation Demo ===")

    # Setup generator
    generator = CoherenceGuidedGenerator(
        coherence_measure=HybridCoherence(),
        beam_search_config={
            "coherence_weight": 0.5,
            "lm_weight": 0.5,
            "beam_size": 4,
            "max_length": 30,
        },
        filter_config={"min_coherence_threshold": 0.3, "max_candidates": 3},
        guidance_enabled=True,
    )

    test_contexts = [
        "Machine learning is transforming how we analyze data and make predictions",
        "Climate change research focuses on understanding environmental impacts",
    ]

    for i, context in enumerate(test_contexts, 1):
        print(f"\n🎯 Generation Test {i}:")
        print(f"Context: {context}")

        # Generate with guidance
        generated_text, guidance_history = generator.generate(
            context=context,
            prompt="Recent advances in",
            max_length=25,
            num_candidates=4,
            return_guidance=True,
        )

        print(f"Generated: '{generated_text}'")

        if guidance_history:
            guidance = guidance_history[0]
            print(f"Coherence: {guidance.coherence_score:.3f}")
            print(f"Trend: {guidance.coherence_trend}")
            print(f"Confidence: {guidance.confidence:.3f}")

            if guidance.recommendations:
                print(f"Recommendations:")
                for rec in guidance.recommendations[:2]:
                    print(f"  - {rec}")

    # Iterative generation
    print(f"\n🔄 Iterative Generation:")
    iterative_session = generator.generate_iterative(
        context="Artificial intelligence research focuses on developing intelligent systems",
        prompt="Modern AI systems",
        max_iterations=3,
        improvement_threshold=0.05,
    )

    print(f"Final text: '{iterative_session.generated_text}'")
    print(f"Final coherence: {iterative_session.final_coherence:.3f}")
    print(f"Iterations: {iterative_session.metadata.get('iterations', 0)}")
    print(f"Converged: {iterative_session.metadata.get('converged', False)}")

    # Session analytics
    analytics = generator.get_session_analytics()
    if not analytics.get("no_sessions"):
        print(f"\n📊 Session Analytics:")
        perf = analytics["performance_metrics"]
        print(f"  Average coherence: {perf['avg_coherence']:.3f}")
        print(f"  Average session time: {perf['avg_session_time']:.3f}s")

        qual = analytics["quality_distribution"]
        print(f"  Quality distribution:")
        print(f"    High (>0.7): {qual['high_quality']:.1%}")
        print(f"    Medium (0.4-0.7): {qual['medium_quality']:.1%}")
        print(f"    Low (<0.4): {qual['low_quality']:.1%}")


def streaming_guidance_demo():
    """Demonstrate real-time streaming coherence guidance."""
    print("\n=== Streaming Coherence Guidance Demo ===")

    # Setup streaming guide
    guide = StreamingCoherenceGuide(
        coherence_measure=SemanticCoherence(),
        window_size=20,
        guidance_frequency=5,  # Guidance every 5 tokens
    )

    context = "Machine learning algorithms are designed to learn patterns from data"
    guide.start_session(context)

    # Simulate streaming generation
    streaming_tokens = [
        "Deep",
        "learning",
        "models",
        "use",
        "neural",
        "networks",
        "to",
        "process",
        "complex",
        "data",
        "patterns",
        "and",
        "make",
        "accurate",
        "predictions",
        "about",
        "future",
        "outcomes",
        "in",
        "various",
        "domains",
        "including",
        "computer",
        "vision",
        "and",
        "natural",
        "language",
        "processing",
        "applications",
        ".",
    ]

    print(f"Context: {context}")
    print(f"Simulating streaming generation with {len(streaming_tokens)} tokens...")

    guidance_updates = []

    for i, token in enumerate(streaming_tokens):
        guidance = guide.add_token(token)

        if guidance:
            guidance_updates.append((i + 1, guidance))
            print(f"\n🔄 Guidance Update at token {i + 1}:")
            print(f"  Current text: '{' '.join(list(guide.generated_tokens))}'")
            print(f"  Coherence: {guidance.coherence_score:.3f}")
            print(f"  Trend: {guidance.coherence_trend}")
            print(f"  Confidence: {guidance.confidence:.3f}")

            if guidance.recommendations:
                print(f"  Recommendations: {guidance.recommendations[0]}")

    # Session summary
    summary = guide.get_session_summary()
    print(f"\n📋 Streaming Session Summary:")
    stats = summary["session_stats"]
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Guidance updates: {stats['guidance_updates']}")
    print(f"  Final coherence: {stats['final_coherence']:.3f}")
    print(f"  Average coherence: {stats['avg_coherence']:.3f}")

    trend = summary["trend_analysis"]
    print(f"  Overall trend: {trend['overall_trend']}")
    print(f"  Peak coherence: {trend['peak_coherence']:.3f}")


def comparative_analysis_demo():
    """Compare different generation strategies."""
    print("\n=== Comparative Generation Analysis ===")

    context = "Artificial intelligence research aims to create intelligent systems"
    prompt = "Recent developments in"

    # Different generator configurations
    configurations = {
        "High Coherence": {
            "coherence_weight": 0.7,
            "lm_weight": 0.3,
            "min_coherence_threshold": 0.5,
        },
        "Balanced": {
            "coherence_weight": 0.5,
            "lm_weight": 0.5,
            "min_coherence_threshold": 0.3,
        },
        "High Fluency": {
            "coherence_weight": 0.3,
            "lm_weight": 0.7,
            "min_coherence_threshold": 0.2,
        },
    }

    results = {}

    print(f"Context: {context}")
    print(f"Prompt: '{prompt}'")

    for config_name, config in configurations.items():
        generator = CoherenceGuidedGenerator(
            coherence_measure=HybridCoherence(),
            beam_search_config={
                "coherence_weight": config["coherence_weight"],
                "lm_weight": config["lm_weight"],
                "beam_size": 4,
                "max_length": 20,
            },
            filter_config={
                "min_coherence_threshold": config["min_coherence_threshold"]
            },
        )

        generated_text, guidance = generator.generate(
            context=context, prompt=prompt, return_guidance=True
        )

        results[config_name] = {
            "text": generated_text,
            "coherence": guidance[0].coherence_score if guidance else 0,
            "config": config,
        }

    print(f"\n📊 Comparative Results:")
    for config_name, result in results.items():
        print(f"\n  {config_name}:")
        print(f"    Generated: '{result['text']}'")
        print(f"    Coherence: {result['coherence']:.3f}")
        print(f"    Coherence weight: {result['config']['coherence_weight']}")

    # Find best balance
    best_config = max(results.items(), key=lambda x: x[1]["coherence"])
    print(
        f"\n🏆 Best coherence achieved by: {best_config[0]} ({best_config[1]['coherence']:.3f})"
    )


def main():
    """Run all coherence-guided generation demonstrations."""
    print("🚀 Coherify Coherence-Guided Generation Demonstration")
    print("=" * 65)

    try:
        beam_search_demo()
        filtering_demo()
        guided_generation_demo()
        streaming_guidance_demo()
        comparative_analysis_demo()

        print("\n" + "=" * 65)
        print("✅ All coherence-guided generation features demonstrated successfully!")
        print(
            "🎯 Coherify now provides comprehensive generation guidance with coherence!"
        )

    except Exception as e:
        print(f"❌ Error running generation demos: {e}")
        print("Make sure you have all required dependencies installed.")
        raise


if __name__ == "__main__":
    main()
