#!/usr/bin/env python3
"""
API Providers demonstration of the coherify library.

This example demonstrates:
1. Setting up OpenAI and Anthropic API providers
2. Using API-enhanced coherence measures
3. Temperature variance analysis
4. Reasoning model integration (OpenAI o3)
5. Multi-provider comparison
6. Production-ready API configuration
"""

import os
import time
from coherify import (
    PropositionSet,
    Proposition,
    setup_providers,
    get_provider,
    list_available_providers,
    APICoherenceConfig,
    APIEnhancedSemanticCoherence,
    APIEnhancedEntailmentCoherence,
    APIEnhancedHybridCoherence,
    OpenAIProvider,
)


def setup_demo_providers():
    """Set up API providers for demonstration."""
    print("=== Setting Up API Providers ===")

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"OpenAI API Key Available: {'✅' if has_openai else '❌'}")
    print(f"Anthropic API Key Available: {'✅' if has_anthropic else '❌'}")

    if not has_openai and not has_anthropic:
        print("\n⚠️  No API keys found. Please set:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
        print("\nUsing mock demonstration instead...")
        return None

    # Setup providers
    manager = setup_providers(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_model="gpt-4o",
        anthropic_model="claude-3-5-sonnet-20241022",
    )

    available_providers = list_available_providers()
    print(f"\n🔧 Available Providers: {available_providers}")

    # Test provider connections
    for provider_name in available_providers:
        try:
            provider = get_provider(provider_name)
            models = provider.get_available_models()
            print(f"  {provider_name}: {len(models)} models available")
        except Exception as e:
            print(f"  {provider_name}: Connection failed - {e}")

    return manager


def api_enhanced_coherence_demo():
    """Demonstrate API-enhanced coherence measures."""
    print("\n=== API-Enhanced Coherence Measures ===")

    # Test proposition set
    prop_set = PropositionSet.from_qa_pair(
        "Machine Learning Applications",
        """Machine learning algorithms analyze large datasets to identify patterns and relationships. 
        Neural networks use interconnected nodes to process information similar to biological brains.
        Deep learning models require substantial computational resources for training on complex data.
        These systems can make predictions and decisions based on learned patterns from historical data.""",
    )

    print(f"📄 Test Content: {len(prop_set.propositions)} propositions")
    for i, prop in enumerate(prop_set.propositions, 1):
        print(f"  {i}. {prop.text[:80]}...")

    # Configure API enhancement
    config = APICoherenceConfig(
        use_temperature_variance=True,
        temperature_range=[0.3, 0.7, 1.0],
        variance_weight=0.3,
        enable_reasoning_trace=True,
    )

    # Test different API-enhanced measures
    measures = []

    try:
        # Try OpenAI provider
        openai_provider = get_provider("openai")
        measures.append(
            (
                "OpenAI Semantic",
                APIEnhancedSemanticCoherence(config=config, provider=openai_provider),
            )
        )
        measures.append(
            (
                "OpenAI Hybrid",
                APIEnhancedHybridCoherence(config=config, provider=openai_provider),
            )
        )
    except Exception:
        print("  OpenAI provider not available")

    try:
        # Try Anthropic provider
        anthropic_provider = get_provider("anthropic")
        measures.append(
            (
                "Anthropic Semantic",
                APIEnhancedSemanticCoherence(
                    config=config, provider=anthropic_provider
                ),
            )
        )
        measures.append(
            (
                "Anthropic Hybrid",
                APIEnhancedHybridCoherence(config=config, provider=anthropic_provider),
            )
        )
    except Exception:
        print("  Anthropic provider not available")

    if not measures:
        print("  ❌ No API providers available for testing")
        return

    # Evaluate with each measure
    results = []

    for measure_name, measure in measures:
        print(f"\n🔍 Testing {measure_name}:")

        start_time = time.time()
        try:
            result = measure.compute(prop_set)
            eval_time = time.time() - start_time

            print(f"  Score: {result.score:.3f}")
            print(f"  Evaluation time: {eval_time:.2f}s")

            # Show API enhancement details
            if "api_enhancement" in result.details:
                enhancement = result.details["api_enhancement"]

                if "embeddings_coherence" in enhancement:
                    print(
                        f"  API Embeddings Score: {enhancement['embeddings_coherence']:.3f}"
                    )

                if "temperature_variance" in enhancement:
                    print(
                        f"  Temperature Variance Score: {enhancement['temperature_variance']:.3f}"
                    )

                if enhancement.get("reasoning_analysis"):
                    reasoning = enhancement["reasoning_analysis"]
                    if reasoning.get("reasoning_score"):
                        print(f"  Reasoning Score: {reasoning['reasoning_score']:.3f}")
                    if reasoning.get("has_trace"):
                        print(f"  Has Reasoning Trace: ✅")

            results.append(
                {
                    "name": measure_name,
                    "score": result.score,
                    "time": eval_time,
                    "provider": result.details.get("provider", "unknown"),
                    "details": result.details,
                }
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Summary comparison
    if results:
        print(f"\n📊 Results Summary:")
        best_score = max(results, key=lambda x: x["score"])
        fastest = min(results, key=lambda x: x["time"])

        print(f"  Best Score: {best_score['name']} ({best_score['score']:.3f})")
        print(f"  Fastest: {fastest['name']} ({fastest['time']:.2f}s)")

        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"  Average Score: {avg_score:.3f}")


def temperature_variance_demo():
    """Demonstrate temperature variance analysis."""
    print("\n=== Temperature Variance Analysis ===")

    try:
        provider = get_provider()  # Get default provider
        print(f"Using provider: {provider.provider_name}")

        # Test prompt for variance analysis
        prompt = "Explain the concept of machine learning in simple terms"

        print(f"🌡️  Generating responses with different temperatures...")

        # Generate with multiple temperatures
        temperatures = [0.2, 0.7, 1.2]
        responses = provider.generate_with_temperatures(
            prompt=prompt, temperatures=temperatures, max_tokens=100
        )

        print(f"\n📝 Generated Responses:")
        for i, response in enumerate(responses):
            temp = temperatures[i]
            print(f"\n  Temperature {temp}:")
            print(f"    {response.text[:150]}...")
            print(f"    Tokens: {response.tokens_used}")

        # Analyze variance
        response_texts = [r.text for r in responses]
        unique_responses = len(set(response_texts))

        print(f"\n📈 Variance Analysis:")
        print(f"  Total responses: {len(responses)}")
        print(f"  Unique responses: {unique_responses}")
        print(f"  Diversity ratio: {unique_responses / len(responses):.2f}")

        # Length analysis
        lengths = [len(r.text.split()) for r in responses]
        print(f"  Word count range: {min(lengths)}-{max(lengths)} words")
        print(f"  Average length: {sum(lengths) / len(lengths):.1f} words")

        # Use variance in coherence assessment
        test_prop_set = PropositionSet(
            [
                Proposition(
                    "Machine learning enables computers to learn patterns from data."
                ),
                Proposition(
                    "Neural networks are inspired by biological brain structures."
                ),
                Proposition(
                    "Deep learning uses multiple layers for complex pattern recognition."
                ),
            ]
        )

        config = APICoherenceConfig(
            use_temperature_variance=True,
            temperature_range=temperatures,
            variance_weight=0.4,
        )

        enhanced_measure = APIEnhancedSemanticCoherence(
            config=config, provider=provider
        )
        result = enhanced_measure.compute(test_prop_set)

        print(f"\n🎯 Coherence with Temperature Variance:")
        print(f"  Final score: {result.score:.3f}")

        enhancement = result.details.get("api_enhancement", {})
        if "temperature_variance" in enhancement:
            print(f"  Variance contribution: {enhancement['temperature_variance']:.3f}")

    except Exception as e:
        print(f"❌ Temperature variance demo failed: {e}")


def reasoning_model_demo():
    """Demonstrate reasoning model integration."""
    print("\n=== Reasoning Model Integration ===")

    try:
        # Try to get OpenAI provider for reasoning models
        openai_provider = get_provider("openai")

        # Check if reasoning models are available
        available_models = openai_provider.get_available_models()
        reasoning_models = [m for m in available_models if m.startswith("o")]

        if not reasoning_models:
            print("  ⚠️  No OpenAI reasoning models (o3, o3-mini) available")
            return

        print(f"🧠 Available reasoning models: {reasoning_models}")

        # Use reasoning model for coherence analysis
        reasoning_model = reasoning_models[0]
        print(f"  Using model: {reasoning_model}")

        # Configure for reasoning analysis
        config = APICoherenceConfig(
            enable_reasoning_trace=True,
            use_temperature_variance=False,  # Focus on reasoning
        )

        # Create provider with reasoning model
        reasoning_provider = OpenAIProvider(model_name=reasoning_model)

        # Test complex coherence scenario
        complex_prop_set = PropositionSet.from_qa_pair(
            "Complex AI Reasoning",
            """Artificial intelligence systems can exhibit emergent behaviors that were not explicitly programmed. 
            These behaviors arise from complex interactions between simple rules and large amounts of data.
            Machine learning models can sometimes produce unexpected outputs that challenge human understanding.
            However, the fundamental algorithms underlying AI are deterministic mathematical operations.""",
        )

        print(f"\n📚 Analyzing complex proposition set with reasoning model...")

        enhanced_measure = APIEnhancedSemanticCoherence(
            config=config, provider=reasoning_provider
        )

        start_time = time.time()
        result = enhanced_measure.compute(complex_prop_set)
        reasoning_time = time.time() - start_time

        print(f"\n🎯 Reasoning Model Results:")
        print(f"  Coherence Score: {result.score:.3f}")
        print(f"  Analysis Time: {reasoning_time:.2f}s")

        # Show reasoning analysis details
        enhancement = result.details.get("api_enhancement", {})
        reasoning_analysis = enhancement.get("reasoning_analysis")

        if reasoning_analysis:
            print(
                f"  Reasoning Score: {reasoning_analysis.get('reasoning_score', 'N/A')}"
            )
            print(
                f"  Has Reasoning Trace: {'✅' if reasoning_analysis.get('has_trace') else '❌'}"
            )

            if reasoning_analysis.get("reasoning_text"):
                reasoning_text = reasoning_analysis["reasoning_text"][:200]
                print(f"  Reasoning Summary: {reasoning_text}...")

    except Exception as e:
        print(f"❌ Reasoning model demo failed: {e}")


def multi_provider_comparison():
    """Compare coherence measures across multiple providers."""
    print("\n=== Multi-Provider Comparison ===")

    available_providers = list_available_providers()
    if len(available_providers) < 2:
        print("  ⚠️  Need at least 2 providers for comparison")
        return

    # Test proposition set
    test_prop_set = PropositionSet.from_qa_pair(
        "AI Ethics and Safety",
        """AI systems should be designed with human values and safety as primary considerations.
        Machine learning models can perpetuate biases present in their training data.
        Transparent and explainable AI is crucial for building trust in automated systems.
        Regular auditing and monitoring of AI systems helps ensure they operate as intended.""",
    )

    print(
        f"🔍 Comparing coherence measures across {len(available_providers)} providers"
    )

    comparison_results = {}

    for provider_name in available_providers:
        print(f"\n  Testing {provider_name}...")

        try:
            provider = get_provider(provider_name)

            # Test multiple measure types
            measures = {
                "Semantic": APIEnhancedSemanticCoherence(provider=provider),
                "Entailment": APIEnhancedEntailmentCoherence(provider=provider),
                "Hybrid": APIEnhancedHybridCoherence(provider=provider),
            }

            provider_results = {}

            for measure_name, measure in measures.items():
                try:
                    start_time = time.time()
                    result = measure.compute(test_prop_set)
                    eval_time = time.time() - start_time

                    provider_results[measure_name] = {
                        "score": result.score,
                        "time": eval_time,
                    }

                    print(f"    {measure_name}: {result.score:.3f} ({eval_time:.2f}s)")

                except Exception as e:
                    print(f"    {measure_name}: Failed - {e}")

            comparison_results[provider_name] = provider_results

        except Exception as e:
            print(f"    Provider {provider_name} failed: {e}")

    # Summary analysis
    if comparison_results:
        print(f"\n📊 Comparison Summary:")

        # Find best scores by measure type
        for measure_type in ["Semantic", "Entailment", "Hybrid"]:
            scores = []
            for provider, results in comparison_results.items():
                if measure_type in results:
                    scores.append((provider, results[measure_type]["score"]))

            if scores:
                best_provider, best_score = max(scores, key=lambda x: x[1])
                avg_score = sum(score for _, score in scores) / len(scores)

                print(f"  {measure_type}:")
                print(f"    Best: {best_provider} ({best_score:.3f})")
                print(f"    Average: {avg_score:.3f}")

        # Performance comparison
        all_times = []
        for provider, results in comparison_results.items():
            provider_times = [r["time"] for r in results.values()]
            if provider_times:
                avg_time = sum(provider_times) / len(provider_times)
                all_times.append((provider, avg_time))

        if all_times:
            fastest_provider, fastest_time = min(all_times, key=lambda x: x[1])
            print(f"\n⚡ Performance:")
            print(f"    Fastest provider: {fastest_provider} ({fastest_time:.2f}s avg)")


def production_configuration_demo():
    """Demonstrate production-ready API configuration."""
    print("\n=== Production Configuration ===")

    print("🏭 Production-Ready API Setup:")

    # Environment variable configuration
    print("\n📋 Environment Variables:")
    env_vars = [
        "OPENAI_API_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ]

    for var in env_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Not set"
        print(f"  {var}: {status}")

    # Production configuration example
    production_config = APICoherenceConfig(
        use_temperature_variance=True,
        temperature_range=[0.1, 0.5, 0.9],  # Conservative range for production
        variance_weight=0.2,  # Lower weight for stability
        enable_reasoning_trace=False,  # Disable for speed in production
    )

    print(f"\n⚙️  Production Configuration:")
    print(f"  Temperature variance: {production_config.use_temperature_variance}")
    print(f"  Temperature range: {production_config.temperature_range}")
    print(f"  Variance weight: {production_config.variance_weight}")
    print(f"  Reasoning trace: {production_config.enable_reasoning_trace}")

    # Error handling and fallbacks
    print(f"\n🛡️  Error Handling:")

    try:
        # Demonstrate graceful fallback
        primary_provider = get_provider("openai")
        fallback_provider = get_provider("anthropic")

        print(f"  Primary provider: {primary_provider.provider_name}")
        print(f"  Fallback provider: {fallback_provider.provider_name}")

        # Test with fallback logic
        test_text = (
            "AI systems require careful evaluation for coherence and reliability."
        )

        try:
            result = primary_provider.generate_text(test_text, max_tokens=50)
            print(f"  Primary provider response: ✅ ({len(result.text)} chars)")
        except Exception as e:
            print(f"  Primary provider failed: {e}")
            try:
                result = fallback_provider.generate_text(test_text, max_tokens=50)
                print(f"  Fallback provider response: ✅ ({len(result.text)} chars)")
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")

    except Exception as e:
        print(f"  Configuration test failed: {e}")

    # Rate limiting considerations
    print(f"\n🚦 Rate Limiting Considerations:")
    print(f"  - Implement exponential backoff for API failures")
    print(f"  - Cache frequently computed embeddings and probabilities")
    print(f"  - Use batch processing for multiple evaluations")
    print(f"  - Monitor API usage and costs")

    # Cost optimization
    print(f"\n💰 Cost Optimization:")
    print(f"  - Use smaller models for simple coherence tasks")
    print(f"  - Enable caching to reduce redundant API calls")
    print(f"  - Implement smart model selection based on task complexity")
    print(f"  - Monitor token usage and set limits")


def main():
    """Run all API provider demonstrations."""
    print("🚀 Coherify API Providers Demonstration")
    print("=" * 55)
    print("External API integration for production-quality coherence evaluation")
    print("=" * 55)

    try:
        # Setup providers
        manager = setup_demo_providers()

        if manager:
            # Run demonstrations
            api_enhanced_coherence_demo()
            temperature_variance_demo()
            reasoning_model_demo()
            multi_provider_comparison()

        # Always show production configuration
        production_configuration_demo()

        print("\n" + "=" * 55)
        print("✅ API Providers demonstration completed!")
        print("🎯 Ready for production-quality coherence evaluation!")

    except Exception as e:
        print(f"❌ Error running API provider demos: {e}")
        print("Make sure you have API keys set up and required dependencies installed.")
        raise


if __name__ == "__main__":
    main()
