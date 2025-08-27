#!/usr/bin/env python3
"""
Approximation Algorithms demonstration of the coherify library.

This example demonstrates:
1. Sampling-based approximation for large proposition sets
2. Clustering-based approximation with hierarchical analysis
3. Incremental coherence tracking for dynamic sets
4. Streaming coherence estimation for continuous data
5. Adaptive approximation strategy selection
"""

import random
import time

from coherify import Proposition, PropositionSet
from coherify.approximation import (
    ClusterBasedApproximator,
    DiversitySampler,
    HierarchicalCoherenceApproximator,
    ImportanceSampler,
    IncrementalCoherenceTracker,
    RandomSampler,
    SamplingBasedApproximator,
    StratifiedSampler,
    StreamingCoherenceEstimator,
)
from coherify.measures import HybridCoherence, SemanticCoherence


def create_large_proposition_set(size: int = 200) -> PropositionSet:
    """Create a large proposition set for testing approximation algorithms."""

    # Topic clusters for realistic data
    topics = {
        "machine_learning": [
            "Machine learning algorithms learn patterns from data automatically.",
            "Neural networks are inspired by biological brain structures.",
            "Deep learning uses multiple layers to extract complex features.",
            "Supervised learning requires labeled training examples.",
            "Unsupervised learning finds hidden patterns without labels.",
            "Reinforcement learning learns through trial and error.",
            "Gradient descent optimizes model parameters iteratively.",
            "Overfitting occurs when models memorize training data.",
            "Cross-validation helps assess model generalization.",
            "Feature engineering improves model performance significantly.",
        ],
        "climate_science": [
            "Climate change is caused by greenhouse gas emissions.",
            "Carbon dioxide levels have increased dramatically since industrialization.",
            "Global temperatures are rising at unprecedented rates.",
            "Ice caps are melting due to global warming.",
            "Sea levels are rising threateningly for coastal regions.",
            "Extreme weather events are becoming more frequent.",
            "Renewable energy can help reduce carbon emissions.",
            "Deforestation contributes to climate change significantly.",
            "Ocean acidification affects marine ecosystems.",
            "Climate models predict future temperature changes.",
        ],
        "space_exploration": [
            "Space exploration advances human scientific knowledge.",
            "Rockets use propulsion to escape Earth's gravity.",
            "Satellites orbit Earth for communication and observation.",
            "The International Space Station hosts scientific experiments.",
            "Mars exploration seeks signs of past life.",
            "Telescopes reveal distant galaxies and stars.",
            "Astronauts experience microgravity in space.",
            "Space missions require precise navigation systems.",
            "Solar panels power spacecraft during missions.",
            "Space technology benefits life on Earth.",
        ],
        "random_facts": [
            "Bananas are berries but strawberries are not.",
            "Octopi have three hearts and blue blood.",
            "Honey never spoils under proper storage conditions.",
            "A group of flamingos is called a flamboyance.",
            "Sharks have existed longer than trees.",
            "The shortest war in history lasted 38 minutes.",
            "Butterflies taste with their feet.",
            "A group of pandas is called an embarrassment.",
            "Wombat droppings are cube-shaped.",
            "Dolphins have names for each other.",
        ],
    }

    # Generate propositions
    propositions = []
    topics_list = list(topics.keys())

    for i in range(size):
        # Bias towards coherent topics (80% coherent, 20% random)
        if i < size * 0.8:
            topic = random.choice(topics_list)
            prop_text = random.choice(topics[topic])
        else:
            # Mix topics for incoherent examples
            topic = random.choice(topics_list)
            prop_text = random.choice(topics[topic])

        # Add some variation
        if random.random() < 0.1:  # 10% chance of slight modification
            prop_text = f"Research shows that {prop_text.lower()}"

        propositions.append(Proposition(text=prop_text))

    return PropositionSet(
        propositions=propositions,
        context="Large-scale coherence analysis dataset",
        metadata={"generated": True, "size": size, "topics": list(topics.keys())},
    )


def sampling_approximation_demo():
    """Demonstrate sampling-based approximation algorithms."""
    print("=== Sampling-Based Approximation Demo ===")

    # Create large proposition set
    large_set = create_large_proposition_set(150)
    print(f"Created proposition set with {len(large_set.propositions)} propositions")

    # Different sampling strategies
    samplers = {
        "Random": RandomSampler(seed=42),
        "Stratified": StratifiedSampler(strata_key="length", num_strata=5, seed=42),
        "Diversity": DiversitySampler(coherence_measure=SemanticCoherence(), seed=42),
        "Importance": ImportanceSampler(importance_strategy="centrality", seed=42),
    }

    sample_sizes = [20, 40, 60]

    print(f"\n📊 Sampling Strategy Comparison:")

    for strategy_name, sampler in samplers.items():
        print(f"\n{strategy_name} Sampling:")

        approximator = SamplingBasedApproximator(
            sampler=sampler,
            coherence_measure=HybridCoherence(),
            num_bootstrap_samples=5,
        )

        for sample_size in sample_sizes:
            result = approximator.approximate_coherence(
                large_set, sample_size, compute_confidence=True
            )

            ci_str = ""
            if result.confidence_interval:
                ci_str = f" (CI: {result.confidence_interval[0]:.3f}-{result.confidence_interval[1]:.3f})"

            print(
                f"  Sample {sample_size:2d}: Score {result.approximate_score:.3f}{ci_str}, "
                f"Time: {result.computation_time:.3f}s, Reduction: {result.metadata['reduction_ratio']:.1%}"
            )

    # Quality evaluation
    print(f"\n🔬 Sampling Quality Evaluation:")
    best_sampler = samplers["Random"]  # Use random for comparison
    approximator = SamplingBasedApproximator(best_sampler, HybridCoherence())

    evaluation = approximator.evaluate_sampling_quality(
        large_set, sample_sizes=[10, 20, 30, 40], num_trials=3
    )

    print(f"Total propositions: {evaluation['total_propositions']}")
    print(f"Sampler: {evaluation['sampler_type']}")

    for sample_size, stats in evaluation["sample_size_results"].items():
        print(
            f"  Sample {sample_size}: Mean {stats['mean_score']:.3f} ±{stats['std_score']:.3f}, "
            f"Time: {stats['mean_time']:.3f}s"
        )


def clustering_approximation_demo():
    """Demonstrate clustering-based approximation algorithms."""
    print("\n=== Clustering-Based Approximation Demo ===")

    # Create medium-sized set for clustering
    medium_set = create_large_proposition_set(80)
    print(f"Created proposition set with {len(medium_set.propositions)} propositions")

    # Test different clustering methods
    clustering_methods = ["kmeans", "hierarchical"]

    for method in clustering_methods:
        print(f"\n🔄 {method.title()} Clustering:")

        approximator = ClusterBasedApproximator(
            coherence_measure=SemanticCoherence(),
            clustering_method=method,
            cluster_selection_strategy="centroid",
        )

        # Test different cluster counts
        cluster_counts = [5, 10, 15, 20]

        for n_clusters in cluster_counts:
            result = approximator.approximate_coherence(medium_set, n_clusters)

            print(
                f"  {n_clusters:2d} clusters: Score {result.approximate_score:.3f}, "
                f"Time: {result.computation_time:.3f}s, "
                f"Reduction: {result.metadata['reduction_ratio']:.1%}"
            )
            print(f"    Cluster sizes: {result.cluster_sizes}")

    # Hierarchical approximation
    print(f"\n🌳 Hierarchical Coherence Approximation:")

    hierarchical = HierarchicalCoherenceApproximator(
        coherence_measure=SemanticCoherence(), max_depth=4, min_cluster_size=2
    )

    result = hierarchical.approximate_coherence(medium_set)

    print(f"Hierarchical Score: {result.approximate_score:.3f}")
    print(f"Hierarchy depth: {result.metadata['hierarchy_depth']}")
    print(f"Levels built: {result.metadata['levels_built']}")
    print(f"Final clusters: {result.num_clusters}")
    print(f"Cluster sizes: {result.cluster_sizes}")

    # Get hierarchy summary
    summary = hierarchical.get_hierarchy_summary(medium_set)
    print(f"\n📊 Hierarchy Summary:")
    for level_info in summary["levels"]:
        print(
            f"  Level {level_info['level']}: {level_info['num_clusters']} clusters, "
            f"reduction: {level_info['reduction_ratio']:.1%}"
        )


def incremental_tracking_demo():
    """Demonstrate incremental coherence tracking."""
    print("\n=== Incremental Coherence Tracking Demo ===")

    # Initialize tracker
    tracker = IncrementalCoherenceTracker(
        coherence_measure=SemanticCoherence(), max_cache_size=100, recompute_threshold=5
    )

    # Simulate adding propositions incrementally
    propositions_to_add = [
        "Machine learning is a branch of artificial intelligence.",
        "Neural networks process information like the human brain.",
        "Deep learning uses multiple layers for complex pattern recognition.",
        "Supervised learning requires labeled training data.",
        "The weather today is sunny with no clouds.",  # Incoherent
        "Unsupervised learning finds patterns without labeled examples.",
        "Pizza is a popular Italian food dish.",  # Incoherent
        "Reinforcement learning uses rewards to guide learning.",
        "Gradient descent optimizes model parameters iteratively.",
        "Cross-validation helps prevent overfitting in models.",
    ]

    print(f"Adding {len(propositions_to_add)} propositions incrementally:")

    for i, prop_text in enumerate(propositions_to_add):
        proposition = Proposition(text=prop_text)
        update = tracker.add_proposition(proposition)

        coherent_marker = (
            "🟢" if update.new_score > 0.7 else "🟡" if update.new_score > 0.4 else "🔴"
        )
        incremental_marker = "⚡" if update.incremental_computation else "🔄"

        print(
            f"  {i+1:2d}. {coherent_marker} {incremental_marker} Score: {update.old_score:.3f} → {update.new_score:.3f} "
            f"({update.new_score - update.old_score:+.3f}) in {update.update_time:.4f}s"
        )
        print(f"      Text: {prop_text[:60]}...")

    # Show final state
    state = tracker.get_current_state()
    print(f"\n📈 Final Tracker State:")
    print(f"  Propositions: {state['num_propositions']}")
    print(f"  Final score: {state['current_score']:.3f}")
    print(f"  Cache hit rate: {state['cache_hit_rate']:.1%}")
    print(f"  Total updates: {state['total_updates']}")
    print(f"  Full recomputations: {state['full_recomputations']}")


def streaming_estimation_demo():
    """Demonstrate streaming coherence estimation."""
    print("\n=== Streaming Coherence Estimation Demo ===")

    # Initialize streaming estimator
    estimator = StreamingCoherenceEstimator(
        coherence_measure=SemanticCoherence(),
        window_size=20,
        reservoir_size=15,
        update_frequency=5,
    )

    # Simulate streaming data
    print("Simulating streaming proposition data:")

    # Create stream with different phases
    stream_phases = [
        (
            "Coherent ML Phase",
            [
                "Machine learning algorithms analyze data patterns.",
                "Neural networks mimic brain structure and function.",
                "Deep learning processes multiple abstraction layers.",
                "Supervised learning uses labeled training examples.",
                "Feature engineering improves model performance.",
            ],
        ),
        (
            "Mixed Topic Phase",
            [
                "Unsupervised learning finds hidden data patterns.",
                "The ocean contains vast amounts of water.",
                "Reinforcement learning optimizes through rewards.",
                "Cats are popular household pets worldwide.",
                "Cross-validation prevents model overfitting.",
            ],
        ),
        (
            "Random Facts Phase",
            [
                "Bananas are technically berries, not fruits.",
                "Octopi have three hearts and blue blood.",
                "Honey never spoils under proper conditions.",
                "Sharks have existed longer than trees.",
                "Dolphins have individual names for communication.",
            ],
        ),
    ]

    proposition_count = 0

    for phase_name, propositions in stream_phases:
        print(f"\n🌊 {phase_name}:")

        for prop_text in propositions:
            proposition = Proposition(text=prop_text)
            result = estimator.add_proposition(proposition)
            proposition_count += 1

            if proposition_count % 3 == 0:  # Show every 3rd update
                print(
                    f"  {proposition_count:2d}. Window: {result['window_coherence']:.3f}, "
                    f"Reservoir: {result['reservoir_coherence']:.3f}, "
                    f"Global: {result['global_estimate']:.3f}"
                )

    # Show final statistics
    stats = estimator.get_statistics()
    print(f"\n📊 Streaming Statistics:")
    print(f"  Total seen: {stats['total_propositions_seen']}")
    print(f"  Window size: {stats['current_window_size']}")
    print(f"  Reservoir size: {stats['current_reservoir_size']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1%}")
    print(f"  Coherence computations: {stats['coherence_computations']}")
    print(f"  Final estimates:")
    print(f"    Window: {stats['current_estimates']['window']:.3f}")
    print(f"    Reservoir: {stats['current_estimates']['reservoir']:.3f}")
    print(f"    Global: {stats['current_estimates']['global']:.3f}")


def scalability_comparison_demo():
    """Compare scalability of different approximation methods."""
    print("\n=== Scalability Comparison Demo ===")

    sizes = [50, 100, 200]
    methods = {}

    # Setup different methods
    methods["Exact"] = lambda ps: SemanticCoherence().compute(ps).score

    # Sampling method
    sampler = RandomSampler(seed=42)
    sampling_approx = SamplingBasedApproximator(sampler, SemanticCoherence())
    methods["Sampling"] = lambda ps: sampling_approx.approximate_coherence(
        ps, 30
    ).approximate_score

    # Clustering method
    clustering_approx = ClusterBasedApproximator(SemanticCoherence())
    methods["Clustering"] = lambda ps: clustering_approx.approximate_coherence(
        ps, 15
    ).approximate_score

    print(f"{'Method':<12} {'Size':<6} {'Score':<8} {'Time (s)':<10} {'Speedup':<8}")
    print("-" * 55)

    for size in sizes:
        prop_set = create_large_proposition_set(size)
        exact_time = None

        for method_name, method_func in methods.items():
            if method_name == "Exact" and size > 150:
                # Skip exact computation for very large sets
                continue

            start_time = time.time()
            try:
                score = method_func(prop_set)
                computation_time = time.time() - start_time

                if method_name == "Exact":
                    exact_time = computation_time
                    speedup = "1.0x"
                else:
                    speedup = (
                        f"{exact_time/computation_time:.1f}x" if exact_time else "N/A"
                    )

                print(
                    f"{method_name:<12} {size:<6} {score:<8.3f} {computation_time:<10.3f} {speedup:<8}"
                )

            except Exception:
                print(
                    f"{method_name:<12} {size:<6} {'ERROR':<8} {'N/A':<10} {'N/A':<8}"
                )


def adaptive_strategy_demo():
    """Demonstrate adaptive approximation strategy selection."""
    print("\n=== Adaptive Approximation Strategy Demo ===")

    # This is a simplified demo since we haven't implemented the full adaptive system
    # In practice, the adaptive approximator would analyze data characteristics
    # and choose the optimal strategy

    test_sets = [
        ("Small coherent set", create_large_proposition_set(25)),
        ("Medium mixed set", create_large_proposition_set(75)),
        ("Large diverse set", create_large_proposition_set(150)),
    ]

    print("Strategy recommendations based on set characteristics:")

    for set_name, prop_set in test_sets:
        size = len(prop_set.propositions)

        # Simple strategy selection logic
        if size <= 30:
            recommended = "Direct computation"
            reason = "Small size allows exact computation"
        elif size <= 100:
            recommended = "Clustering approximation"
            reason = "Medium size benefits from clustering"
        else:
            recommended = "Sampling approximation"
            reason = "Large size requires efficient sampling"

        print(f"\n📋 {set_name} ({size} propositions):")
        print(f"  Recommended: {recommended}")
        print(f"  Reason: {reason}")

        # Test the recommendation
        start_time = time.time()

        if "Direct" in recommended:
            score = SemanticCoherence().compute(prop_set).score
        elif "Clustering" in recommended:
            approximator = ClusterBasedApproximator(SemanticCoherence())
            result = approximator.approximate_coherence(prop_set, 10)
            score = result.approximate_score
        else:  # Sampling
            sampler = RandomSampler(seed=42)
            approximator = SamplingBasedApproximator(sampler, SemanticCoherence())
            result = approximator.approximate_coherence(prop_set, 40)
            score = result.approximate_score

        computation_time = time.time() - start_time
        print(f"  Result: Score {score:.3f} in {computation_time:.3f}s")


def main():
    """Run all approximation algorithm demonstrations."""
    print("🚀 Coherify Approximation Algorithms Demonstration")
    print("=" * 70)

    try:
        sampling_approximation_demo()
        clustering_approximation_demo()
        incremental_tracking_demo()
        streaming_estimation_demo()
        scalability_comparison_demo()
        adaptive_strategy_demo()

        print("\n" + "=" * 70)
        print("✅ All approximation algorithms demonstrated successfully!")
        print(
            "⚡ Coherify now provides scalable coherence computation for large datasets!"
        )

    except Exception as e:
        print(f"❌ Error running approximation demos: {e}")
        print("Make sure you have all required dependencies installed.")
        raise


if __name__ == "__main__":
    main()
