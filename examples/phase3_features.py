#!/usr/bin/env python3
"""
Phase 3 features demonstration of the coherify library.

This example demonstrates:
1. Traditional Shogenji coherence with proper probability estimation
2. Visualization tools for coherence analysis and patterns
3. Comprehensive coherence analysis and reporting
4. Advanced probability estimation approaches
"""

import matplotlib.pyplot as plt

from coherify import PropositionSet
from coherify.measures import (
    ConfidenceBasedProbabilityEstimator,
    EnsembleProbabilityEstimator,
    EntailmentCoherence,
    HybridCoherence,
    SemanticCoherence,
    ShogunjiCoherence,
)
from coherify.measures.entailment import SimpleNLIModel
from coherify.utils.visualization import CoherenceAnalyzer, CoherenceVisualizer


def traditional_shogenji_demo():
    """Demonstrate traditional Shogenji coherence with probability estimation."""
    print("=== Traditional Shogenji Coherence Demo ===")

    # Use confidence-based probability estimator for demo
    prob_estimator = ConfidenceBasedProbabilityEstimator()
    shogenji = ShogunjiCoherence(probability_estimator=prob_estimator)

    print("\n1. Coherent Propositions Example:")
    coherent_answer = "All mammals are warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded."
    prop_set1 = PropositionSet.from_qa_pair(
        question="Are dogs warm-blooded?", answer=coherent_answer
    )

    result1 = shogenji.compute(prop_set1)
    print(f"Answer: {coherent_answer}")
    print(f"Shogenji Score: {result1.score:.3f}")
    print(f"Joint Probability: {result1.details['joint_probability']:.3f}")
    print(
        f"Individual Probabilities: {[f'{p:.3f}' for p in result1.details['individual_probabilities']]}"
    )
    print(f"Interpretation: {result1.details['interpretation']}")

    print("\n2. Incoherent Propositions Example:")
    incoherent_answer = (
        "Cats are animals. Ice cream is cold. The moon is made of cheese."
    )
    prop_set2 = PropositionSet.from_qa_pair(
        question="Tell me about cats.", answer=incoherent_answer
    )

    result2 = shogenji.compute(prop_set2)
    print(f"Answer: {incoherent_answer}")
    print(f"Shogenji Score: {result2.score:.3f}")
    print(f"Interpretation: {result2.details['interpretation']}")


def probability_estimation_comparison():
    """Compare different probability estimation approaches."""
    print("\n=== Probability Estimation Comparison ===")

    # Test proposition
    test_answer = (
        "Paris is the capital of France. France is in Europe. Europe is a continent."
    )
    prop_set = PropositionSet.from_qa_pair(
        question="Where is Paris?", answer=test_answer
    )

    # Different probability estimators
    estimators = {
        "Confidence-Based": ConfidenceBasedProbabilityEstimator(),
        "Ensemble": EnsembleProbabilityEstimator(),
    }

    print(f"Test Answer: {test_answer}")
    print("\nProbability Estimation Results:")

    for name, estimator in estimators.items():
        shogenji = ShogunjiCoherence(probability_estimator=estimator)
        result = shogenji.compute(prop_set)

        print(f"\n{name} Estimator:")
        print(f"  Shogenji Score: {result.score:.3f}")
        print(f"  Joint Probability: {result.details['joint_probability']:.3f}")
        print(
            f"  Individual Probs: {[f'{p:.3f}' for p in result.details['individual_probabilities']]}"
        )


def visualization_demo():
    """Demonstrate visualization tools."""
    print("\n=== Coherence Visualization Demo ===")

    # Create test data
    test_answer = "Machine learning is a branch of AI. Neural networks are used in ML. Deep learning uses multiple layers."
    prop_set = PropositionSet.from_qa_pair(
        question="What is machine learning?", answer=test_answer
    )

    # Multiple coherence measures
    measures = [
        SemanticCoherence(),
        EntailmentCoherence(nli_model=SimpleNLIModel()),
        HybridCoherence(
            semantic_weight=0.6, entailment_weight=0.4, nli_model=SimpleNLIModel()
        ),
    ]

    measure_names = ["Semantic", "Entailment", "Hybrid"]

    # Compute results
    results = [measure.compute(prop_set) for measure in measures]

    print(f"Test Answer: {test_answer}")
    print("\nCoherence Scores:")
    for name, result in zip(measure_names, results):
        print(f"  {name}: {result.score:.3f}")

    # Create visualizations
    visualizer = CoherenceVisualizer()

    print("\n1. Creating coherence scores comparison plot...")
    fig1 = visualizer.plot_coherence_scores(
        results, measure_names, title="Coherence Measures Comparison"
    )
    plt.show(block=False)

    # Component analysis for hybrid measure
    hybrid_result = results[2]  # Hybrid measure
    if "component_scores" in hybrid_result.details:
        print("2. Creating component analysis plot...")
        fig2 = visualizer.plot_component_analysis(
            hybrid_result, title="Hybrid Coherence Component Analysis"
        )
        plt.show(block=False)

    # Similarity matrix (if available)
    semantic_measure = measures[0]
    if hasattr(semantic_measure, "encoder"):
        try:
            print("3. Creating similarity matrix...")
            texts = [p.text for p in prop_set.propositions]
            embeddings = semantic_measure.encoder.encode(texts)

            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(embeddings)

            fig3 = visualizer.plot_similarity_matrix(
                similarity_matrix, texts, title="Proposition Similarity Matrix"
            )
            plt.show(block=False)

        except Exception as e:
            print(f"Note: Could not create similarity matrix: {e}")


def comprehensive_analysis_demo():
    """Demonstrate comprehensive coherence analysis."""
    print("\n=== Comprehensive Analysis Demo ===")

    # Test with a more complex example
    complex_answer = (
        "Climate change is caused by greenhouse gases. "
        "Carbon dioxide is a major greenhouse gas. "
        "Human activities increase CO2 levels. "
        "Burning fossil fuels releases CO2. "
        "Therefore, human activities contribute to climate change."
    )

    prop_set = PropositionSet.from_qa_pair(
        question="How do human activities affect climate change?", answer=complex_answer
    )

    # Set up measures
    measures = [
        SemanticCoherence(),
        EntailmentCoherence(nli_model=SimpleNLIModel()),
        HybridCoherence(
            semantic_weight=0.5, entailment_weight=0.5, nli_model=SimpleNLIModel()
        ),
        ShogunjiCoherence(probability_estimator=ConfidenceBasedProbabilityEstimator()),
    ]

    # Create analyzer
    analyzer = CoherenceAnalyzer()

    print(f"Complex Answer: {complex_answer}")
    print(f"Number of propositions: {len(prop_set.propositions)}")

    # Individual measure analysis
    print("\n1. Individual Measure Analysis:")
    for measure in measures:
        analysis = analyzer.analyze_proposition_set(prop_set, measure, detailed=True)
        print(f"\n{measure.__class__.__name__}:")
        print(f"  Overall Coherence: {analysis['overall_coherence']:.3f}")
        print(f"  Mean Pairwise: {analysis.get('mean_pairwise_coherence', 'N/A')}")
        print(f"  Std Pairwise: {analysis.get('std_pairwise_coherence', 'N/A')}")

        if analysis.get("most_coherent_pair"):
            most_coherent = analysis["most_coherent_pair"]
            print(f"  Most Coherent Pair: {most_coherent['coherence']:.3f}")

        if analysis.get("least_coherent_pair"):
            least_coherent = analysis["least_coherent_pair"]
            print(f"  Least Coherent Pair: {least_coherent['coherence']:.3f}")

    # Comparative analysis
    print("\n2. Comparative Analysis:")
    comparison = analyzer.compare_measures(prop_set, measures)

    stats = comparison["score_statistics"]
    print(f"Score Statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Range: {stats['range']:.3f}")

    # Best performing measure
    measure_results = comparison["measure_results"]
    best_measure = max(measure_results, key=lambda x: x["score"])
    print(f"  Best Measure: {best_measure['name']} ({best_measure['score']:.3f})")


def evolution_analysis_demo():
    """Demonstrate coherence evolution analysis."""
    print("\n=== Coherence Evolution Demo ===")

    # Simulate improving coherence over iterations
    iterations = [
        "Machine learning",
        "Machine learning is useful",
        "Machine learning is useful for data analysis",
        "Machine learning is useful for data analysis and pattern recognition",
        "Machine learning is useful for data analysis and pattern recognition in various domains",
    ]

    # Measure coherence evolution
    semantic_measure = SemanticCoherence()
    results_over_time = []

    for i, text in enumerate(iterations):
        prop_set = PropositionSet.from_qa_pair("What is machine learning?", text)
        if len(prop_set.propositions) > 1:  # Need multiple propositions for coherence
            result = semantic_measure.compute(prop_set)
            results_over_time.append((i + 1, result))

    if results_over_time:
        print("Coherence evolution as text becomes more detailed:")
        for step, result in results_over_time:
            print(f"  Step {step}: {result.score:.3f}")

        # Create evolution plot
        visualizer = CoherenceVisualizer()
        fig = visualizer.plot_coherence_evolution(
            results_over_time,
            x_label="Iteration",
            title="Coherence Evolution Over Text Development",
        )
        plt.show(block=False)


def main():
    """Run all Phase 3 feature demonstrations."""
    print("🚀 Coherify Phase 3 Features Demonstration")
    print("=" * 50)

    try:
        traditional_shogenji_demo()
        probability_estimation_comparison()
        visualization_demo()
        comprehensive_analysis_demo()
        evolution_analysis_demo()

        print("\n" + "=" * 50)
        print("✅ All Phase 3 features demonstrated successfully!")
        print(
            "📊 Coherify now provides comprehensive coherence analysis with visualization!"
        )
        print("\nNote: Close any open matplotlib windows to continue.")

        # Keep plots open for viewing
        if plt.get_fignums():
            print("Displaying visualizations...")
            plt.show()

    except Exception as e:
        print(f"❌ Error running Phase 3 demos: {e}")
        print("Make sure you have installed visualization dependencies:")
        print("pip install -e '.[viz]'")
        raise


if __name__ == "__main__":
    main()
