#!/usr/bin/env python3
"""
FaithBench Benchmark Runner

This script demonstrates running the FaithBench hallucination detection 
benchmark with Coherify's faithfulness-coherence evaluation.

Usage:
    python examples/run_faithbench_benchmark.py [--use-api] [--sample-size N]
"""

import os
import json
import argparse
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import required libraries
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️  requests not installed. Install with: pip install requests")

from coherify import (
    HybridCoherence, SemanticCoherence,
    setup_providers, get_provider
)

from coherify.benchmarks.faithbench_adapter import (
    FaithBenchAdapter, FaithBenchConfig, FaithfulnessCoherence,
    FaithBenchSample, FaithBenchAnnotation, FaithBenchMetadata
)

from coherify.measures.multi_response import (
    TemperatureVarianceCoherence, SelfConsistencyCoherence
)


def load_faithbench_data(sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load FaithBench dataset."""
    print("📚 Loading FaithBench data...")
    
    # For now, create comprehensive mock FaithBench data
    # In production, this would load from the actual GitHub repository
    print("  🔧 Using mock FaithBench data...")
    mock_data = create_comprehensive_faithbench_mock_data()
    
    if sample_size:
        mock_data = mock_data[:sample_size]
    
    print(f"  ✅ Using {len(mock_data)} mock FaithBench samples")
    return mock_data


def create_comprehensive_faithbench_mock_data() -> List[Dict[str, Any]]:
    """Create comprehensive mock FaithBench data covering different scenarios."""
    return [
        # Faithful summary - no hallucinations
        {
            "sample_id": 1,
            "source": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was the world's tallest man-made structure until the Chrysler Building was built in New York in 1930.",
            "summary": "The Eiffel Tower is an iron tower in Paris, France, named after engineer Gustave Eiffel. Built between 1887-1889, it was the world's tallest structure until 1930.",
            "annotations": [
                {
                    "annot_id": 1,
                    "annotator_id": "annotator_1",
                    "annotator_name": "Alice",
                    "label": ["Consistent"],
                    "note": "Summary accurately reflects source content",
                    "summary_span": "The Eiffel Tower is an iron tower in Paris, France",
                    "summary_start": 0,
                    "summary_end": 50,
                    "source_span": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France",
                    "source_start": 0,
                    "source_end": 87
                }
            ],
            "metadata": {
                "summarizer": "gpt-3.5-turbo",
                "hhemv1": 0.85,
                "hhem-2.1": 0.90,
                "trueteacher": 0,
                "true_nli": 0,
                "gpt_3.5_turbo": 0,
                "gpt_4o": 0
            }
        },
        # Intrinsic hallucination - contradicts source
        {
            "sample_id": 2,
            "source": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have been the main driver since the 1800s, primarily through burning fossil fuels like coal, oil and gas.",
            "summary": "Climate change refers to long-term shifts in global temperatures. Human activities have been the main driver since the 1900s, primarily through deforestation and agriculture.",
            "annotations": [
                {
                    "annot_id": 2,
                    "annotator_id": "annotator_2", 
                    "annotator_name": "Bob",
                    "label": ["Unwanted", "Unwanted.Intrinsic"],
                    "note": "Summary incorrectly states 1900s instead of 1800s, and lists wrong primary causes",
                    "summary_span": "since the 1900s, primarily through deforestation and agriculture",
                    "summary_start": 85,
                    "summary_end": 145,
                    "source_span": "since the 1800s, primarily through burning fossil fuels",
                    "source_start": 180,
                    "source_end": 235
                }
            ],
            "metadata": {
                "summarizer": "mistral-7b",
                "hhemv1": 0.25,
                "hhem-2.1": 0.35,
                "trueteacher": 1,
                "true_nli": 1,
                "gpt_3.5_turbo": 1,
                "gpt_4o": 1
            }
        },
        # Extrinsic hallucination - adds unsupported information
        {
            "sample_id": 3,
            "source": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar. This process is fundamental to life on Earth.",
            "summary": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and sugar. This process was discovered by Joseph Priestley in 1772 and is fundamental to all life on Earth.",
            "annotations": [
                {
                    "annot_id": 3,
                    "annotator_id": "annotator_3",
                    "annotator_name": "Carol", 
                    "label": ["Unwanted", "Unwanted.Extrinsic"],
                    "note": "Summary adds unsupported historical information about Joseph Priestley",
                    "summary_span": "This process was discovered by Joseph Priestley in 1772",
                    "summary_start": 120,
                    "summary_end": 170,
                    "source_span": None,
                    "source_start": None,
                    "source_end": None
                }
            ],
            "metadata": {
                "summarizer": "llama-2-7b",
                "hhemv1": 0.40,
                "hhem-2.1": 0.45,
                "trueteacher": 1,
                "true_nli": 0,
                "gpt_3.5_turbo": 0,
                "gpt_4o": 1
            }
        },
        # Questionable case - ambiguous
        {
            "sample_id": 4,
            "source": "The Amazon rainforest covers approximately 5.5 million square kilometers and spans across nine South American countries. It contains an estimated 390 billion individual trees.",
            "summary": "The vast Amazon rainforest spans multiple South American countries and contains hundreds of billions of trees, making it one of the most important ecosystems on Earth.",
            "annotations": [
                {
                    "annot_id": 4,
                    "annotator_id": "annotator_4",
                    "annotator_name": "David",
                    "label": ["Questionable"],
                    "note": "Summary adds 'most important ecosystems' claim which isn't explicitly in source but could be reasonable inference",
                    "summary_span": "making it one of the most important ecosystems on Earth",
                    "summary_start": 110,
                    "summary_end": 160,
                    "source_span": None,
                    "source_start": None,
                    "source_end": None
                }
            ],
            "metadata": {
                "summarizer": "claude-3-sonnet",
                "hhemv1": 0.65,
                "hhem-2.1": 0.70,
                "trueteacher": 0,
                "true_nli": 0,
                "gpt_3.5_turbo": 0,
                "gpt_4o": 0
            }
        },
        # Complex case with multiple annotations
        {
            "sample_id": 5,
            "source": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans. AI applications include advanced web search engines, recommendation systems, and autonomous vehicles. The term was first coined in 1956 at the Dartmouth Conference.",
            "summary": "Artificial intelligence refers to machine intelligence, unlike human intelligence. AI applications include web search, recommendations, and self-driving cars. The field was established in 1955 by Alan Turing at Cambridge University.",
            "annotations": [
                {
                    "annot_id": 5,
                    "annotator_id": "annotator_5",
                    "annotator_name": "Eve",
                    "label": ["Consistent"],
                    "note": "Accurate paraphrase of AI definition and applications",
                    "summary_span": "Artificial intelligence refers to machine intelligence, unlike human intelligence. AI applications include web search, recommendations, and self-driving cars.",
                    "summary_start": 0,
                    "summary_end": 130,
                    "source_span": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans. AI applications include advanced web search engines, recommendation systems, and autonomous vehicles.",
                    "source_start": 0,
                    "source_end": 220
                },
                {
                    "annot_id": 6,
                    "annotator_id": "annotator_6",
                    "annotator_name": "Frank",
                    "label": ["Unwanted", "Unwanted.Intrinsic"],
                    "note": "Incorrect date (1955 vs 1956) and wrong person/location (Alan Turing/Cambridge vs Dartmouth Conference)",
                    "summary_span": "The field was established in 1955 by Alan Turing at Cambridge University.",
                    "summary_start": 131,
                    "summary_end": 195,
                    "source_span": "The term was first coined in 1956 at the Dartmouth Conference.",
                    "source_start": 240,
                    "source_end": 302
                }
            ],
            "metadata": {
                "summarizer": "gpt-4",
                "hhemv1": 0.55,
                "hhem-2.1": 0.60,
                "trueteacher": 1,
                "true_nli": 1,
                "gpt_3.5_turbo": 1,
                "gpt_4o": 0
            }
        }
    ]


def run_faithbench_benchmark(data: List[Dict[str, Any]], 
                           use_api: bool = False,
                           sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Run FaithBench benchmark evaluation."""
    print(f"\n📊 Running FaithBench Hallucination Detection Benchmark")
    print("=" * 60)
    
    if sample_size:
        data = data[:sample_size]
    
    # Setup provider if using API
    provider = None
    if use_api:
        try:
            setup_providers()
            provider = get_provider()
            print(f"  🌐 Using API provider: {provider.provider_name}")
        except Exception as e:
            print(f"  ⚠️ Failed to setup API provider: {e}")
            use_api = False
    
    # Setup FaithBench adapter
    config = FaithBenchConfig(
        enable_multi_response=use_api,
        num_responses_per_sample=3,
        temperature_range=(0.1, 0.5),
        reasoning_trace_enabled=True,
        aggregation_strategy="majority",
        faithfulness_weight=0.7,
        coherence_weight=0.3
    )
    
    adapter = FaithBenchAdapter(config=config, provider=provider)
    
    # Setup coherence measures
    measures = [
        HybridCoherence(),
        FaithfulnessCoherence(provider=provider),
    ]
    
    if use_api:
        measures.extend([
            TemperatureVarianceCoherence(provider=provider),
            SelfConsistencyCoherence(provider=provider)
        ])
    
    print(f"  📊 Evaluating {len(data)} FaithBench samples...")
    
    # Process each sample
    results = {
        "benchmark_name": "FaithBench",
        "num_samples": len(data),
        "sample_results": [],
        "multi_response_results": [],
        "measures": {},
        "evaluation_time": 0.0,
        "faithbench_specific_metrics": {
            "accuracy": 0.0,
            "faithfulness_consistency": 0.0,
            "faithbench_score": 0.0,
            "hallucination_detection_rate": 0.0
        }
    }
    
    start_time = time.time()
    correct_predictions = 0
    faithfulness_consistency_scores = []
    faithbench_scores = []
    hallucination_detected = 0
    total_hallucinated_samples = 0
    
    for i, sample in enumerate(data):
        print(f"    Processing sample {i+1}/{len(data)}: Sample ID {sample.get('sample_id', 'unknown')}...")
        
        try:
            # Standard adaptation
            prop_set = adapter.adapt_single(sample)
            
            # Multi-response adaptation if enabled
            multi_result = None
            if config.enable_multi_response and provider:
                try:
                    multi_result = adapter.adapt_single_with_multi_response(sample)
                    results["multi_response_results"].append(multi_result)
                    
                    # Extract FaithBench-specific metrics
                    if "response_evaluation" in multi_result:
                        eval_data = multi_result["response_evaluation"]
                        if eval_data.get("is_correct"):
                            correct_predictions += 1
                        faithfulness_consistency_scores.append(eval_data.get("faithfulness_consistency", 0.0))
                        faithbench_scores.append(eval_data.get("faithbench_score", 0.0))
                        
                        # Track hallucination detection
                        if eval_data.get("ground_truth_hallucinated"):
                            total_hallucinated_samples += 1
                            if not eval_data.get("majority_faithful"):
                                hallucination_detected += 1
                        
                except Exception as e:
                    print(f"      Multi-response failed: {e}")
            else:
                # For local evaluation, check ground truth
                faithbench_sample = adapter._parse_faithbench_sample(sample)
                if faithbench_sample.has_hallucination:
                    total_hallucinated_samples += 1
            
            # Evaluate with each coherence measure
            sample_coherence = {}
            for measure in measures:
                try:
                    if isinstance(measure, FaithfulnessCoherence):
                        # Special handling for faithfulness coherence
                        faithbench_sample = adapter._parse_faithbench_sample(sample)
                        faithfulness_result = measure.evaluate_source_summary_coherence(
                            faithbench_sample.source,
                            faithbench_sample.summary,
                            f"FaithBench faithfulness evaluation"
                        )
                        sample_coherence[measure.__class__.__name__] = faithfulness_result["overall_faithfulness"]
                    else:
                        coherence_result = measure.compute(prop_set)
                        sample_coherence[measure.__class__.__name__] = coherence_result.score
                        
                except Exception as e:
                    print(f"      Measure {measure.__class__.__name__} failed: {e}")
                    sample_coherence[measure.__class__.__name__] = 0.0
            
            results["sample_results"].append({
                "sample_index": i,
                "sample_id": sample.get("sample_id", 0),
                "source_length": len(sample.get("source", "")),
                "summary_length": len(sample.get("summary", "")),
                "num_annotations": len(sample.get("annotations", [])),
                "coherence_scores": sample_coherence,
                "proposition_count": len(prop_set.propositions),
                "has_hallucination": prop_set.metadata.get("aggregated_label", False)
            })
            
        except Exception as e:
            print(f"      Sample {i} failed: {e}")
            continue
    
    results["evaluation_time"] = time.time() - start_time
    
    # Aggregate results by measure
    for measure in measures:
        measure_name = measure.__class__.__name__
        scores = [r["coherence_scores"].get(measure_name, 0.0) for r in results["sample_results"]]
        
        if scores:
            import numpy as np
            results["measures"][measure_name] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(min(scores)),
                "max_score": float(max(scores)),
                "num_samples": len(scores)
            }
    
    # Compute FaithBench-specific metrics
    if results["multi_response_results"]:
        results["faithbench_specific_metrics"] = {
            "accuracy": correct_predictions / len(data) if len(data) > 0 else 0.0,
            "faithfulness_consistency": float(np.mean(faithfulness_consistency_scores)) if faithfulness_consistency_scores else 0.0,
            "faithbench_score": float(np.mean(faithbench_scores)) if faithbench_scores else 0.0,
            "hallucination_detection_rate": hallucination_detected / total_hallucinated_samples if total_hallucinated_samples > 0 else 0.0,
            "num_multi_response_samples": len(results["multi_response_results"]),
            "total_hallucinated_samples": total_hallucinated_samples
        }
    
    print(f"  ✅ Completed in {results['evaluation_time']:.2f}s")
    return results


def analyze_faithbench_results(results: Dict[str, Any]):
    """Analyze and display FaithBench benchmark results."""
    print("\n📊 FaithBench Benchmark Analysis:")
    print("-" * 50)
    
    # Overall performance
    num_samples = results["num_samples"]
    eval_time = results["evaluation_time"]
    
    print(f"Samples evaluated: {num_samples}")
    print(f"Total time: {eval_time:.2f}s ({eval_time/num_samples:.3f}s per sample)")
    
    # Coherence scores by measure
    print(f"\n🎯 Coherence Scores by Measure:")
    measures = results.get("measures", {})
    for measure_name, stats in measures.items():
        print(f"  {measure_name}: {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")
    
    # FaithBench-specific metrics
    faithbench_metrics = results.get("faithbench_specific_metrics", {})
    if faithbench_metrics:
        print(f"\n📊 FaithBench-Specific Metrics:")
        print(f"  Prediction Accuracy: {faithbench_metrics.get('accuracy', 0.0):.1%}")
        print(f"  Faithfulness Consistency: {faithbench_metrics.get('faithfulness_consistency', 0.0):.3f}")
        print(f"  FaithBench Score: {faithbench_metrics.get('faithbench_score', 0.0):.3f}")
        print(f"  Hallucination Detection Rate: {faithbench_metrics.get('hallucination_detection_rate', 0.0):.1%}")
        print(f"  Multi-response samples: {faithbench_metrics.get('num_multi_response_samples', 0)}")
        print(f"  Hallucinated samples: {faithbench_metrics.get('total_hallucinated_samples', 0)}")
    
    # Sample-level analysis
    print(f"\n📋 Sample Analysis:")
    sample_results = results.get("sample_results", [])
    
    # Group by hallucination status
    faithful_samples = [s for s in sample_results if not s.get("has_hallucination", False)]
    hallucinated_samples = [s for s in sample_results if s.get("has_hallucination", False)]
    
    if faithful_samples:
        faithful_coherence = sum(
            sum(s["coherence_scores"].values()) / len(s["coherence_scores"]) 
            if s["coherence_scores"] else 0.0 
            for s in faithful_samples
        ) / len(faithful_samples)
        print(f"  Faithful samples: {len(faithful_samples)}, avg coherence {faithful_coherence:.3f}")
    
    if hallucinated_samples:
        hallucinated_coherence = sum(
            sum(s["coherence_scores"].values()) / len(s["coherence_scores"]) 
            if s["coherence_scores"] else 0.0 
            for s in hallucinated_samples
        ) / len(hallucinated_samples)
        print(f"  Hallucinated samples: {len(hallucinated_samples)}, avg coherence {hallucinated_coherence:.3f}")
    
    # Multi-response analysis
    multi_results = results.get("multi_response_results", [])
    if multi_results:
        print(f"\n🔄 Multi-Response Analysis:")
        consistent_predictions = 0
        good_faithfulness_reasoning = 0
        
        for result in multi_results:
            eval_data = result.get("response_evaluation", {})
            if eval_data.get("is_prediction_consistent"):
                consistent_predictions += 1
            if eval_data.get("faithfulness_consistency", 0.0) > 0.6:
                good_faithfulness_reasoning += 1
        
        print(f"  Prediction consistency: {consistent_predictions}/{len(multi_results)} ({consistent_predictions/len(multi_results):.1%})")
        print(f"  Good faithfulness reasoning: {good_faithfulness_reasoning}/{len(multi_results)} ({good_faithfulness_reasoning/len(multi_results):.1%})")


def main():
    """Main FaithBench benchmark runner."""
    parser = argparse.ArgumentParser(description="Run FaithBench benchmark with Coherify")
    parser.add_argument("--use-api", action="store_true", help="Use API-enhanced multi-response evaluation")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Enable clean output unless verbose
    if not args.verbose:
        try:
            from coherify.utils.clean_output import enable_clean_output
            enable_clean_output()
        except ImportError:
            pass
    
    print("🚀 FaithBench Hallucination Detection Benchmark Runner")
    print("=" * 60)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    print(f"  requests library: {'✅' if HAS_REQUESTS else '❌'}")
    
    if args.use_api:
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        print(f"  OpenAI API key: {'✅' if has_openai else '❌'}")
        print(f"  Anthropic API key: {'✅' if has_anthropic else '❌'}")
        
        if not (has_openai or has_anthropic):
            print("\n⚠️  No API keys found. Running in local-only mode.")
            args.use_api = False
    
    try:
        # Load data
        data = load_faithbench_data(sample_size=args.sample_size)
        
        if not data:
            print("❌ Failed to load any FaithBench data")
            return
        
        # Run benchmark
        results = run_faithbench_benchmark(data, use_api=args.use_api, sample_size=args.sample_size)
        
        # Analyze results
        analyze_faithbench_results(results)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.save_results}")
        
        print("\n" + "=" * 60)
        print("✅ FaithBench benchmark evaluation completed!")
        print("\n💡 Key Insights:")
        print("  - Faithfulness-coherence evaluation detects hallucinations")
        print("  - Multi-response evaluation reveals inconsistency patterns")
        print("  - Source-summary coherence indicates faithfulness quality")
        print("  - Span-level analysis identifies specific problem areas")
        
        print("\n🚀 Next steps:")
        print("  - Try with API providers for multi-response evaluation")
        print("  - Experiment with different aggregation strategies")
        print("  - Compare faithfulness vs coherence correlation")
        print("  - Integrate with real FaithBench dataset from GitHub")
        
    except Exception as e:
        print(f"\n❌ FaithBench benchmark run failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install requests")
        print("  2. Set API keys for multi-response evaluation")
        print("  3. Check GitHub repository for actual FaithBench data")
        raise


if __name__ == "__main__":
    main()