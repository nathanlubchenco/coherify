#!/usr/bin/env python3
"""
FEVER Benchmark Runner

This script demonstrates running the FEVER (Fact Extraction and VERification)
benchmark with Coherify's evidence-based coherence evaluation.

Usage:
    python examples/run_fever_benchmark.py [--use-api] [--sample-size N]
"""

import os
import json
import argparse
import time
from typing import Dict, List, Any, Optional

# Try to import required libraries
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not installed. Install with: pip install datasets")

from coherify import HybridCoherence, setup_providers, get_provider

from coherify.benchmarks.fever_adapter import (
    FEVERAdapter,
    FEVERConfig,
    EvidenceBasedCoherence,
)

from coherify.measures.multi_response import (
    TemperatureVarianceCoherence,
    SelfConsistencyCoherence,
)


def load_fever_data(sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load FEVER dataset."""
    print("📚 Loading FEVER data...")

    # Method 1: Use datasets library if available
    if HAS_DATASETS:
        try:
            print("  📥 Loading FEVER from Hugging Face...")
            dataset = load_dataset("kilt_tasks", "fever")
            data = dataset["validation"].select(
                range(min(sample_size or 10, len(dataset["validation"])))
            )
            print(f"  ✅ Loaded {len(data)} real FEVER samples")
            return list(data)

        except Exception as e:
            print(f"  ⚠️ Failed to load FEVER from datasets: {e}")

    # Fallback: Create mock FEVER data
    print("  🔧 Using mock FEVER data...")
    mock_data = create_comprehensive_fever_mock_data()

    if sample_size:
        mock_data = mock_data[:sample_size]

    print(f"  ✅ Using {len(mock_data)} mock FEVER samples")
    return mock_data


def create_comprehensive_fever_mock_data() -> List[Dict[str, Any]]:
    """Create comprehensive mock FEVER data covering different scenarios."""
    return [
        # SUPPORTS case - straightforward
        {
            "id": 1,
            "claim": "Barack Obama was the 44th President of the United States.",
            "label": "SUPPORTS",
            "evidence": [
                [[101, 1001, "Barack_Obama", 0], [101, 1002, "Barack_Obama", 1]]
            ],
        },
        # REFUTES case - clear contradiction
        {
            "id": 2,
            "claim": "The Earth is flat and has no curvature.",
            "label": "REFUTES",
            "evidence": [[[102, 2001, "Earth", 5], [102, 2002, "Earth", 12]]],
        },
        # NOT ENOUGH INFO case
        {
            "id": 3,
            "claim": "John Smith ate breakfast this morning at 7:30 AM.",
            "label": "NOT ENOUGH INFO",
            "evidence": [[[103, 3001, None, None]]],
        },
        # Multi-hop reasoning case - requires evidence composition
        {
            "id": 4,
            "claim": "Albert Einstein developed the theory that led to nuclear weapons.",
            "label": "SUPPORTS",
            "evidence": [
                [
                    [104, 4001, "Albert_Einstein", 3],
                    [104, 4002, "E=mc²", 0],
                    [104, 4003, "Nuclear_weapon", 8],
                ]
            ],
        },
        # Cross-document evidence case
        {
            "id": 5,
            "claim": "Leonardo da Vinci painted the Mona Lisa and designed flying machines.",
            "label": "SUPPORTS",
            "evidence": [
                [
                    [105, 5001, "Leonardo_da_Vinci", 2],
                    [105, 5002, "Mona_Lisa", 0],
                    [105, 5003, "Flying_machine", 4],
                ]
            ],
        },
        # Complex temporal reasoning
        {
            "id": 6,
            "claim": "World War II ended before the invention of the computer.",
            "label": "REFUTES",
            "evidence": [
                [
                    [106, 6001, "World_War_II", 15],
                    [106, 6002, "Computer", 3],
                    [106, 6003, "ENIAC", 1],
                ]
            ],
        },
        # Scientific claim requiring multiple evidence
        {
            "id": 7,
            "claim": "Water boils at 100 degrees Celsius at sea level pressure.",
            "label": "SUPPORTS",
            "evidence": [
                [
                    [107, 7001, "Water", 8],
                    [107, 7002, "Boiling_point", 2],
                    [107, 7003, "Atmospheric_pressure", 5],
                ]
            ],
        },
        # Ambiguous case that could go either way
        {
            "id": 8,
            "claim": "Most people prefer chocolate ice cream over vanilla.",
            "label": "NOT ENOUGH INFO",
            "evidence": [
                [[108, 8001, "Ice_cream", 12], [108, 8002, "Flavor_preference", 3]]
            ],
        },
    ]


def run_fever_benchmark(
    data: List[Dict[str, Any]], use_api: bool = False, sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Run FEVER benchmark evaluation."""
    print(f"\n🔍 Running FEVER Fact-Checking Benchmark")
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

    # Setup FEVER adapter
    config = FEVERConfig(
        enable_multi_response=use_api,
        num_responses_per_sample=3,
        temperature_range=(0.1, 0.6),
        reasoning_trace_enabled=True,
        evidence_coherence_weight=0.7,
        claim_coherence_weight=0.3,
    )

    adapter = FEVERAdapter(config=config, provider=provider)

    # Setup coherence measures
    measures = [
        HybridCoherence(),
        EvidenceBasedCoherence(provider=provider),
    ]

    if use_api:
        measures.extend(
            [
                TemperatureVarianceCoherence(provider=provider),
                SelfConsistencyCoherence(provider=provider),
            ]
        )

    print(f"  📊 Evaluating {len(data)} FEVER samples...")

    # Process each sample
    results = {
        "benchmark_name": "FEVER",
        "num_samples": len(data),
        "sample_results": [],
        "multi_response_results": [],
        "measures": {},
        "evaluation_time": 0.0,
        "fever_specific_metrics": {
            "label_accuracy": 0.0,
            "evidence_consistency": 0.0,
            "fever_score": 0.0,
        },
    }

    start_time = time.time()
    label_correct = 0
    evidence_consistency_scores = []
    fever_scores = []

    for i, sample in enumerate(data):
        print(
            f"    Processing sample {i+1}/{len(data)}: {sample.get('claim', '')[:60]}..."
        )

        try:
            # Standard adaptation
            prop_set = adapter.adapt_single(sample)

            # Multi-response adaptation if enabled
            multi_result = None
            if config.enable_multi_response and provider:
                try:
                    multi_result = adapter.adapt_single_with_multi_response(sample)
                    results["multi_response_results"].append(multi_result)

                    # Extract FEVER-specific metrics
                    if "response_evaluation" in multi_result:
                        eval_data = multi_result["response_evaluation"]
                        if eval_data.get("is_correct"):
                            label_correct += 1
                        evidence_consistency_scores.append(
                            eval_data.get("evidence_consistency", 0.0)
                        )
                        fever_scores.append(eval_data.get("fever_score", 0.0))

                except Exception as e:
                    print(f"      Multi-response failed: {e}")

            # Evaluate with each coherence measure
            sample_coherence = {}
            for measure in measures:
                try:
                    if isinstance(measure, EvidenceBasedCoherence):
                        # Special handling for evidence-based coherence
                        fever_sample = adapter._parse_fever_sample(sample)
                        evidence_sentences = []
                        for evidence_set in fever_sample.evidence_sets:
                            evidence_sentences.extend(evidence_set.evidence_sentences)

                        evidence_result = measure.evaluate_evidence_coherence(
                            fever_sample.claim,
                            evidence_sentences,
                            f"FEVER fact-checking: {fever_sample.label}",
                        )
                        sample_coherence[measure.__class__.__name__] = evidence_result[
                            "overall_coherence"
                        ]
                    else:
                        coherence_result = measure.compute(prop_set)
                        sample_coherence[measure.__class__.__name__] = (
                            coherence_result.score
                        )

                except Exception as e:
                    print(f"      Measure {measure.__class__.__name__} failed: {e}")
                    sample_coherence[measure.__class__.__name__] = 0.0

            results["sample_results"].append(
                {
                    "sample_index": i,
                    "claim": sample.get("claim", ""),
                    "label": sample.get("label", ""),
                    "coherence_scores": sample_coherence,
                    "proposition_count": len(prop_set.propositions),
                }
            )

        except Exception as e:
            print(f"      Sample {i} failed: {e}")
            continue

    results["evaluation_time"] = time.time() - start_time

    # Aggregate results by measure
    for measure in measures:
        measure_name = measure.__class__.__name__
        scores = [
            r["coherence_scores"].get(measure_name, 0.0)
            for r in results["sample_results"]
        ]

        if scores:
            import numpy as np

            results["measures"][measure_name] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(min(scores)),
                "max_score": float(max(scores)),
                "num_samples": len(scores),
            }

    # Compute FEVER-specific metrics
    if results["multi_response_results"]:
        results["fever_specific_metrics"] = {
            "label_accuracy": label_correct / len(data),
            "evidence_consistency": (
                float(np.mean(evidence_consistency_scores))
                if evidence_consistency_scores
                else 0.0
            ),
            "fever_score": float(np.mean(fever_scores)) if fever_scores else 0.0,
            "num_multi_response_samples": len(results["multi_response_results"]),
        }

    print(f"  ✅ Completed in {results['evaluation_time']:.2f}s")
    return results


def analyze_fever_results(results: Dict[str, Any]):
    """Analyze and display FEVER benchmark results."""
    print("\n📊 FEVER Benchmark Analysis:")
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

    # FEVER-specific metrics
    fever_metrics = results.get("fever_specific_metrics", {})
    if fever_metrics:
        print(f"\n🔍 FEVER-Specific Metrics:")
        label_accuracy = fever_metrics.get('label_accuracy', 0.0)
        print(f"  Label Accuracy: {label_accuracy:.1%}")
        print(
            f"  Evidence Consistency: {fever_metrics.get('evidence_consistency', 0.0):.3f}"
        )
        print(f"  FEVER Score: {fever_metrics.get('fever_score', 0.0):.3f}")
        print(
            f"  Multi-response samples: {fever_metrics.get('num_multi_response_samples', 0)}"
        )
        
        # Add performance validation for FEVER
        try:
            from coherify.benchmarks.native_metrics import BenchmarkPerformanceExpectations
            
            is_realistic, explanation = BenchmarkPerformanceExpectations.is_performance_realistic(
                "fever", label_accuracy
            )
            
            if not is_realistic:
                print(f"  ⚠️  Performance Warning: {explanation}")
            elif label_accuracy > 0:
                expectations = BenchmarkPerformanceExpectations.get_expectations("fever")
                if expectations:
                    best_model = expectations.get("best_model", 0)
                    print(f"  ℹ️  Research Context: Best published result ~{best_model:.1%}")
        except ImportError:
            pass  # Skip validation if not available

    # Sample-level analysis
    print(f"\n📋 Sample Analysis:")
    sample_results = results.get("sample_results", [])

    # Group by label
    label_groups = {}
    for sample in sample_results:
        label = sample.get("label", "UNKNOWN")
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(sample)

    for label, samples in label_groups.items():
        avg_coherence = sum(
            (
                sum(s["coherence_scores"].values()) / len(s["coherence_scores"])
                if s["coherence_scores"]
                else 0.0
            )
            for s in samples
        ) / len(samples)

        print(f"  {label}: {len(samples)} samples, avg coherence {avg_coherence:.3f}")

    # Multi-response analysis
    multi_results = results.get("multi_response_results", [])
    if multi_results:
        print(f"\n🔄 Multi-Response Analysis:")
        consistent_count = 0
        evidence_usage_count = 0

        for result in multi_results:
            eval_data = result.get("response_evaluation", {})
            if eval_data.get("is_label_consistent"):
                consistent_count += 1
            if eval_data.get("evidence_consistency", 0.0) > 0.5:
                evidence_usage_count += 1

        print(
            f"  Label consistency: {consistent_count}/{len(multi_results)} ({consistent_count/len(multi_results):.1%})"
        )
        print(
            f"  Good evidence usage: {evidence_usage_count}/{len(multi_results)} ({evidence_usage_count/len(multi_results):.1%})"
        )


def main():
    """Main FEVER benchmark runner."""
    parser = argparse.ArgumentParser(description="Run FEVER benchmark with Coherify")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API-enhanced multi-response evaluation",
    )
    parser.add_argument(
        "--sample-size", type=int, default=8, help="Number of samples to evaluate"
    )
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

    print("🚀 FEVER Fact-Checking Benchmark Runner")
    print("=" * 50)

    # Check dependencies
    print("🔍 Checking dependencies...")
    print(f"  datasets library: {'✅' if HAS_DATASETS else '❌'}")

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
        data = load_fever_data(sample_size=args.sample_size)

        if not data:
            print("❌ Failed to load any FEVER data")
            return

        # Run benchmark
        results = run_fever_benchmark(
            data, use_api=args.use_api, sample_size=args.sample_size
        )

        # Analyze results
        analyze_fever_results(results)

        # Save results if requested
        if args.save_results:
            with open(args.save_results, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.save_results}")

        print("\n" + "=" * 50)
        print("✅ FEVER benchmark evaluation completed!")
        print("\n💡 Key Insights:")
        print("  - Evidence-based coherence enables better fact-checking")
        print("  - Multi-response evaluation detects uncertainty in claims")
        print("  - Temperature variance reveals model confidence patterns")
        print("  - Cross-evidence coherence identifies reasoning consistency")

        print("\n🚀 Next steps:")
        print("  - Try with API providers for multi-response evaluation")
        print("  - Experiment with different evidence coherence weights")
        print("  - Compare results across different fact-checking domains")
        print("  - Integrate with real Wikipedia evidence retrieval")

    except Exception as e:
        print(f"\n❌ FEVER benchmark run failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install datasets")
        print("  2. Set API keys for multi-response evaluation")
        print("  3. Check internet connection for dataset downloads")
        raise


if __name__ == "__main__":
    main()
