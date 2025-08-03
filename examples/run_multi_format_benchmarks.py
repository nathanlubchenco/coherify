#!/usr/bin/env python3
"""
Multi-Format Benchmark Runner

This script demonstrates running multiple benchmark formats with Coherify,
showcasing multi-response coherence evaluation across different task types.

Usage:
    python examples/run_multi_format_benchmarks.py [options]
"""

import os
import json
import argparse
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import required libraries
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not installed. Install with: pip install datasets")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not installed. Install with: pip install numpy")

from coherify import (
    PropositionSet, Proposition,
    SemanticCoherence, HybridCoherence,
    setup_providers, get_provider
)

from coherify.measures.multi_response import (
    TemperatureVarianceCoherence, SelfConsistencyCoherence,
    MultiResponseConfig
)

from coherify.benchmarks.multi_format_adapters import (
    GSM8KAdapter, HellaSwagAdapter, MMLUAdapter,
    MultiResponseBenchmarkConfig
)

from coherify.benchmarks.fever_adapter import (
    FEVERAdapter, FEVERConfig, EvidenceBasedCoherence
)


class MultiBenchmarkRunner:
    """Runner for multiple benchmark formats with coherence evaluation."""
    
    def __init__(self, use_api: bool = False, verbose: bool = False):
        self.use_api = use_api
        self.verbose = verbose
        self.provider = None
        
        if use_api:
            try:
                setup_providers()
                self.provider = get_provider()
                print(f"🌐 Using API provider: {self.provider.provider_name}")
            except Exception as e:
                print(f"⚠️  Failed to setup API provider: {e}")
                self.use_api = False
    
    def run_gsm8k_benchmark(self, sample_size: int = 20) -> Dict[str, Any]:
        """Run GSM8K mathematical reasoning benchmark."""
        print(f"\n🧮 Running GSM8K Mathematical Reasoning Benchmark")
        print("=" * 60)
        
        # Load GSM8K data
        data = self._load_gsm8k_data(sample_size)
        if not data:
            return {"error": "Failed to load GSM8K data"}
        
        # Setup adapter
        config = MultiResponseBenchmarkConfig(
            enable_multi_response=self.use_api,
            num_responses_per_sample=3,
            reasoning_trace_enabled=True,
            max_response_length=1024
        )
        
        adapter = GSM8KAdapter(config=config, provider=self.provider)
        
        # Setup coherence measures
        measures = [
            HybridCoherence(),
            TemperatureVarianceCoherence(provider=self.provider),
            SelfConsistencyCoherence(provider=self.provider)
        ]
        
        # Run evaluation
        results = self._evaluate_benchmark(data, adapter, measures, "GSM8K")
        
        # Analyze mathematical reasoning coherence
        math_analysis = self._analyze_mathematical_coherence(results)
        results["mathematical_analysis"] = math_analysis
        
        return results
    
    def run_hellaswag_benchmark(self, sample_size: int = 50) -> Dict[str, Any]:
        """Run HellaSwag commonsense reasoning benchmark."""
        print(f"\n🤔 Running HellaSwag Commonsense Reasoning Benchmark")
        print("=" * 60)
        
        # Load HellaSwag data
        data = self._load_hellaswag_data(sample_size)
        if not data:
            return {"error": "Failed to load HellaSwag data"}
        
        # Setup adapter
        config = MultiResponseBenchmarkConfig(
            enable_multi_response=self.use_api,
            num_responses_per_sample=4,  # Match number of choices
            temperature_range=(0.2, 0.6),
            use_self_consistency=True
        )
        
        adapter = HellaSwagAdapter(config=config, provider=self.provider)
        
        # Setup coherence measures
        measures = [
            SemanticCoherence(),
            HybridCoherence(),
            SelfConsistencyCoherence(provider=self.provider)
        ]
        
        # Run evaluation
        results = self._evaluate_benchmark(data, adapter, measures, "HellaSwag")
        
        # Analyze commonsense coherence
        commonsense_analysis = self._analyze_commonsense_coherence(results)
        results["commonsense_analysis"] = commonsense_analysis
        
        return results
    
    def run_mmlu_benchmark(self, sample_size: int = 30, subjects: List[str] = None) -> Dict[str, Any]:
        """Run MMLU knowledge consistency benchmark."""
        print(f"\n📚 Running MMLU Knowledge Consistency Benchmark")
        print("=" * 60)
        
        if subjects is None:
            subjects = ["college_biology", "high_school_chemistry", "elementary_mathematics"]
        
        # Load MMLU data across subjects
        data = self._load_mmlu_data(sample_size, subjects)
        if not data:
            return {"error": "Failed to load MMLU data"}
        
        # Setup adapter
        config = MultiResponseBenchmarkConfig(
            enable_multi_response=self.use_api,
            num_responses_per_sample=3,
            temperature_range=(0.1, 0.5),
            reasoning_trace_enabled=True
        )
        
        adapter = MMLUAdapter(config=config, provider=self.provider)
        
        # Setup coherence measures
        measures = [
            HybridCoherence(),
            TemperatureVarianceCoherence(provider=self.provider)
        ]
        
        # Run evaluation
        results = self._evaluate_benchmark(data, adapter, measures, "MMLU")
        
        # Analyze cross-domain coherence
        domain_analysis = self._analyze_cross_domain_coherence(results, subjects)
        results["domain_analysis"] = domain_analysis
        
        return results
    
    def _load_gsm8k_data(self, sample_size: int) -> List[Dict[str, Any]]:
        """Load GSM8K dataset."""
        if HAS_DATASETS:
            try:
                print("  📥 Loading GSM8K from Hugging Face...")
                dataset = load_dataset("gsm8k", "main")
                data = dataset["test"].select(range(min(sample_size, len(dataset["test"]))))
                print(f"  ✅ Loaded {len(data)} GSM8K samples")
                return list(data)
            except Exception as e:
                print(f"  ❌ Failed to load GSM8K: {e}")
        
        # Fallback mock data
        print("  🔧 Using mock GSM8K data...")
        mock_data = [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She sells them for $2 each, so she makes 9 * 2 = $18. #### 18"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts are needed?",
                "answer": "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, so 2 / 2 = 1 bolt of white fiber. In total, 2 + 1 = 3 bolts are needed. #### 3"
            }
        ]
        return mock_data[:sample_size]
    
    def _load_hellaswag_data(self, sample_size: int) -> List[Dict[str, Any]]:
        """Load HellaSwag dataset."""
        if HAS_DATASETS:
            try:
                print("  📥 Loading HellaSwag from Hugging Face...")
                dataset = load_dataset("hellaswag")
                data = dataset["validation"].select(range(min(sample_size, len(dataset["validation"]))))
                print(f"  ✅ Loaded {len(data)} HellaSwag samples")
                return list(data)
            except Exception as e:
                print(f"  ❌ Failed to load HellaSwag: {e}")
        
        # Fallback mock data
        print("  🔧 Using mock HellaSwag data...")
        mock_data = [
            {
                "ctx": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
                "endings": [
                    "rinses the bucket off with soap and blow dry the dog.",
                    "uses a hose to keep washing the dog.",
                    "gets the dog wet, then it runs away again.",
                    "gets into a bathtub with the dog."
                ],
                "label": 2
            },
            {
                "ctx": "A man is standing on a ladder cleaning windows. He",
                "endings": [
                    "is standing on a sponge.",
                    "is holding a squeegee.",
                    "is wearing a yellow hat.",
                    "is cleaning the ceiling."
                ],
                "label": 1
            }
        ]
        return mock_data[:sample_size]
    
    def _load_mmlu_data(self, sample_size: int, subjects: List[str]) -> List[Dict[str, Any]]:
        """Load MMLU dataset across multiple subjects."""
        all_data = []
        
        if HAS_DATASETS:
            for subject in subjects:
                try:
                    print(f"  📥 Loading MMLU {subject}...")
                    dataset = load_dataset("cais/mmlu", subject)
                    subject_data = dataset["test"].select(range(min(sample_size // len(subjects), len(dataset["test"]))))
                    
                    # Add subject info to each sample
                    for sample in subject_data:
                        sample["subject"] = subject
                    
                    all_data.extend(list(subject_data))
                    print(f"  ✅ Loaded {len(subject_data)} {subject} samples")
                except Exception as e:
                    print(f"  ❌ Failed to load MMLU {subject}: {e}")
        
        if not all_data:
            # Fallback mock data
            print("  🔧 Using mock MMLU data...")
            mock_data = [
                {
                    "question": "Which of the following is the basic unit of life?",
                    "choices": ["Atom", "Molecule", "Cell", "Tissue"],
                    "answer": 2,
                    "subject": "biology"
                },
                {
                    "question": "What is the chemical symbol for gold?",
                    "choices": ["Go", "Gd", "Au", "Ag"],
                    "answer": 2,
                    "subject": "chemistry"
                }
            ]
            all_data = mock_data
        
        return all_data[:sample_size]
    
    def _evaluate_benchmark(self, 
                          data: List[Dict[str, Any]], 
                          adapter, 
                          measures: List, 
                          benchmark_name: str) -> Dict[str, Any]:
        """Evaluate benchmark with coherence measures."""
        print(f"  🏃 Evaluating {len(data)} {benchmark_name} samples...")
        
        results = {
            "benchmark_name": benchmark_name,
            "num_samples": len(data),
            "measures": {},
            "sample_results": [],
            "multi_response_results": [],
            "evaluation_time": 0.0
        }
        
        start_time = time.time()
        
        # Process each sample
        for i, sample in enumerate(data):
            if self.verbose:
                print(f"    Processing sample {i+1}/{len(data)}...")
            
            try:
                # Standard adaptation
                prop_set = adapter.adapt_single(sample)
                
                # Multi-response adaptation if enabled
                multi_result = None
                if adapter.config.enable_multi_response and self.provider:
                    try:
                        multi_result = adapter.adapt_single_with_multi_response(sample)
                        results["multi_response_results"].append(multi_result)
                    except Exception as e:
                        if self.verbose:
                            print(f"      Multi-response failed: {e}")
                
                # Evaluate with each coherence measure
                sample_coherence = {}
                for measure in measures:
                    try:
                        coherence_result = measure.compute(prop_set)
                        sample_coherence[measure.__class__.__name__] = coherence_result.score
                    except Exception as e:
                        if self.verbose:
                            print(f"      Measure {measure.__class__.__name__} failed: {e}")
                        sample_coherence[measure.__class__.__name__] = 0.0
                
                results["sample_results"].append({
                    "sample_index": i,
                    "coherence_scores": sample_coherence,
                    "proposition_count": len(prop_set.propositions)
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"      Sample {i} failed: {e}")
                continue
        
        results["evaluation_time"] = time.time() - start_time
        
        # Aggregate results by measure
        for measure in measures:
            measure_name = measure.__class__.__name__
            scores = [r["coherence_scores"].get(measure_name, 0.0) for r in results["sample_results"]]
            
            if scores:
                results["measures"][measure_name] = {
                    "mean_score": float(np.mean(scores)) if HAS_NUMPY else sum(scores) / len(scores),
                    "std_score": float(np.std(scores)) if HAS_NUMPY else 0.0,
                    "min_score": float(min(scores)),
                    "max_score": float(max(scores)),
                    "num_samples": len(scores)
                }
        
        print(f"  ✅ Completed in {results['evaluation_time']:.2f}s")
        return results
    
    def _analyze_mathematical_coherence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence specific to mathematical reasoning."""
        analysis = {
            "reasoning_consistency": 0.0,
            "numerical_accuracy": 0.0,
            "step_coherence": 0.0
        }
        
        multi_results = results.get("multi_response_results", [])
        if not multi_results:
            return analysis
        
        consistent_count = 0
        accurate_count = 0
        
        for result in multi_results:
            evaluation = result.get("response_evaluation", {})
            
            # Check self-consistency in mathematical reasoning
            if evaluation.get("is_consistent", False):
                consistent_count += 1
            
            # Check numerical accuracy
            if evaluation.get("accuracy", 0.0) > 0.8:
                accurate_count += 1
        
        total = len(multi_results)
        if total > 0:
            analysis["reasoning_consistency"] = consistent_count / total
            analysis["numerical_accuracy"] = accurate_count / total
        
        return analysis
    
    def _analyze_commonsense_coherence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence specific to commonsense reasoning."""
        analysis = {
            "choice_consistency": 0.0,
            "reasoning_quality": 0.0,
            "confidence_stability": 0.0
        }
        
        multi_results = results.get("multi_response_results", [])
        if not multi_results:
            return analysis
        
        consistent_choices = 0
        high_confidence = 0
        
        for result in multi_results:
            evaluation = result.get("response_evaluation", {})
            
            # Check choice consistency across responses
            if evaluation.get("is_consistent", False):
                consistent_choices += 1
            
            # Check confidence indicators
            consistency_score = evaluation.get("consistency_score", 0.0)
            if consistency_score > 0.7:
                high_confidence += 1
        
        total = len(multi_results)
        if total > 0:
            analysis["choice_consistency"] = consistent_choices / total
            analysis["confidence_stability"] = high_confidence / total
        
        return analysis
    
    def _analyze_cross_domain_coherence(self, 
                                      results: Dict[str, Any], 
                                      subjects: List[str]) -> Dict[str, Any]:
        """Analyze coherence across knowledge domains."""
        analysis = {
            "domain_consistency": {},
            "cross_domain_stability": 0.0,
            "subject_coherence_scores": {}
        }
        
        # Group results by subject
        sample_results = results.get("sample_results", [])
        multi_results = results.get("multi_response_results", [])
        
        subject_scores = {}
        for subject in subjects:
            subject_scores[subject] = []
        
        # Aggregate scores by subject
        for i, multi_result in enumerate(multi_results):
            if i < len(sample_results):
                sample = multi_result.get("sample", {})
                subject = sample.get("subject", "unknown")
                
                if subject in subject_scores:
                    coherence_scores = sample_results[i]["coherence_scores"]
                    avg_score = sum(coherence_scores.values()) / len(coherence_scores) if coherence_scores else 0.0
                    subject_scores[subject].append(avg_score)
        
        # Compute per-subject statistics
        for subject, scores in subject_scores.items():
            if scores:
                analysis["subject_coherence_scores"][subject] = {
                    "mean": float(np.mean(scores)) if HAS_NUMPY else sum(scores) / len(scores),
                    "std": float(np.std(scores)) if HAS_NUMPY else 0.0,
                    "count": len(scores)
                }
        
        # Compute cross-domain stability
        subject_means = [stats["mean"] for stats in analysis["subject_coherence_scores"].values()]
        if len(subject_means) > 1:
            analysis["cross_domain_stability"] = 1.0 - (float(np.std(subject_means)) if HAS_NUMPY else 0.0)
        
        return analysis
    
    def run_fever_benchmark(self, sample_size: int = 15) -> Dict[str, Any]:
        """Run FEVER fact-checking benchmark."""
        print(f"\n🔍 Running FEVER Fact-Checking Benchmark")
        print("=" * 60)
        
        # Load FEVER data (using mock data for now)
        data = self._load_fever_data(sample_size)
        if not data:
            return {"error": "Failed to load FEVER data"}
        
        # Setup adapter
        config = FEVERConfig(
            enable_multi_response=self.use_api,
            num_responses_per_sample=3,
            temperature_range=(0.1, 0.6),
            reasoning_trace_enabled=True,
            evidence_coherence_weight=0.7
        )
        
        adapter = FEVERAdapter(config=config, provider=self.provider)
        
        # Setup coherence measures
        measures = [
            HybridCoherence(),
            EvidenceBasedCoherence(provider=self.provider)
        ]
        
        if self.use_api:
            measures.append(TemperatureVarianceCoherence(provider=self.provider))
        
        # Run evaluation
        results = self._evaluate_benchmark(data, adapter, measures, "FEVER")
        
        # Analyze fact-checking coherence
        factcheck_analysis = self._analyze_factchecking_coherence(results)
        results["factchecking_analysis"] = factcheck_analysis
        
        return results
    
    def _load_fever_data(self, sample_size: int) -> List[Dict[str, Any]]:
        """Load FEVER dataset (mock for now)."""
        print("  📥 Loading FEVER data...")
        
        # Create comprehensive mock FEVER data
        mock_data = [
            {
                "id": 1,
                "claim": "Barack Obama was the 44th President of the United States.",
                "label": "SUPPORTS",
                "evidence": [[[101, 1001, "Barack_Obama", 0]]]
            },
            {
                "id": 2,
                "claim": "The Earth is flat and has no curvature.",
                "label": "REFUTES", 
                "evidence": [[[102, 2001, "Earth", 5]]]
            },
            {
                "id": 3,
                "claim": "John Smith ate breakfast this morning at 7:30 AM.",
                "label": "NOT ENOUGH INFO",
                "evidence": [[[103, 3001, None, None]]]
            },
            {
                "id": 4,
                "claim": "Albert Einstein developed the theory that led to nuclear weapons.",
                "label": "SUPPORTS",
                "evidence": [[[104, 4001, "Albert_Einstein", 3], [104, 4002, "E=mc²", 0]]]
            },
            {
                "id": 5,
                "claim": "Water boils at 100 degrees Celsius at sea level pressure.",
                "label": "SUPPORTS",
                "evidence": [[[105, 5001, "Water", 8], [105, 5002, "Boiling_point", 2]]]
            }
        ]
        
        print(f"  ✅ Using {min(len(mock_data), sample_size)} FEVER samples (mock data)")
        return mock_data[:sample_size]
    
    def _analyze_factchecking_coherence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence specific to fact-checking tasks."""
        analysis = {
            "evidence_consistency": 0.0,
            "claim_verification_coherence": 0.0,
            "multi_evidence_coherence": 0.0
        }
        
        multi_results = results.get("multi_response_results", [])
        if not multi_results:
            return analysis
        
        evidence_consistent_count = 0
        high_coherence_count = 0
        
        for result in multi_results:
            evaluation = result.get("response_evaluation", {})
            
            # Check evidence consistency
            if evaluation.get("evidence_consistency", 0.0) > 0.6:
                evidence_consistent_count += 1
            
            # Check overall coherence
            if evaluation.get("fever_score", 0.0) > 0.7:
                high_coherence_count += 1
        
        total = len(multi_results)
        if total > 0:
            analysis["evidence_consistency"] = evidence_consistent_count / total
            analysis["claim_verification_coherence"] = high_coherence_count / total
        
        return analysis
    
    def run_comparative_analysis(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run comparative analysis across all benchmarks."""
        print(f"\n📊 Running Comparative Analysis Across Benchmarks")
        print("=" * 60)
        
        analysis = {
            "benchmark_comparison": {},
            "measure_performance": {},
            "coherence_insights": {},
            "recommendations": []
        }
        
        # Compare performance across benchmarks
        for benchmark_name, results in all_results.items():
            if "error" in results:
                continue
            
            measures = results.get("measures", {})
            analysis["benchmark_comparison"][benchmark_name] = {
                "num_samples": results.get("num_samples", 0),
                "evaluation_time": results.get("evaluation_time", 0.0),
                "avg_coherence": sum(m.get("mean_score", 0) for m in measures.values()) / len(measures) if measures else 0.0
            }
        
        # Compare measure performance across benchmarks
        all_measure_names = set()
        for results in all_results.values():
            if "measures" in results:
                all_measure_names.update(results["measures"].keys())
        
        for measure_name in all_measure_names:
            measure_scores = []
            for benchmark_name, results in all_results.items():
                if "measures" in results and measure_name in results["measures"]:
                    score = results["measures"][measure_name].get("mean_score", 0.0)
                    measure_scores.append((benchmark_name, score))
            
            if measure_scores:
                analysis["measure_performance"][measure_name] = {
                    "scores_by_benchmark": dict(measure_scores),
                    "overall_mean": sum(score for _, score in measure_scores) / len(measure_scores),
                    "score_range": max(score for _, score in measure_scores) - min(score for _, score in measure_scores)
                }
        
        # Generate insights and recommendations
        insights = []
        recommendations = []
        
        # Find best-performing measure
        if analysis["measure_performance"]:
            best_measure = max(analysis["measure_performance"].items(), 
                             key=lambda x: x[1]["overall_mean"])
            insights.append(f"Best overall measure: {best_measure[0]} (avg score: {best_measure[1]['overall_mean']:.3f})")
            recommendations.append(f"Use {best_measure[0]} for general coherence evaluation")
        
        # Identify benchmark-specific patterns
        for benchmark_name, comparison in analysis["benchmark_comparison"].items():
            if comparison["avg_coherence"] > 0.7:
                insights.append(f"{benchmark_name} shows high coherence (avg: {comparison['avg_coherence']:.3f})")
            elif comparison["avg_coherence"] < 0.4:
                insights.append(f"{benchmark_name} shows low coherence (avg: {comparison['avg_coherence']:.3f})")
                recommendations.append(f"Consider multi-response evaluation for {benchmark_name}")
        
        analysis["coherence_insights"] = insights
        analysis["recommendations"] = recommendations
        
        return analysis


def main():
    """Main multi-benchmark runner."""
    parser = argparse.ArgumentParser(description="Run multi-format benchmarks with Coherify")
    parser.add_argument("--benchmarks", nargs="+", 
                       choices=["gsm8k", "hellaswag", "mmlu", "fever", "all"],
                       default=["all"],
                       help="Benchmarks to run")
    parser.add_argument("--use-api", action="store_true", 
                       help="Use API-enhanced multi-response evaluation")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Number of samples per benchmark")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed progress")
    parser.add_argument("--save-results", type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Enable clean output unless verbose
    if not args.verbose:
        try:
            from coherify.utils.clean_output import enable_clean_output
            enable_clean_output()
        except ImportError:
            pass
    
    print("🚀 Multi-Format Benchmark Runner")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    print(f"  datasets library: {'✅' if HAS_DATASETS else '❌'}")
    print(f"  numpy library: {'✅' if HAS_NUMPY else '❌'}")
    
    if args.use_api:
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        print(f"  OpenAI API key: {'✅' if has_openai else '❌'}")
        print(f"  Anthropic API key: {'✅' if has_anthropic else '❌'}")
        
        if not (has_openai or has_anthropic):
            print("\n⚠️  No API keys found. Running in local-only mode.")
            args.use_api = False
    
    # Initialize runner
    runner = MultiBenchmarkRunner(use_api=args.use_api, verbose=args.verbose)
    
    # Determine which benchmarks to run
    benchmarks_to_run = args.benchmarks
    if "all" in benchmarks_to_run:
        benchmarks_to_run = ["gsm8k", "hellaswag", "mmlu", "fever"]
    
    # Run benchmarks
    all_results = {}
    
    try:
        if "gsm8k" in benchmarks_to_run:
            all_results["GSM8K"] = runner.run_gsm8k_benchmark(args.sample_size)
        
        if "hellaswag" in benchmarks_to_run:
            all_results["HellaSwag"] = runner.run_hellaswag_benchmark(args.sample_size)
        
        if "mmlu" in benchmarks_to_run:
            all_results["MMLU"] = runner.run_mmlu_benchmark(args.sample_size)
        
        if "fever" in benchmarks_to_run:
            all_results["FEVER"] = runner.run_fever_benchmark(args.sample_size)
        
        # Run comparative analysis
        if len(all_results) > 1:
            comparative_analysis = runner.run_comparative_analysis(all_results)
            all_results["Comparative_Analysis"] = comparative_analysis
            
            # Print summary
            print(f"\n🎯 Key Insights:")
            for insight in comparative_analysis.get("coherence_insights", []):
                print(f"  • {insight}")
            
            print(f"\n💡 Recommendations:")
            for rec in comparative_analysis.get("recommendations", []):
                print(f"  • {rec}")
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.save_results}")
        
        print("\n" + "=" * 50)
        print("✅ Multi-format benchmark evaluation completed!")
        print("\n🚀 Next steps:")
        print("  - Try different coherence measures")
        print("  - Enable API multi-response evaluation")
        print("  - Analyze results across task types")
        print("  - Compare coherence patterns between benchmarks")
        
    except Exception as e:
        print(f"\n❌ Benchmark run failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install datasets numpy")
        print("  2. Set API keys for multi-response evaluation")
        print("  3. Check internet connection for dataset downloads")
        raise


if __name__ == "__main__":
    main()