#!/usr/bin/env python3
"""
FEVER Benchmark Runner - FIXED VERSION

This script properly implements the 3-stage research pipeline:
1. Baseline FEVER evaluation (single response)
2. K-pass majority voting
3. Coherence-enhanced selection

The goal is to IMPROVE FEVER accuracy, not just measure coherence.
"""

import os
import json
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import random

# Import Coherify components
from coherify.generation.model_runner import ModelRunner, GenerationResult
from coherify.evaluators.response_selectors import (
    MajorityVotingSelector,
    CoherenceSelector
)
from coherify.evaluators.hybrid_selectors import HybridCoherenceConsistencySelector
from coherify.generation.temperature_strategies import AdaptiveTemperatureSelector

# Try to import datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not installed. Install with: pip install datasets")


class FEVERSample:
    """Single FEVER sample with claim and evidence."""
    
    def __init__(self, data: Dict[str, Any]):
        self.claim = data.get("claim", "")
        self.label = data.get("label", "NOT ENOUGH INFO")
        self.evidence = data.get("evidence", [])
        self.id = data.get("id", "")


class FEVEREvaluator:
    """
    Proper FEVER evaluator that measures fact-checking accuracy.
    
    This evaluator:
    1. Generates actual predictions for FEVER labels
    2. Compares different selection strategies
    3. Reports actual FEVER accuracy (not just coherence)
    """
    
    LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    
    def __init__(self, model_runner: Optional[ModelRunner] = None):
        """
        Initialize FEVER evaluator.
        
        Args:
            model_runner: ModelRunner for generating predictions
        """
        self.model_runner = model_runner
        self.temp_selector = AdaptiveTemperatureSelector()
        
        # Initialize selectors for Stage 2 and 3
        self.majority_selector = MajorityVotingSelector()
        self.coherence_selector = CoherenceSelector()
        self.hybrid_selector = HybridCoherenceConsistencySelector(alpha=0.6)
    
    def create_prompt(self, sample: FEVERSample) -> str:
        """Create a prompt for FEVER fact-checking."""
        prompt = f"""You are a fact-checker. Given a claim, determine if it is SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Claim: {sample.claim}

Based on your knowledge, does the evidence SUPPORT the claim, REFUTE the claim, or is there NOT ENOUGH INFO?

Answer with exactly one of: SUPPORTS, REFUTES, NOT ENOUGH INFO

Your answer:"""
        return prompt
    
    def extract_label(self, response: str) -> str:
        """Extract FEVER label from model response."""
        response_upper = response.upper()
        
        # Check for exact matches first
        for label in self.LABELS:
            if label in response_upper:
                return label
        
        # Fallback
        return "NOT ENOUGH INFO"
    
    def evaluate_single(self, sample: FEVERSample) -> str:
        """
        Stage 1: Generate single prediction.
        
        Returns:
            Predicted label
        """
        if not self.model_runner:
            # Mock prediction for testing
            return random.choice(self.LABELS)
        
        prompt = self.create_prompt(sample)
        result = self.model_runner.generate_response(prompt, temperature=0.3)
        
        return self.extract_label(result.text)
    
    def evaluate_majority_voting(self, sample: FEVERSample, k: int = 5) -> str:
        """
        Stage 2: Generate K predictions and use majority voting.
        
        Returns:
            Predicted label from majority voting
        """
        if not self.model_runner:
            # Mock predictions for testing
            predictions = [random.choice(self.LABELS) for _ in range(k)]
        else:
            prompt = self.create_prompt(sample)
            
            # Use adaptive temperatures for diversity
            temperatures = self.temp_selector.select_temperatures(prompt, k)
            results = []
            
            for temp in temperatures:
                result = self.model_runner.generate_response(prompt, temperature=temp)
                results.append(result)
            
            predictions = [self.extract_label(r.text) for r in results]
        
        # Majority voting
        vote_counts = Counter(predictions)
        winner = vote_counts.most_common(1)[0][0]
        
        return winner
    
    def evaluate_coherence_selection(self, sample: FEVERSample, k: int = 5) -> str:
        """
        Stage 3: Use coherence-based selection.
        
        Returns:
            Predicted label from best coherent response
        """
        if not self.model_runner:
            # Mock for testing
            return random.choice(self.LABELS)
        
        prompt = self.create_prompt(sample)
        
        # Generate K diverse responses
        responses = self.model_runner.generate_k_responses(
            prompt, k, temperature_strategy="adaptive"
        )
        
        # Select best response using hybrid selector
        result = self.hybrid_selector.select(responses, prompt)
        
        return self.extract_label(result.selected_response)
    
    def evaluate_dataset(self, 
                        samples: List[FEVERSample],
                        method: str = "single") -> Dict[str, Any]:
        """
        Evaluate entire dataset with specified method.
        
        Args:
            samples: List of FEVER samples
            method: Evaluation method ("single", "majority", "coherence")
            
        Returns:
            Evaluation results with accuracy metrics
        """
        correct = 0
        predictions = []
        
        print(f"\n🔍 Evaluating {len(samples)} samples with {method} method...")
        
        for i, sample in enumerate(samples):
            # Get prediction based on method
            if method == "single":
                pred = self.evaluate_single(sample)
            elif method == "majority":
                pred = self.evaluate_majority_voting(sample, k=5)
            elif method == "coherence":
                pred = self.evaluate_coherence_selection(sample, k=5)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            predictions.append(pred)
            
            # Check if correct
            if pred == sample.label:
                correct += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples...")
        
        # Calculate metrics
        accuracy = correct / len(samples) if samples else 0
        
        # Calculate per-label accuracy
        label_stats = {label: {"correct": 0, "total": 0} for label in self.LABELS}
        
        for sample, pred in zip(samples, predictions):
            label_stats[sample.label]["total"] += 1
            if pred == sample.label:
                label_stats[sample.label]["correct"] += 1
        
        label_accuracy = {}
        for label, stats in label_stats.items():
            if stats["total"] > 0:
                label_accuracy[label] = stats["correct"] / stats["total"]
            else:
                label_accuracy[label] = 0.0
        
        return {
            "method": method,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(samples),
            "label_accuracy": label_accuracy,
            "predictions": predictions
        }


def load_fever_data(sample_size: Optional[int] = None) -> List[FEVERSample]:
    """Load FEVER dataset."""
    print("📚 Loading FEVER data...")
    
    samples = []
    
    if HAS_DATASETS:
        try:
            print("  Loading FEVER data from local file or HuggingFace...")
            # Try to load from a preprocessed file first
            import os
            fever_path = "data/fever_validation.jsonl"
            
            if os.path.exists(fever_path):
                print(f"  Loading from local file: {fever_path}")
                # Read JSONL file (each line is a JSON object)
                with open(fever_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            samples.append(FEVERSample(item))
                            if sample_size and len(samples) >= sample_size:
                                break
            else:
                # Use mock data with more realistic examples
                print("  Note: Using high-quality mock FEVER data")
                print("  (Real FEVER dataset requires manual download)")
                raise Exception("Using mock data")
            
            print(f"  ✅ Loaded {len(samples)} FEVER samples")
            return samples
            
        except Exception as e:
            print(f"  ⚠️ Failed to load FEVER data: {e}")
    
    # Fallback to mock data
    print("  📦 Using mock FEVER data for testing")
    mock_data = [
        {
            "claim": "Paris is the capital of France",
            "label": "SUPPORTS",
            "evidence": [["Paris", "Paris is the capital and most populous city of France"]]
        },
        {
            "claim": "The Earth is flat",
            "label": "REFUTES",
            "evidence": [["Earth", "Earth is the third planet from the Sun and has a spherical shape"]]
        },
        {
            "claim": "Shakespeare wrote 100 plays",
            "label": "REFUTES",
            "evidence": [["Shakespeare", "Shakespeare wrote approximately 37-39 plays"]]
        },
        {
            "claim": "Water boils at 100 degrees Celsius at sea level",
            "label": "SUPPORTS",
            "evidence": [["Water", "At standard atmospheric pressure, water boils at 100°C"]]
        },
        {
            "claim": "The moon is made of cheese",
            "label": "REFUTES",
            "evidence": [["Moon", "The Moon is a rocky celestial body"]]
        }
    ]
    
    # Repeat to get desired sample size
    while len(samples) < (sample_size or 5):
        for data in mock_data:
            samples.append(FEVERSample(data))
            if sample_size and len(samples) >= sample_size:
                break
    
    return samples[:sample_size] if sample_size else samples


def compare_methods(evaluator: FEVEREvaluator, 
                   samples: List[FEVERSample]) -> Dict[str, Any]:
    """
    Compare all three methods on the same dataset.
    
    Returns:
        Comparison results
    """
    print("\n" + "="*60)
    print("FEVER 3-Stage Pipeline Comparison")
    print("="*60)
    
    results = {}
    
    # Stage 1: Baseline (single response)
    print("\n📊 Stage 1: Baseline Evaluation")
    results["baseline"] = evaluator.evaluate_dataset(samples, method="single")
    print(f"  Accuracy: {results['baseline']['accuracy']:.1%}")
    
    # Stage 2: Majority Voting (K=5)
    print("\n📊 Stage 2: Majority Voting (K=5)")
    results["majority"] = evaluator.evaluate_dataset(samples, method="majority")
    print(f"  Accuracy: {results['majority']['accuracy']:.1%}")
    
    # Stage 3: Coherence Selection (K=5)
    print("\n📊 Stage 3: Coherence-Enhanced Selection (K=5)")
    results["coherence"] = evaluator.evaluate_dataset(samples, method="coherence")
    print(f"  Accuracy: {results['coherence']['accuracy']:.1%}")
    
    # Calculate improvements
    baseline_acc = results["baseline"]["accuracy"]
    majority_improvement = results["majority"]["accuracy"] - baseline_acc
    coherence_improvement = results["coherence"]["accuracy"] - baseline_acc
    
    print("\n" + "="*60)
    print("📈 Performance Summary")
    print("="*60)
    print(f"Baseline:  {baseline_acc:.1%}")
    print(f"Majority:  {results['majority']['accuracy']:.1%} ({majority_improvement:+.1%})")
    print(f"Coherence: {results['coherence']['accuracy']:.1%} ({coherence_improvement:+.1%})")
    
    # Per-label breakdown
    print("\n📊 Per-Label Accuracy:")
    for label in FEVEREvaluator.LABELS:
        print(f"\n{label}:")
        for method in ["baseline", "majority", "coherence"]:
            acc = results[method]["label_accuracy"].get(label, 0)
            print(f"  {method}: {acc:.1%}")
    
    return results


def main():
    """Main FEVER benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run FEVER benchmark with 3-stage pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model to use (default, gpt4-mini, gpt4, claude)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--k-responses",
        type=int,
        default=5,
        help="Number of responses for majority voting and coherence selection"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison of all three methods"
    )
    parser.add_argument(
        "--method",
        choices=["single", "majority", "coherence"],
        default="single",
        help="Evaluation method to use"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    print("🚀 FEVER Fact-Checking Benchmark")
    print("="*50)
    
    # Load data
    samples = load_fever_data(args.sample_size)
    
    # Setup model runner if using real model
    model_runner = None
    if args.model != "mock":
        try:
            # Load config
            with open("config/benchmark_config.json") as f:
                config = json.load(f)
            
            # Map model names properly
            model_key = args.model
            if model_key == "gpt4o":
                model_key = "gpt4o"
            elif model_key == "gpt4":
                model_key = "gpt4"
            elif model_key == "gpt4-mini":
                model_key = "gpt4-mini"
                
            model_config = config["models"].get(model_key, config["models"]["default"])
            
            # Check for API key
            import os
            if model_config["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
                print("⚠️  Warning: OPENAI_API_KEY not set. Using mock predictions.")
                print("  To use real model, set: export OPENAI_API_KEY='your-key'")
                model_runner = None
            else:
                model_runner = ModelRunner(model_config)
                print(f"✅ Using model: {model_config.get('model', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Failed to setup model: {e}")
            print("  Using mock predictions")
            model_runner = None
    
    # Create evaluator
    evaluator = FEVEREvaluator(model_runner)
    
    # Run evaluation
    if args.compare:
        results = compare_methods(evaluator, samples)
    else:
        results = {
            args.method: evaluator.evaluate_dataset(samples, method=args.method)
        }
        
        print(f"\n📊 Results for {args.method} method:")
        print(f"  Accuracy: {results[args.method]['accuracy']:.1%}")
        print(f"  Correct: {results[args.method]['correct']}/{results[args.method]['total']}")
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.save_results}")
    
    print("\n✅ FEVER benchmark completed!")
    
    # Show key insights
    if args.compare:
        print("\n💡 Key Insights:")
        if results["majority"]["accuracy"] > results["baseline"]["accuracy"]:
            print("  ✅ Majority voting improves over baseline")
        if results["coherence"]["accuracy"] > results["majority"]["accuracy"]:
            print("  ✅ Coherence selection further improves accuracy")
        
        best_method = max(results.keys(), key=lambda k: results[k]["accuracy"])
        print(f"  🏆 Best method: {best_method} ({results[best_method]['accuracy']:.1%})")


if __name__ == "__main__":
    main()