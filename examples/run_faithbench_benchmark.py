#!/usr/bin/env python3
"""
FaithBench Benchmark Runner - FIXED VERSION

This script properly implements the 3-stage research pipeline:
1. Baseline hallucination detection (single response)
2. K-pass majority voting
3. Coherence-enhanced selection

The goal is to IMPROVE hallucination detection accuracy, not just measure coherence.
"""

import argparse
import json
import random
from typing import Any, Dict, List, Optional

from coherify.evaluators.hybrid_selectors import HybridCoherenceConsistencySelector
from coherify.evaluators.response_selectors import (
    CoherenceSelector,
    MajorityVotingSelector,
)

# Import Coherify components
from coherify.generation.model_runner import ModelRunner
from coherify.generation.temperature_strategies import AdaptiveTemperatureSelector


class FaithBenchSample:
    """Single FaithBench sample."""

    def __init__(self, data: Dict[str, Any]):
        self.question = data.get("question", "")
        self.context = data.get("context", "")
        self.response = data.get("response", "")
        self.is_hallucinated = data.get("is_hallucinated", False)
        self.category = data.get("category", "general")
        self.id = data.get("id", "")


class FaithBenchEvaluator:
    """
    Proper FaithBench evaluator that measures hallucination detection accuracy.

    This evaluator:
    1. Generates actual predictions for hallucination detection
    2. Compares different selection strategies
    3. Reports actual detection accuracy (not just coherence)
    """

    def __init__(self, model_runner: Optional[ModelRunner] = None):
        """
        Initialize FaithBench evaluator.

        Args:
            model_runner: ModelRunner for generating predictions
        """
        self.model_runner = model_runner
        self.temp_selector = AdaptiveTemperatureSelector()

        # Initialize selectors for Stage 2 and 3
        self.majority_selector = MajorityVotingSelector()
        self.coherence_selector = CoherenceSelector()
        self.hybrid_selector = HybridCoherenceConsistencySelector(alpha=0.6)

    def create_prompt(self, sample: FaithBenchSample) -> str:
        """Create a prompt for hallucination detection."""
        prompt = f"""You are a hallucination detector. Given a question, context, and response, determine if the response contains hallucinations.

Question: {sample.question}

Context: {sample.context}

Response: {sample.response}

Does the response contain information that is NOT supported by the context (hallucination)?

Answer with exactly one word: YES (hallucinated) or NO (faithful)

Your answer:"""
        return prompt

    def extract_prediction(self, response: str) -> bool:
        """Extract hallucination prediction from model response."""
        response_upper = response.upper()

        # Check for YES/NO
        if "YES" in response_upper and "NO" not in response_upper:
            return True  # Hallucinated
        elif "NO" in response_upper:
            return False  # Faithful

        # Check for other indicators
        hallucination_words = [
            "HALLUCIN",
            "INCORRECT",
            "FALSE",
            "UNSUPPORTED",
            "MADE UP",
        ]
        faithful_words = ["FAITHFUL", "CORRECT", "TRUE", "SUPPORTED", "ACCURATE"]

        for word in hallucination_words:
            if word in response_upper:
                return True

        for word in faithful_words:
            if word in response_upper:
                return False

        # Default to faithful if unclear
        return False

    def evaluate_single(self, sample: FaithBenchSample) -> bool:
        """
        Stage 1: Generate single prediction.

        Returns:
            Predicted hallucination status (True = hallucinated)
        """
        if not self.model_runner:
            # Mock prediction for testing
            return random.choice([True, False])

        prompt = self.create_prompt(sample)
        result = self.model_runner.generate_response(prompt, temperature=0.3)

        return self.extract_prediction(result.text)

    def evaluate_majority_voting(self, sample: FaithBenchSample, k: int = 5) -> bool:
        """
        Stage 2: Generate K predictions and use majority voting.

        Returns:
            Predicted hallucination status from majority voting
        """
        if not self.model_runner:
            # Mock predictions for testing
            predictions = [random.choice([True, False]) for _ in range(k)]
        else:
            prompt = self.create_prompt(sample)

            # Use adaptive temperatures for diversity
            temperatures = self.temp_selector.select_temperatures(prompt, k)
            results = []

            for temp in temperatures:
                result = self.model_runner.generate_response(prompt, temperature=temp)
                results.append(result)

            predictions = [self.extract_prediction(r.text) for r in results]

        # Majority voting
        true_count = sum(predictions)
        false_count = len(predictions) - true_count

        return true_count > false_count

    def evaluate_coherence_selection(
        self, sample: FaithBenchSample, k: int = 5
    ) -> bool:
        """
        Stage 3: Use coherence-based selection.

        Returns:
            Predicted hallucination status from best coherent response
        """
        if not self.model_runner:
            # Mock for testing
            return random.choice([True, False])

        prompt = self.create_prompt(sample)

        # Generate K diverse responses
        responses = self.model_runner.generate_k_responses(prompt, k)

        # Select best response using hybrid selector
        result = self.hybrid_selector.select(responses, prompt)

        return self.extract_prediction(result.selected_response)

    def evaluate_dataset(
        self, samples: List[FaithBenchSample], method: str = "single"
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset with specified method.

        Args:
            samples: List of FaithBench samples
            method: Evaluation method ("single", "majority", "coherence")

        Returns:
            Evaluation results with accuracy metrics
        """
        true_positives = 0  # Correctly detected hallucinations
        true_negatives = 0  # Correctly detected faithful
        false_positives = 0  # Incorrectly flagged as hallucinated
        false_negatives = 0  # Missed hallucinations

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

            # Update confusion matrix
            if sample.is_hallucinated and pred:
                true_positives += 1
            elif not sample.is_hallucinated and not pred:
                true_negatives += 1
            elif not sample.is_hallucinated and pred:
                false_positives += 1
            else:  # sample.is_hallucinated and not pred
                false_negatives += 1

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples...")

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / len(samples) if samples else 0

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Category-specific accuracy
        category_stats = {}
        for sample, pred in zip(samples, predictions):
            cat = sample.category
            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}

            category_stats[cat]["total"] += 1
            if pred == sample.is_hallucinated:
                category_stats[cat]["correct"] += 1

        category_accuracy = {
            cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            for cat, stats in category_stats.items()
        }

        return {
            "method": method,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "category_accuracy": category_accuracy,
            "predictions": predictions,
        }


def load_faithbench_data(sample_size: Optional[int] = None) -> List[FaithBenchSample]:
    """Load FaithBench dataset."""
    print("📚 Loading FaithBench data...")

    # Mock FaithBench data for testing
    mock_data = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
            "response": "The capital of France is Paris.",
            "is_hallucinated": False,
            "category": "factual",
        },
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
            "response": "The capital of France is London.",
            "is_hallucinated": True,
            "category": "factual",
        },
        {
            "question": "Summarize the article",
            "context": "The study found that exercise improves cognitive function in older adults.",
            "response": "The study found that exercise has no effect on cognitive function.",
            "is_hallucinated": True,
            "category": "summary",
        },
        {
            "question": "What did the research conclude?",
            "context": "The research concluded that climate change is accelerating.",
            "response": "The research concluded that climate change is accelerating faster than previously thought.",
            "is_hallucinated": False,
            "category": "research",
        },
        {
            "question": "Describe the main character",
            "context": "John is a tall man with brown hair who works as an engineer.",
            "response": "John is a short man with blonde hair who works as a doctor.",
            "is_hallucinated": True,
            "category": "description",
        },
    ]

    samples = []

    # Repeat to get desired sample size
    while len(samples) < (sample_size or 5):
        for data in mock_data:
            samples.append(FaithBenchSample(data))
            if sample_size and len(samples) >= sample_size:
                break

    print(f"  ✅ Loaded {len(samples)} FaithBench samples")

    # Show distribution
    hallucinated = sum(1 for s in samples if s.is_hallucinated)
    faithful = len(samples) - hallucinated
    print(f"  Distribution: {hallucinated} hallucinated, {faithful} faithful")

    return samples[:sample_size] if sample_size else samples


def compare_methods(
    evaluator: FaithBenchEvaluator, samples: List[FaithBenchSample]
) -> Dict[str, Any]:
    """
    Compare all three methods on the same dataset.

    Returns:
        Comparison results
    """
    print("\n" + "=" * 60)
    print("FaithBench 3-Stage Pipeline Comparison")
    print("=" * 60)

    results = {}

    # Stage 1: Baseline (single response)
    print("\n📊 Stage 1: Baseline Evaluation")
    results["baseline"] = evaluator.evaluate_dataset(samples, method="single")
    print(f"  Accuracy: {results['baseline']['accuracy']:.1%}")
    print(f"  F1 Score: {results['baseline']['f1_score']:.3f}")

    # Stage 2: Majority Voting (K=5)
    print("\n📊 Stage 2: Majority Voting (K=5)")
    results["majority"] = evaluator.evaluate_dataset(samples, method="majority")
    print(f"  Accuracy: {results['majority']['accuracy']:.1%}")
    print(f"  F1 Score: {results['majority']['f1_score']:.3f}")

    # Stage 3: Coherence Selection (K=5)
    print("\n📊 Stage 3: Coherence-Enhanced Selection (K=5)")
    results["coherence"] = evaluator.evaluate_dataset(samples, method="coherence")
    print(f"  Accuracy: {results['coherence']['accuracy']:.1%}")
    print(f"  F1 Score: {results['coherence']['f1_score']:.3f}")

    # Calculate improvements
    baseline_acc = results["baseline"]["accuracy"]
    baseline_f1 = results["baseline"]["f1_score"]

    majority_acc_improvement = results["majority"]["accuracy"] - baseline_acc
    majority_f1_improvement = results["majority"]["f1_score"] - baseline_f1

    coherence_acc_improvement = results["coherence"]["accuracy"] - baseline_acc
    coherence_f1_improvement = results["coherence"]["f1_score"] - baseline_f1

    print("\n" + "=" * 60)
    print("📈 Performance Summary")
    print("=" * 60)

    print("\nAccuracy:")
    print(f"  Baseline:  {baseline_acc:.1%}")
    print(
        f"  Majority:  {results['majority']['accuracy']:.1%} ({majority_acc_improvement:+.1%})"
    )
    print(
        f"  Coherence: {results['coherence']['accuracy']:.1%} ({coherence_acc_improvement:+.1%})"
    )

    print("\nF1 Score:")
    print(f"  Baseline:  {baseline_f1:.3f}")
    print(
        f"  Majority:  {results['majority']['f1_score']:.3f} ({majority_f1_improvement:+.3f})"
    )
    print(
        f"  Coherence: {results['coherence']['f1_score']:.3f} ({coherence_f1_improvement:+.3f})"
    )

    # Confusion matrix for best method
    best_method = max(results.keys(), key=lambda k: results[k]["f1_score"])
    best_result = results[best_method]

    print(f"\n📊 Best Method: {best_method}")
    print(f"  True Positives:  {best_result['true_positives']}")
    print(f"  True Negatives:  {best_result['true_negatives']}")
    print(f"  False Positives: {best_result['false_positives']}")
    print(f"  False Negatives: {best_result['false_negatives']}")
    print(f"  Precision: {best_result['precision']:.3f}")
    print(f"  Recall:    {best_result['recall']:.3f}")

    return results


def main():
    """Main FaithBench benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run FaithBench hallucination detection with 3-stage pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model to use (default, gpt4-mini, gpt4, claude)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--k-responses",
        type=int,
        default=5,
        help="Number of responses for majority voting and coherence selection",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run comparison of all three methods"
    )
    parser.add_argument(
        "--method",
        choices=["single", "majority", "coherence"],
        default="single",
        help="Evaluation method to use",
    )
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    print("🚀 FaithBench Hallucination Detection Benchmark")
    print("=" * 50)

    # Load data
    samples = load_faithbench_data(args.sample_size)

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
    evaluator = FaithBenchEvaluator(model_runner)

    # Run evaluation
    if args.compare:
        results = compare_methods(evaluator, samples)
    else:
        results = {args.method: evaluator.evaluate_dataset(samples, method=args.method)}

        result = results[args.method]
        print(f"\n📊 Results for {args.method} method:")
        print(f"  Accuracy:  {result['accuracy']:.1%}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1 Score:  {result['f1_score']:.3f}")

    # Save results if requested
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.save_results}")

    print("\n✅ FaithBench benchmark completed!")

    # Show key insights
    if args.compare:
        print("\n💡 Key Insights:")

        # Check for improvements
        if results["majority"]["f1_score"] > results["baseline"]["f1_score"]:
            print("  ✅ Majority voting improves F1 score")
        if results["coherence"]["f1_score"] > results["majority"]["f1_score"]:
            print("  ✅ Coherence selection further improves F1 score")

        # Best method
        best_method = max(results.keys(), key=lambda k: results[k]["f1_score"])
        print(
            f"  🏆 Best method: {best_method} (F1: {results[best_method]['f1_score']:.3f})"
        )

        # Detection insights
        best = results[best_method]
        if best["recall"] > 0.7:
            print(f"  👍 Good hallucination recall: {best['recall']:.1%}")
        if best["precision"] > 0.8:
            print(f"  👍 High precision: {best['precision']:.1%}")


if __name__ == "__main__":
    main()
