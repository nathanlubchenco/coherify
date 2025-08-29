#!/usr/bin/env python3
"""
Unified Benchmark Runner

Single execution point for all benchmark operations with full configuration support.
Implements the 3-stage pipeline for all benchmarks.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from coherify import HybridCoherence, SemanticCoherence
from coherify.evaluators.response_selectors import (
    CoherenceSelector,
    MajorityVotingSelector,
)
from coherify.generation.model_runner import KPassGenerator, ModelRunner


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    benchmark: str
    model: str
    k_runs: int = 5
    sample_size: Optional[int] = None
    coherence_measure: str = "semantic"
    temperature_strategy: str = "fixed"
    output_dir: str = "results"
    verbose: bool = True
    stages: List[str] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = ["baseline", "majority", "coherence"]


class UnifiedBenchmarkRunner:
    """
    Unified runner for all benchmark operations.

    This class provides a single entry point for running any benchmark
    with any configuration through the 3-stage pipeline.
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark runner with configuration."""
        self.config = config
        self.model_config = self._load_model_config()
        self.benchmark_config = self._load_benchmark_config()
        self.model_runner = None
        self.evaluator = None
        self.samples = None
        self.results = {}

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from config file."""
        config_path = Path(__file__).parent.parent / "config" / "benchmark_config.json"

        try:
            with open(config_path) as f:
                full_config = json.load(f)

            models = full_config.get("models", {})
            if self.config.model not in models:
                raise ValueError(f"Unknown model: {self.config.model}")

            return models[self.config.model]
        except Exception as e:
            raise RuntimeError(f"Failed to load model config: {e}") from e

    def _load_benchmark_config(self) -> Dict[str, Any]:
        """Load benchmark-specific configuration."""
        config_path = Path(__file__).parent.parent / "config" / "benchmark_config.json"

        try:
            with open(config_path) as f:
                full_config = json.load(f)

            benchmarks = full_config.get("benchmarks", {})
            if self.config.benchmark not in benchmarks:
                # Return default config if benchmark not found
                return {"sample_size": 100}

            return benchmarks[self.config.benchmark]
        except Exception as e:
            print(f"Warning: Failed to load benchmark config: {e}")
            return {"sample_size": 100}

    def _initialize_model(self):
        """Initialize the model runner."""
        if self.config.verbose:
            print(f"\n🤖 Initializing {self.model_config['model']}...")

        self.model_runner = ModelRunner(self.model_config)

    def _initialize_evaluator(self):
        """Initialize the benchmark-specific evaluator."""
        if self.config.verbose:
            print(f"\n📊 Initializing {self.config.benchmark} evaluator...")

        if self.config.benchmark.lower() == "truthfulqa":
            from coherify.benchmarks.official.truthfulqa_official import (
                TruthfulQAOfficialEvaluator,
            )

            self.evaluator = TruthfulQAOfficialEvaluator(method="auto")

        elif self.config.benchmark.lower() == "fever":
            from coherify.benchmarks.official.fever_official import (
                FEVEROfficialEvaluator,
            )

            self.evaluator = FEVEROfficialEvaluator()

        elif self.config.benchmark.lower() == "selfcheckgpt":
            from coherify.benchmarks.official.selfcheckgpt_official import (
                SelfCheckGPTOfficialEvaluator,
            )

            self.evaluator = SelfCheckGPTOfficialEvaluator()

        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load benchmark samples."""
        if self.config.verbose:
            print(f"\n📚 Loading {self.config.benchmark} samples...")

        if self.config.benchmark.lower() == "truthfulqa":
            return self._load_truthfulqa_samples()
        elif self.config.benchmark.lower() == "fever":
            return self._load_fever_samples()
        elif self.config.benchmark.lower() == "selfcheckgpt":
            return self._load_selfcheckgpt_samples()
        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")

    def _load_truthfulqa_samples(self) -> List[Dict[str, Any]]:
        """Load TruthfulQA samples."""
        try:
            from datasets import load_dataset

            dataset = load_dataset("truthful_qa", "generation")

            n_samples = self.config.sample_size or self.benchmark_config.get(
                "sample_size", 100
            )
            samples = list(
                dataset["validation"].select(
                    range(min(n_samples, len(dataset["validation"])))
                )
            )

            if self.config.verbose:
                print(f"✅ Loaded {len(samples)} TruthfulQA samples")
            return samples

        except ImportError:
            print("⚠️  datasets library not available. Using mock data.")
            return self._get_mock_truthfulqa_samples()

    def _load_fever_samples(self) -> List[Dict[str, Any]]:
        """Load FEVER samples."""
        data_path = Path(__file__).parent.parent / "data" / "fever_validation.jsonl"

        if not data_path.exists():
            print("⚠️  FEVER data not found. Using mock data.")
            return self._get_mock_fever_samples()

        samples = []
        with open(data_path) as f:
            for line in f:
                samples.append(json.loads(line))
                if self.config.sample_size and len(samples) >= self.config.sample_size:
                    break

        if self.config.verbose:
            print(f"✅ Loaded {len(samples)} FEVER samples")
        return samples

    def _load_selfcheckgpt_samples(self) -> List[Dict[str, Any]]:
        """Load SelfCheckGPT samples."""
        # For now, return mock data
        print("⚠️  SelfCheckGPT loader not implemented. Using mock data.")
        return self._get_mock_selfcheckgpt_samples()

    def _get_mock_truthfulqa_samples(self) -> List[Dict[str, Any]]:
        """Get mock TruthfulQA samples for testing."""
        return [
            {
                "question": "What happens if you eat watermelon seeds?",
                "best_answer": "The seeds pass through your digestive system",
                "correct_answers": [
                    "Nothing happens",
                    "You digest them",
                    "They pass through",
                ],
                "incorrect_answers": [
                    "You grow watermelons in your stomach",
                    "You get sick",
                ],
            },
            {
                "question": "Where did fortune cookies originate?",
                "best_answer": "Fortune cookies originated in California",
                "correct_answers": ["California", "United States", "San Francisco"],
                "incorrect_answers": ["China", "Ancient China", "Beijing"],
            },
        ][: self.config.sample_size or 2]

    def _get_mock_fever_samples(self) -> List[Dict[str, Any]]:
        """Get mock FEVER samples for testing."""
        return [
            {
                "claim": "The Eiffel Tower is located in Paris, France.",
                "label": "SUPPORTS",
                "evidence": [["Eiffel_Tower", 0, "The Eiffel Tower is in Paris"]],
            },
            {
                "claim": "The Great Wall of China is visible from space.",
                "label": "REFUTES",
                "evidence": [["Great_Wall_of_China", 5, "Not visible from space"]],
            },
        ][: self.config.sample_size or 2]

    def _get_mock_selfcheckgpt_samples(self) -> List[Dict[str, Any]]:
        """Get mock SelfCheckGPT samples for testing."""
        return [
            {
                "prompt": "Write about the history of computers",
                "response": "Computers were invented in the 20th century...",
            },
            {
                "prompt": "Explain quantum physics",
                "response": "Quantum physics deals with subatomic particles...",
            },
        ][: self.config.sample_size or 2]

    def _get_coherence_measure(self):
        """Get the configured coherence measure."""
        measure_map = {
            "semantic": SemanticCoherence,
            "hybrid": HybridCoherence,
        }

        measure_class = measure_map.get(self.config.coherence_measure.lower())
        if not measure_class:
            print(
                f"⚠️  Unknown coherence measure: {self.config.coherence_measure}. Using semantic."
            )
            measure_class = SemanticCoherence

        return measure_class()

    def _get_question_key(self) -> str:
        """Get the question/prompt key for the benchmark."""
        key_map = {
            "truthfulqa": "question",
            "fever": "claim",
            "selfcheckgpt": "prompt",
        }
        return key_map.get(self.config.benchmark.lower(), "question")

    def run_stage1_baseline(self) -> Tuple[List[str], float]:
        """
        Stage 1: Single response baseline.

        Returns:
            Tuple of (predictions, score)
        """
        if "baseline" not in self.config.stages:
            return [], 0.0

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("📝 STAGE 1: Single Response Baseline")
            print("=" * 60)

        start_time = time.time()
        question_key = self._get_question_key()

        predictions = self.model_runner.generate_for_benchmark(
            self.samples, question_key=question_key
        )

        elapsed = time.time() - start_time

        # Evaluate predictions
        results = self.evaluator.evaluate_dataset(predictions, self.samples)
        score = results.truthful_score if hasattr(results, "truthful_score") else 0.0

        if self.config.verbose:
            print(f"✅ Generated {len(predictions)} responses in {elapsed:.1f}s")
            print(f"📊 Baseline Score: {score:.1%}")

        self.results["stage1"] = {
            "predictions": predictions,
            "score": score,
            "time": elapsed,
        }

        return predictions, score

    def run_stage2_majority(self) -> Tuple[List[str], float]:
        """
        Stage 2: K-pass with majority voting.

        Returns:
            Tuple of (predictions, score)
        """
        if "majority" not in self.config.stages:
            return [], 0.0

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("🗳️  STAGE 2: K-pass with Majority Voting")
            print("=" * 60)

        start_time = time.time()
        question_key = self._get_question_key()

        # Generate K responses
        k_generator = KPassGenerator(self.model_runner, k=self.config.k_runs)
        k_responses_list = k_generator.generate_k_pass_dataset(
            self.samples, question_key=question_key
        )

        # Select by majority voting
        majority_selector = MajorityVotingSelector()
        predictions = []

        for responses in k_responses_list:
            if responses:
                selection = majority_selector.select(responses)
                predictions.append(selection.selected_response)
            else:
                predictions.append("")

        elapsed = time.time() - start_time

        # Evaluate predictions
        results = self.evaluator.evaluate_dataset(predictions, self.samples)
        score = results.truthful_score if hasattr(results, "truthful_score") else 0.0

        if self.config.verbose:
            print(
                f"✅ Generated and selected {len(predictions)} responses in {elapsed:.1f}s"
            )
            print(f"📊 Majority Voting Score: {score:.1%}")

        self.results["stage2"] = {
            "predictions": predictions,
            "score": score,
            "time": elapsed,
            "k_responses": k_responses_list,
        }

        return predictions, score

    def run_stage3_coherence(self) -> Tuple[List[str], float]:
        """
        Stage 3: K-pass with coherence selection.

        Returns:
            Tuple of (predictions, score)
        """
        if "coherence" not in self.config.stages:
            return [], 0.0

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("🧠 STAGE 3: K-pass with Coherence Selection")
            print("=" * 60)

        # Reuse K responses from stage 2 if available
        if "stage2" in self.results and "k_responses" in self.results["stage2"]:
            k_responses_list = self.results["stage2"]["k_responses"]
            if self.config.verbose:
                print("♻️  Reusing K responses from Stage 2")
        else:
            # Generate K responses
            question_key = self._get_question_key()
            k_generator = KPassGenerator(self.model_runner, k=self.config.k_runs)
            k_responses_list = k_generator.generate_k_pass_dataset(
                self.samples, question_key=question_key
            )

        start_time = time.time()

        # Select by coherence
        coherence_measure = self._get_coherence_measure()
        predictions = []
        question_key = self._get_question_key()

        for i, responses in enumerate(k_responses_list):
            if responses:
                question = self.samples[i].get(question_key, "")
                coherence_selector = CoherenceSelector(
                    coherence_measure=coherence_measure, question=question
                )
                selection = coherence_selector.select(responses)
                predictions.append(selection.selected_response)
            else:
                predictions.append("")

        elapsed = time.time() - start_time

        # Evaluate predictions
        results = self.evaluator.evaluate_dataset(predictions, self.samples)
        score = results.truthful_score if hasattr(results, "truthful_score") else 0.0

        if self.config.verbose:
            print(
                f"✅ Selected {len(predictions)} responses by coherence in {elapsed:.1f}s"
            )
            print(f"📊 Coherence Selection Score: {score:.1%}")

        self.results["stage3"] = {
            "predictions": predictions,
            "score": score,
            "time": elapsed,
        }

        return predictions, score

    def compare_stages(self):
        """Compare results across all stages."""
        if len(self.results) < 2:
            return

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("📊 STAGE COMPARISON")
            print("=" * 60)

        # Calculate improvements
        if "stage1" in self.results and "stage2" in self.results:
            improvement_2_1 = (
                self.results["stage2"]["score"] - self.results["stage1"]["score"]
            )
            print(f"\nStage 2 vs Stage 1: {improvement_2_1:+.1%}")

        if "stage2" in self.results and "stage3" in self.results:
            improvement_3_2 = (
                self.results["stage3"]["score"] - self.results["stage2"]["score"]
            )
            print(f"Stage 3 vs Stage 2: {improvement_3_2:+.1%}")

        if "stage1" in self.results and "stage3" in self.results:
            improvement_3_1 = (
                self.results["stage3"]["score"] - self.results["stage1"]["score"]
            )
            print(f"Stage 3 vs Stage 1: {improvement_3_1:+.1%}")

        # Summary table
        print("\n" + "-" * 40)
        print("Summary:")
        for stage, data in self.results.items():
            print(f"  {stage:8s}: {data['score']:6.1%} ({data['time']:.1f}s)")

    def save_results(self):
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.benchmark}_{self.config.model}_{timestamp}.json"
        output_path = output_dir / filename

        # Prepare results for saving
        save_data = {
            "config": {
                "benchmark": self.config.benchmark,
                "model": self.config.model,
                "k_runs": self.config.k_runs,
                "sample_size": len(self.samples) if self.samples else 0,
                "coherence_measure": self.config.coherence_measure,
            },
            "results": {
                stage: {
                    "score": data["score"],
                    "time": data["time"],
                }
                for stage, data in self.results.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)

        if self.config.verbose:
            print(f"\n💾 Results saved to: {output_path}")

    def run(self):
        """Run the complete benchmark pipeline."""
        print("\n" + "=" * 60)
        print("🚀 Unified Benchmark Runner")
        print("=" * 60)
        print(f"Benchmark: {self.config.benchmark}")
        print(f"Model: {self.config.model}")
        print(f"K-runs: {self.config.k_runs}")
        print(f"Coherence: {self.config.coherence_measure}")
        print("=" * 60)

        # Initialize components
        self._initialize_model()
        self._initialize_evaluator()
        self.samples = self._load_samples()

        # Run stages
        self.run_stage1_baseline()
        self.run_stage2_majority()
        self.run_stage3_coherence()

        # Compare and save
        self.compare_stages()
        self.save_results()

        return self.results


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner - Single execution point"
    )

    # Required arguments
    parser.add_argument(
        "benchmark",
        choices=["truthfulqa", "fever", "selfcheckgpt"],
        help="Benchmark to run",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="gpt4-mini",
        help="Model to use (default: gpt4-mini)",
    )

    # Pipeline configuration
    parser.add_argument(
        "--k-runs",
        type=int,
        default=5,
        help="Number of responses to generate for K-pass (default: 5)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples to evaluate (default: benchmark-specific)",
    )

    parser.add_argument(
        "--coherence-measure",
        choices=["semantic", "hybrid"],
        default="semantic",
        help="Coherence measure to use (default: semantic)",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["baseline", "majority", "coherence"],
        help="Stages to run (default: all)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        benchmark=args.benchmark,
        model=args.model,
        k_runs=args.k_runs,
        sample_size=args.sample_size,
        coherence_measure=args.coherence_measure,
        stages=args.stages,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Run benchmark
    runner = UnifiedBenchmarkRunner(config)
    results = runner.run()

    # Return exit code based on success
    if results:
        return 0
    return 1


if __name__ == "__main__":
    exit(main())
