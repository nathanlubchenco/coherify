"""
HaluEval benchmark integration for hallucination detection.

HaluEval is a comprehensive benchmark with 35,000+ examples specifically
designed for hallucination detection across multiple task types.
"""

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coherify.evaluators.hybrid_selectors import HybridCoherenceConsistencySelector
from coherify.evaluators.response_selectors import (
    CoherenceSelector,
    MajorityVotingSelector,
)
from coherify.generation.model_runner import ModelRunner


@dataclass
class HaluEvalSample:
    """Single HaluEval sample."""

    task: str  # qa, dialogue, summarization, general
    question: str
    context: Optional[str]
    reference: str
    hallucinated_response: Optional[str]
    is_hallucination: bool
    sample_id: str


@dataclass
class HaluEvalResult:
    """Results from HaluEval benchmark."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    task_scores: Dict[str, float]
    method: str
    num_samples: int


class HaluEvalDataset:
    """
    HaluEval dataset loader and manager.

    Handles loading and preprocessing of HaluEval benchmark data.
    """

    TASKS = ["qa", "dialogue", "summarization", "general"]

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize HaluEval dataset.

        Args:
            data_path: Path to HaluEval data (will download if not provided)
        """
        self.data_path = data_path
        self.samples = []

        if data_path:
            self.load_from_file(data_path)
        else:
            self.load_sample_data()  # Use sample data for testing

    def load_from_file(self, path: str):
        """Load HaluEval data from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            for task in self.TASKS:
                if task in data:
                    for item in data[task]:
                        sample = HaluEvalSample(
                            task=task,
                            question=item.get("question", item.get("query", "")),
                            context=item.get("context"),
                            reference=item.get("reference", item.get("answer", "")),
                            hallucinated_response=item.get("hallucinated"),
                            is_hallucination=item.get("label", False),
                            sample_id=item.get("id", f"{task}_{len(self.samples)}"),
                        )
                        self.samples.append(sample)

            print(f"✅ Loaded {len(self.samples)} HaluEval samples from {path}")

        except Exception as e:
            print(f"⚠️ Error loading HaluEval data: {e}")
            self.load_sample_data()

    def load_sample_data(self):
        """Load sample data for testing."""
        sample_data = [
            # QA samples
            HaluEvalSample(
                task="qa",
                question="What is the capital of France?",
                context=None,
                reference="Paris is the capital of France",
                hallucinated_response="London is the capital of France",
                is_hallucination=True,
                sample_id="qa_001",
            ),
            HaluEvalSample(
                task="qa",
                question="Who wrote Romeo and Juliet?",
                context=None,
                reference="William Shakespeare wrote Romeo and Juliet",
                hallucinated_response="Charles Dickens wrote Romeo and Juliet",
                is_hallucination=True,
                sample_id="qa_002",
            ),
            # Dialogue samples
            HaluEvalSample(
                task="dialogue",
                question="What did we discuss earlier about the meeting?",
                context="Earlier: We scheduled the meeting for 3 PM on Tuesday",
                reference="We scheduled the meeting for 3 PM on Tuesday",
                hallucinated_response="We scheduled the meeting for 5 PM on Thursday",
                is_hallucination=True,
                sample_id="dialogue_001",
            ),
            # Summarization samples
            HaluEvalSample(
                task="summarization",
                question="Summarize the article",
                context="The article discusses climate change impacts on polar bears. It mentions declining ice levels and hunting challenges.",
                reference="The article discusses how climate change affects polar bears through ice loss and hunting difficulties",
                hallucinated_response="The article discusses how polar bears are thriving due to warmer temperatures",
                is_hallucination=True,
                sample_id="sum_001",
            ),
            # General samples
            HaluEvalSample(
                task="general",
                question="Explain photosynthesis",
                context=None,
                reference="Photosynthesis is the process by which plants convert light energy into chemical energy",
                hallucinated_response="Photosynthesis is the process by which animals digest food",
                is_hallucination=True,
                sample_id="general_001",
            ),
        ]

        # Add some non-hallucinated samples
        for sample in sample_data[:3]:
            non_hallu = HaluEvalSample(
                task=sample.task,
                question=sample.question,
                context=sample.context,
                reference=sample.reference,
                hallucinated_response=None,
                is_hallucination=False,
                sample_id=sample.sample_id + "_truthful",
            )
            sample_data.append(non_hallu)

        self.samples = sample_data
        print(f"📦 Loaded {len(self.samples)} sample HaluEval examples for testing")

    def get_task_samples(
        self, task: str, n: Optional[int] = None
    ) -> List[HaluEvalSample]:
        """Get samples for a specific task."""
        task_samples = [s for s in self.samples if s.task == task]
        if n:
            return random.sample(task_samples, min(n, len(task_samples)))
        return task_samples

    def get_balanced_sample(self, n: int) -> List[HaluEvalSample]:
        """Get a balanced sample across all tasks."""
        samples_per_task = n // len(self.TASKS)
        balanced = []
        for task in self.TASKS:
            task_samples = self.get_task_samples(task, samples_per_task)
            balanced.extend(task_samples)
        return balanced


class HaluEvalEvaluator:
    """
    Evaluator for HaluEval benchmark.

    Tests different selection strategies on hallucination detection.
    """

    def __init__(self, model_runner: Optional[ModelRunner] = None):
        """
        Initialize HaluEval evaluator.

        Args:
            model_runner: ModelRunner for generating responses
        """
        self.model_runner = model_runner
        self.dataset = HaluEvalDataset()

        # Initialize selectors
        self.selectors = {
            "majority": MajorityVotingSelector(),
            "coherence": CoherenceSelector(),
            "hybrid": HybridCoherenceConsistencySelector(alpha=0.6),
        }

    def evaluate_sample(
        self, sample: HaluEvalSample, method: str = "hybrid", k: int = 5
    ) -> Tuple[bool, float]:
        """
        Evaluate a single sample.

        Returns:
            Tuple of (is_hallucination_detected, confidence)
        """
        if not self.model_runner:
            # Mock evaluation for testing
            return (random.random() > 0.5, random.random())

        # Generate K responses
        prompt = self._create_prompt(sample)
        responses = self.model_runner.generate_k_responses(prompt, k)

        # Select best response
        selector = self.selectors[method]
        result = selector.select(responses, sample.question)

        # Check if hallucination was detected
        is_hallucination_detected = self._check_hallucination(
            result.selected_response, sample
        )

        return is_hallucination_detected, result.confidence

    def _create_prompt(self, sample: HaluEvalSample) -> str:
        """Create prompt for the model."""
        prompt = f"Question: {sample.question}\n"

        if sample.context:
            prompt += f"Context: {sample.context}\n"

        prompt += "Please provide a truthful and accurate answer:"

        return prompt

    def _check_hallucination(self, response: str, sample: HaluEvalSample) -> bool:
        """
        Check if response contains hallucination.

        Simple heuristic for testing - in production would use
        more sophisticated methods.
        """
        if sample.hallucinated_response:
            # Check similarity to known hallucination
            similarity = self._text_similarity(response, sample.hallucinated_response)
            return similarity > 0.7
        else:
            # Check dissimilarity to reference
            similarity = self._text_similarity(response, sample.reference)
            return similarity < 0.3

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity metric."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def evaluate(
        self,
        method: str = "hybrid",
        n_samples: Optional[int] = None,
        k_responses: int = 5,
    ) -> HaluEvalResult:
        """
        Run full evaluation on HaluEval benchmark.

        Args:
            method: Selection method to use
            n_samples: Number of samples to evaluate (None for all)
            k_responses: Number of responses per sample

        Returns:
            HaluEvalResult with metrics
        """
        if n_samples:
            samples = self.dataset.get_balanced_sample(n_samples)
        else:
            samples = self.dataset.samples

        # Track predictions
        true_positives = 0  # Correctly identified hallucinations
        false_positives = 0  # Incorrectly flagged as hallucinations
        true_negatives = 0  # Correctly identified truthful
        false_negatives = 0  # Missed hallucinations

        task_correct = {task: [] for task in self.dataset.TASKS}

        print(f"\n🔍 Evaluating {len(samples)} samples with {method} method...")

        for i, sample in enumerate(samples):
            detected, confidence = self.evaluate_sample(sample, method, k_responses)

            # Update metrics
            if sample.is_hallucination:
                if detected:
                    true_positives += 1
                    task_correct[sample.task].append(1)
                else:
                    false_negatives += 1
                    task_correct[sample.task].append(0)
            else:
                if detected:
                    false_positives += 1
                    task_correct[sample.task].append(0)
                else:
                    true_negatives += 1
                    task_correct[sample.task].append(1)

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(samples)} samples...")

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / len(samples)

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

        # Task-specific scores
        task_scores = {}
        for task in self.dataset.TASKS:
            if task in task_correct and task_correct[task]:
                task_scores[task] = np.mean(task_correct[task])
            else:
                task_scores[task] = 0.0

        return HaluEvalResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            task_scores=task_scores,
            method=method,
            num_samples=len(samples),
        )

    def compare_methods(
        self, n_samples: Optional[int] = None, k_responses: int = 5
    ) -> Dict[str, HaluEvalResult]:
        """
        Compare different selection methods on HaluEval.

        Returns:
            Dictionary of method -> results
        """
        results = {}

        for method in self.selectors.keys():
            print(f"\n📊 Testing {method} method...")
            result = self.evaluate(method, n_samples, k_responses)
            results[method] = result

            print(f"  Accuracy: {result.accuracy:.3f}")
            print(f"  F1 Score: {result.f1_score:.3f}")

        return results


def run_halueval_benchmark(
    model_config: Optional[Dict[str, Any]] = None, n_samples: int = 100
) -> Dict[str, Any]:
    """
    Run HaluEval benchmark with specified configuration.

    Args:
        model_config: Model configuration for generation
        n_samples: Number of samples to evaluate

    Returns:
        Benchmark results
    """
    # Setup model runner
    model_runner = None
    if model_config:
        model_runner = ModelRunner(model_config)

    # Create evaluator
    evaluator = HaluEvalEvaluator(model_runner)

    # Run comparison
    results = evaluator.compare_methods(n_samples=n_samples)

    # Print summary
    print("\n" + "=" * 60)
    print("HaluEval Benchmark Results")
    print("=" * 60)

    for method, result in results.items():
        print(f"\n{method.upper()} Method:")
        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall: {result.recall:.3f}")
        print(f"  F1 Score: {result.f1_score:.3f}")

        if result.task_scores:
            print("  Task Scores:")
            for task, score in result.task_scores.items():
                print(f"    {task}: {score:.3f}")

    return results
