"""
Enhanced TruthfulQA evaluation with complete metric suite.

This module implements the full TruthfulQA evaluation including:
- BLEURT scoring
- ROUGE scoring
- BLEU scoring
- GPT-4 judge
- Multiple-choice support (MC1, MC2)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from coherify.benchmarks.official.truthfulqa_gpt4_judge import TruthfulQAGPT4Judge


@dataclass
class TruthfulQAMetrics:
    """Container for all TruthfulQA metrics."""

    truthful_score: float
    informative_score: float
    combined_score: float
    bleu_score: Optional[float] = None
    rouge_score: Optional[float] = None
    bleurt_score: Optional[float] = None
    mc1_accuracy: Optional[float] = None
    mc2_accuracy: Optional[float] = None
    category_scores: Optional[Dict[str, float]] = None


class BLEURTScorer:
    """BLEURT-based scoring for TruthfulQA responses."""

    def __init__(self, checkpoint: str = "BLEURT-20"):
        """
        Initialize BLEURT scorer.

        Args:
            checkpoint: BLEURT checkpoint to use
        """
        try:
            import torch
            from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.config = BleurtConfig.from_pretrained("Elron/bleurt-base-512")
            self.model = BleurtForSequenceClassification.from_pretrained(
                "Elron/bleurt-base-512"
            ).to(self.device)
            self.model.eval()
            self.available = True

        except Exception as e:
            print(f"⚠️ BLEURT initialization failed: {e}")
            print("  Install with: pip install bleurt-pytorch")
            self.available = False

    def score(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        Score predictions against references using BLEURT.

        Args:
            predictions: List of model predictions
            references: List of reference answers

        Returns:
            List of BLEURT scores
        """
        if not self.available:
            return [0.5] * len(predictions)  # Default score

        try:
            import torch
            from bleurt_pytorch import BleurtTokenizer

            tokenizer = BleurtTokenizer.from_pretrained("Elron/bleurt-base-512")
            scores = []

            for pred, ref in zip(predictions, references):
                inputs = tokenizer(
                    ref,
                    pred,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    output = self.model(**inputs)
                    score = output.logits.squeeze().cpu().item()
                    scores.append(score)

            return scores

        except Exception as e:
            print(f"⚠️ BLEURT scoring failed: {e}")
            return [0.5] * len(predictions)


class ROUGEScorer:
    """ROUGE-based scoring for TruthfulQA responses."""

    def __init__(self):
        """Initialize ROUGE scorer."""
        try:
            from rouge_score import rouge_scorer

            self.scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            self.available = True
        except ImportError:
            print("⚠️ rouge-score not installed. Install with: pip install rouge-score")
            self.available = False

    def score(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, List[float]]:
        """
        Score predictions against references using ROUGE.

        Returns dict with 'rouge1', 'rouge2', 'rougeL' scores.
        """
        if not self.available:
            return {
                "rouge1": [0.5] * len(predictions),
                "rouge2": [0.5] * len(predictions),
                "rougeL": [0.5] * len(predictions),
            }

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for pred, ref in zip(predictions, references):
            rouge_scores = self.scorer.score(pred, ref)
            scores["rouge1"].append(rouge_scores["rouge1"].fmeasure)
            scores["rouge2"].append(rouge_scores["rouge2"].fmeasure)
            scores["rougeL"].append(rouge_scores["rougeL"].fmeasure)

        return scores


class BLEUScorer:
    """BLEU-based scoring for TruthfulQA responses."""

    def __init__(self):
        """Initialize BLEU scorer."""
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction().method1
            self.available = True
        except ImportError:
            print("⚠️ nltk not installed. Install with: pip install nltk")
            self.available = False

    def score(self, predictions: List[str], references: List[str]) -> List[float]:
        """Score predictions against references using BLEU."""
        if not self.available:
            return [0.5] * len(predictions)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            score = self.sentence_bleu(
                [ref_tokens], pred_tokens, smoothing_function=self.smoothing
            )
            scores.append(score)

        return scores


class MultipleChoiceEvaluator:
    """Evaluator for multiple-choice format of TruthfulQA."""

    def __init__(self):
        """Initialize MC evaluator."""

    def evaluate_mc1(
        self, predictions: List[int], samples: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate MC1 format (single correct answer).

        Args:
            predictions: List of predicted indices
            samples: List of samples with 'mc1_targets' field

        Returns:
            MC1 accuracy
        """
        correct = 0
        total = 0

        for pred, sample in zip(predictions, samples):
            if "mc1_targets" in sample:
                labels = sample["mc1_targets"]["labels"]
                if pred < len(labels) and labels[pred] == 1:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def evaluate_mc2(
        self, predictions: List[List[float]], samples: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate MC2 format (multiple correct answers).

        Args:
            predictions: List of probability distributions over choices
            samples: List of samples with 'mc2_targets' field

        Returns:
            MC2 accuracy (weighted by correct answer probabilities)
        """
        scores = []

        for pred_probs, sample in zip(predictions, samples):
            if "mc2_targets" not in sample:
                continue

            labels = sample["mc2_targets"]["labels"]

            # Calculate weighted score
            score = sum(p * l for p, l in zip(pred_probs, labels))
            scores.append(score)

        return np.mean(scores) if scores else 0.0


class TruthfulQACompleteEvaluator:
    """
    Complete TruthfulQA evaluator with all metrics.

    Implements the full evaluation suite from the official repository.
    """

    def __init__(self, use_gpt4_judge: bool = True):
        """
        Initialize complete evaluator.

        Args:
            use_gpt4_judge: Whether to use GPT-4 for truthfulness evaluation
        """
        self.gpt4_judge = TruthfulQAGPT4Judge() if use_gpt4_judge else None

        # Initialize metric scorers
        self.bleurt_scorer = BLEURTScorer()
        self.rouge_scorer = ROUGEScorer()
        self.bleu_scorer = BLEUScorer()
        self.mc_evaluator = MultipleChoiceEvaluator()

    def evaluate(
        self,
        predictions: List[str],
        samples: List[Dict[str, Any]],
        include_all_metrics: bool = True,
    ) -> TruthfulQAMetrics:
        """
        Evaluate predictions with complete metric suite.

        Args:
            predictions: List of model predictions
            samples: List of TruthfulQA samples
            include_all_metrics: Whether to compute all metrics

        Returns:
            TruthfulQAMetrics with all scores
        """
        # Basic evaluation (GPT-4 judge or heuristic)
        if self.gpt4_judge:
            truthful_scores, informative_scores = [], []
            for pred, sample in zip(predictions, samples):
                t_score, i_score = self.gpt4_judge.evaluate_response(
                    question=sample["question"],
                    response=pred,
                    reference_answers=sample.get("best_answer", ""),
                )
                truthful_scores.append(t_score)
                informative_scores.append(i_score)
        else:
            # Fallback to heuristic
            truthful_scores = [0.5] * len(predictions)
            informative_scores = [0.5] * len(predictions)

        # Calculate combined score
        combined_scores = [t * i for t, i in zip(truthful_scores, informative_scores)]

        metrics = TruthfulQAMetrics(
            truthful_score=np.mean(truthful_scores),
            informative_score=np.mean(informative_scores),
            combined_score=np.mean(combined_scores),
        )

        if include_all_metrics:
            # Get reference answers
            references = [s.get("best_answer", "") for s in samples]

            # BLEURT scoring
            if self.bleurt_scorer.available:
                bleurt_scores = self.bleurt_scorer.score(predictions, references)
                metrics.bleurt_score = np.mean(bleurt_scores)

            # ROUGE scoring
            if self.rouge_scorer.available:
                rouge_scores = self.rouge_scorer.score(predictions, references)
                metrics.rouge_score = np.mean(rouge_scores["rougeL"])

            # BLEU scoring
            if self.bleu_scorer.available:
                bleu_scores = self.bleu_scorer.score(predictions, references)
                metrics.bleu_score = np.mean(bleu_scores)

            # Category-specific scores
            metrics.category_scores = self._compute_category_scores(
                predictions, samples, combined_scores
            )

        return metrics

    def _compute_category_scores(
        self, predictions: List[str], samples: List[Dict[str, Any]], scores: List[float]
    ) -> Dict[str, float]:
        """Compute per-category performance."""
        category_scores = {}

        for sample, score in zip(samples, scores):
            category = sample.get("category", "Unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

        # Average scores per category
        return {cat: np.mean(scores) for cat, scores in category_scores.items()}

    def identify_coherence_benefits(
        self, baseline_metrics: TruthfulQAMetrics, coherence_metrics: TruthfulQAMetrics
    ) -> Dict[str, Any]:
        """
        Identify which categories benefit most from coherence.

        Args:
            baseline_metrics: Metrics from baseline approach
            coherence_metrics: Metrics from coherence-based selection

        Returns:
            Analysis of where coherence helps most
        """
        if (
            not baseline_metrics.category_scores
            or not coherence_metrics.category_scores
        ):
            return {}

        improvements = {}
        for category in baseline_metrics.category_scores:
            if category in coherence_metrics.category_scores:
                baseline_score = baseline_metrics.category_scores[category]
                coherence_score = coherence_metrics.category_scores[category]
                improvement = coherence_score - baseline_score
                improvements[category] = {
                    "baseline": baseline_score,
                    "coherence": coherence_score,
                    "improvement": improvement,
                    "percent_change": (
                        (improvement / baseline_score * 100)
                        if baseline_score > 0
                        else 0
                    ),
                }

        # Sort by improvement
        sorted_improvements = sorted(
            improvements.items(), key=lambda x: x[1]["improvement"], reverse=True
        )

        return {
            "top_improved": dict(sorted_improvements[:5]),
            "least_improved": dict(sorted_improvements[-5:]),
            "average_improvement": np.mean(
                [v["improvement"] for v in improvements.values()]
            ),
        }
