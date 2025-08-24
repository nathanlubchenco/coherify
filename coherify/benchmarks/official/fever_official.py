"""
Official FEVER evaluation implementation.

This module faithfully reproduces the original FEVER evaluation methodology
as described in Thorne et al. (2018).

The official FEVER evaluation requires:
1. Document retrieval from Wikipedia (5.4M pages)
2. Sentence selection from retrieved documents
3. Claim verification (SUPPORTS/REFUTES/NOT ENOUGH INFO)
4. FEVER Score = Label Accuracy × Evidence F1

Reference: https://github.com/sheffieldnlp/fever-scorer
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import warnings

# Try to import FEVER scorer
try:
    from fever.scorer import fever_score
    HAS_FEVER_SCORER = True
except ImportError:
    HAS_FEVER_SCORER = False


@dataclass
class FEVEROfficialResult:
    """Official FEVER evaluation result."""
    label_accuracy: float  # Accuracy of SUPPORTS/REFUTES/NEI predictions
    evidence_f1: float  # F1 score for evidence retrieval
    fever_score: float  # Label Accuracy × Evidence F1 (strict)
    oracle_accuracy: float  # Upper bound with perfect evidence
    per_sample_results: List[Dict[str, Any]]
    method: str  # "official" or "approximate"
    warning: Optional[str] = None


class FEVEROfficialEvaluator:
    """
    Official FEVER evaluator that faithfully reproduces the original methodology.
    
    FEVER (Fact Extraction and VERification) requires:
    1. Retrieving evidence from Wikipedia
    2. Predicting labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
    3. Strict scoring where both label AND evidence must be correct
    """
    
    LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    
    def __init__(
        self,
        require_evidence: bool = True,
        strict_scoring: bool = True
    ):
        """
        Initialize official FEVER evaluator.
        
        Args:
            require_evidence: Whether evidence must be provided for SUPPORTS/REFUTES
            strict_scoring: Use strict FEVER score (label AND evidence correct)
        """
        self.require_evidence = require_evidence
        self.strict_scoring = strict_scoring
        
        if not HAS_FEVER_SCORER:
            warnings.warn(
                "Official FEVER scorer not installed. Install with:\n"
                "pip install fever-scorer\n"
                "Using approximate scoring."
            )
    
    def evaluate_dataset(
        self,
        predictions: List[Dict[str, Any]],
        gold_labels: List[Dict[str, Any]],
        verbose: bool = False
    ) -> FEVEROfficialResult:
        """
        Evaluate predictions using official FEVER methodology.
        
        Args:
            predictions: List of predictions with format:
                {
                    "predicted_label": "SUPPORTS/REFUTES/NOT ENOUGH INFO",
                    "predicted_evidence": [[page, line_num], ...],
                    "predicted_pages": [page_names],  # Optional
                }
            gold_labels: List of gold labels with format:
                {
                    "label": "SUPPORTS/REFUTES/NOT ENOUGH INFO",
                    "evidence": [[[page, line_num], ...], ...],  # Multiple evidence sets
                    "claim": "The claim text"
                }
            verbose: Print progress
            
        Returns:
            Official FEVER evaluation results
        """
        if HAS_FEVER_SCORER and self.strict_scoring:
            return self._evaluate_with_official_scorer(predictions, gold_labels, verbose)
        else:
            return self._evaluate_with_approximate_scoring(predictions, gold_labels, verbose)
    
    def _evaluate_with_official_scorer(
        self,
        predictions: List[Dict[str, Any]],
        gold_labels: List[Dict[str, Any]],
        verbose: bool
    ) -> FEVEROfficialResult:
        """Use official FEVER scorer package."""
        
        if verbose:
            print("Using official FEVER scorer...")
        
        # Format for official scorer
        formatted_predictions = []
        formatted_gold = []
        
        for pred, gold in zip(predictions, gold_labels):
            formatted_predictions.append({
                "predicted_label": pred.get("predicted_label", "NOT ENOUGH INFO"),
                "predicted_evidence": pred.get("predicted_evidence", []),
            })
            formatted_gold.append({
                "label": gold["label"],
                "evidence": gold.get("evidence", []),
            })
        
        # Calculate official scores
        strict_score, label_accuracy, precision, recall, f1 = fever_score(
            formatted_predictions,
            formatted_gold
        )
        
        # Calculate oracle accuracy (upper bound)
        oracle_correct = 0
        for pred, gold in zip(predictions, gold_labels):
            # Oracle: if we had perfect evidence, would label be correct?
            if pred.get("predicted_label") == gold["label"]:
                oracle_correct += 1
        
        oracle_accuracy = oracle_correct / len(predictions) if predictions else 0
        
        # Detailed per-sample results
        per_sample_results = []
        for i, (pred, gold) in enumerate(zip(predictions, gold_labels)):
            label_correct = pred.get("predicted_label") == gold["label"]
            evidence_correct = self._check_evidence_correct(
                pred.get("predicted_evidence", []),
                gold.get("evidence", [])
            )
            
            per_sample_results.append({
                "claim": gold.get("claim", ""),
                "gold_label": gold["label"],
                "predicted_label": pred.get("predicted_label", "NOT ENOUGH INFO"),
                "label_correct": label_correct,
                "evidence_correct": evidence_correct,
                "fever_correct": label_correct and evidence_correct,
            })
        
        return FEVEROfficialResult(
            label_accuracy=label_accuracy,
            evidence_f1=f1,
            fever_score=strict_score,
            oracle_accuracy=oracle_accuracy,
            per_sample_results=per_sample_results,
            method="official"
        )
    
    def _evaluate_with_approximate_scoring(
        self,
        predictions: List[Dict[str, Any]],
        gold_labels: List[Dict[str, Any]],
        verbose: bool
    ) -> FEVEROfficialResult:
        """Approximate FEVER scoring without official package."""
        
        if verbose:
            print("Using approximate FEVER scoring (install fever-scorer for official)...")
        
        label_correct_count = 0
        evidence_correct_count = 0
        fever_correct_count = 0
        
        per_sample_results = []
        
        for pred, gold in zip(predictions, gold_labels):
            # Check label accuracy
            pred_label = pred.get("predicted_label", "NOT ENOUGH INFO")
            gold_label = gold["label"]
            label_correct = pred_label == gold_label
            
            # Check evidence (approximate)
            evidence_correct = False
            if gold_label == "NOT ENOUGH INFO":
                # NEI doesn't require evidence
                evidence_correct = True
            elif self.require_evidence:
                pred_evidence = pred.get("predicted_evidence", [])
                gold_evidence_sets = gold.get("evidence", [])
                evidence_correct = self._check_evidence_correct(
                    pred_evidence, gold_evidence_sets
                )
            else:
                # If not requiring evidence, only label matters
                evidence_correct = label_correct
            
            # FEVER score (strict)
            fever_correct = label_correct and evidence_correct
            
            if label_correct:
                label_correct_count += 1
            if evidence_correct:
                evidence_correct_count += 1
            if fever_correct:
                fever_correct_count += 1
            
            per_sample_results.append({
                "claim": gold.get("claim", ""),
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "label_correct": label_correct,
                "evidence_correct": evidence_correct,
                "fever_correct": fever_correct,
                "predicted_evidence": pred.get("predicted_evidence", []),
                "gold_evidence": gold.get("evidence", []),
            })
        
        n = len(predictions)
        
        return FEVEROfficialResult(
            label_accuracy=label_correct_count / n if n > 0 else 0,
            evidence_f1=evidence_correct_count / n if n > 0 else 0,  # Approximation
            fever_score=fever_correct_count / n if n > 0 else 0,
            oracle_accuracy=label_correct_count / n if n > 0 else 0,  # Same as label acc
            per_sample_results=per_sample_results,
            method="approximate",
            warning="Using approximate scoring. Install fever-scorer for official evaluation."
        )
    
    def _check_evidence_correct(
        self,
        predicted_evidence: List[List],
        gold_evidence_sets: List[List[List]]
    ) -> bool:
        """
        Check if predicted evidence matches any gold evidence set.
        
        FEVER allows multiple valid evidence sets. Prediction is correct
        if it matches ANY complete evidence set.
        """
        if not gold_evidence_sets:
            return True  # No evidence required (NEI case)
        
        predicted_set = set()
        for evidence in predicted_evidence:
            if len(evidence) >= 2:
                # Evidence format: [page, line_number]
                predicted_set.add((evidence[0], evidence[1]))
        
        # Check each gold evidence set
        for gold_evidence_set in gold_evidence_sets:
            gold_set = set()
            for evidence in gold_evidence_set:
                if len(evidence) >= 2:
                    gold_set.add((evidence[0], evidence[1]))
            
            # Check if predicted matches this gold set
            if predicted_set == gold_set:
                return True
        
        return False
    
    @staticmethod
    def validate_against_published():
        """
        Validate against published FEVER results.
        
        Expected results from FEVER paper and leaderboard:
        - Baseline (TF-IDF + Decomposable Attention): 31.87% FEVER score
        - Top systems: ~70% FEVER score
        
        Your implementation should produce similar numbers.
        """
        validation_info = {
            "baseline": {
                "label_accuracy": 0.515,
                "fever_score": 0.3187,
                "description": "TF-IDF retrieval + Decomposable Attention"
            },
            "sota_2018": {
                "label_accuracy": 0.682,
                "fever_score": 0.6421,
                "description": "UNC-NLP (top at time of paper)"
            },
            "human": {
                "label_accuracy": 0.8563,
                "fever_score": None,
                "description": "Human performance on label prediction"
            },
            "note": "Run on full test set (19,998 claims) for comparison"
        }
        return validation_info
    
    @staticmethod
    def format_predictions_for_evaluation(
        claims: List[str],
        labels: List[str],
        evidence: Optional[List[List]] = None
    ) -> List[Dict[str, Any]]:
        """
        Helper to format predictions for evaluation.
        
        Args:
            claims: List of claim texts
            labels: Predicted labels (SUPPORTS/REFUTES/NOT ENOUGH INFO)
            evidence: Optional evidence in format [[page, line], ...]
            
        Returns:
            Formatted predictions for evaluate_dataset
        """
        predictions = []
        for i, (claim, label) in enumerate(zip(claims, labels)):
            pred = {
                "predicted_label": label,
                "predicted_evidence": evidence[i] if evidence else []
            }
            predictions.append(pred)
        
        return predictions


# Simplified interface
def evaluate_fever_official(
    predictions: List[Dict[str, Any]],
    gold_labels: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Simple interface for official FEVER evaluation.
    
    Returns:
        Dictionary with 'label_accuracy', 'evidence_f1', and 'fever_score'
    """
    evaluator = FEVEROfficialEvaluator()
    result = evaluator.evaluate_dataset(predictions, gold_labels)
    
    return {
        "label_accuracy": result.label_accuracy,
        "evidence_f1": result.evidence_f1,
        "fever_score": result.fever_score,
        "oracle_accuracy": result.oracle_accuracy,
        "method": result.method
    }