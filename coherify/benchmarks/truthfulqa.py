"""
TruthfulQA-specific benchmark integration.
Provides specialized adapters and evaluation metrics for TruthfulQA dataset.
"""

from typing import Dict, List, Any, Optional, Union
import json

from coherify.core.base import PropositionSet, Proposition, CoherenceResult
from coherify.benchmarks.adapters import BenchmarkAdapter
from coherify.core.base import CoherenceMeasure


class TruthfulQAAdapter(BenchmarkAdapter):
    """
    Specialized adapter for TruthfulQA benchmark.
    
    TruthfulQA has specific structure with questions, best answers,
    correct answers, and incorrect answers that can be leveraged
    for coherence evaluation.
    """
    
    def __init__(self, 
                 evaluation_mode: str = "generation",
                 include_context: bool = True,
                 use_correct_answers: bool = False):
        """
        Initialize TruthfulQA adapter.
        
        Args:
            evaluation_mode: "generation" or "mc" (multiple choice)
            include_context: Whether to include question as context
            use_correct_answers: Whether to use provided correct answers vs generated
        """
        super().__init__("truthfulqa")
        self.evaluation_mode = evaluation_mode
        self.include_context = include_context
        self.use_correct_answers = use_correct_answers
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert TruthfulQA sample to PropositionSet."""
        question = sample.get("question", "")
        
        if self.evaluation_mode == "generation":
            return self._adapt_generation_sample(sample, question)
        elif self.evaluation_mode == "mc":
            return self._adapt_mc_sample(sample, question)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")
    
    def _adapt_generation_sample(self, sample: Dict[str, Any], question: str) -> PropositionSet:
        """Adapt generation-mode sample."""
        if self.use_correct_answers and "correct_answers" in sample:
            # Use provided correct answers
            answers = sample["correct_answers"]
            props = [Proposition(text=ans) for ans in answers]
        elif "best_answer" in sample:
            # Use best answer, segmented
            answer = sample["best_answer"]
            segments = [s.strip() for s in answer.split('.') if s.strip()]
            props = [Proposition(text=seg) for seg in segments]
        elif "answer" in sample:
            # Use generated answer
            answer = sample["answer"]
            segments = [s.strip() for s in answer.split('.') if s.strip()]
            props = [Proposition(text=seg) for seg in segments]
        else:
            # Fallback to question as single proposition
            props = [Proposition(text=question)]
        
        context = question if self.include_context else None
        
        # Add TruthfulQA-specific metadata
        metadata = {
            "benchmark": "truthfulqa",
            "evaluation_mode": "generation",
            "category": sample.get("category", "unknown"),
            "has_correct_answers": "correct_answers" in sample,
            "has_incorrect_answers": "incorrect_answers" in sample
        }
        
        return PropositionSet(
            propositions=props,
            context=context,
            metadata=metadata
        )
    
    def _adapt_mc_sample(self, sample: Dict[str, Any], question: str) -> PropositionSet:
        """Adapt multiple-choice sample."""
        choices = sample.get("mc1_targets", {}) or sample.get("mc2_targets", {})
        
        if isinstance(choices, dict):
            # Extract choices from labels/choices format
            labels = choices.get("labels", [])
            choice_texts = choices.get("choices", [])
            
            # Create propositions from choices
            props = []
            for i, choice in enumerate(choice_texts):
                is_correct = i < len(labels) and labels[i] == 1
                prop_metadata = {"is_correct": is_correct, "choice_index": i}
                props.append(Proposition(text=choice, metadata=prop_metadata))
        
        elif isinstance(choices, list):
            # Simple list of choices
            props = [Proposition(text=choice) for choice in choices]
        
        else:
            # Fallback
            props = [Proposition(text=str(choices))]
        
        context = question if self.include_context else None
        
        metadata = {
            "benchmark": "truthfulqa",
            "evaluation_mode": "mc",
            "category": sample.get("category", "unknown"),
            "num_choices": len(props)
        }
        
        return PropositionSet(
            propositions=props,
            context=context,
            metadata=metadata
        )
    
    def create_positive_negative_sets(self, sample: Dict[str, Any]) -> Dict[str, PropositionSet]:
        """
        Create positive (correct) and negative (incorrect) proposition sets.
        Useful for contrastive coherence evaluation.
        """
        question = sample.get("question", "")
        result = {}
        
        # Positive set (correct answers)
        if "correct_answers" in sample:
            correct_props = [Proposition(text=ans, metadata={"valence": "positive"}) 
                           for ans in sample["correct_answers"]]
            result["positive"] = PropositionSet(
                propositions=correct_props,
                context=question,
                metadata={"valence": "positive", "benchmark": "truthfulqa"}
            )
        
        # Negative set (incorrect answers)
        if "incorrect_answers" in sample:
            incorrect_props = [Proposition(text=ans, metadata={"valence": "negative"}) 
                             for ans in sample["incorrect_answers"]]
            result["negative"] = PropositionSet(
                propositions=incorrect_props,
                context=question,
                metadata={"valence": "negative", "benchmark": "truthfulqa"}
            )
        
        return result


class TruthfulQAEvaluator:
    """
    Specialized evaluator for TruthfulQA with coherence measures.
    Provides TruthfulQA-specific evaluation metrics and analysis.
    """
    
    def __init__(self, coherence_measure: CoherenceMeasure):
        self.coherence_measure = coherence_measure
        self.adapter = TruthfulQAAdapter()
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single TruthfulQA sample."""
        # Get basic coherence
        prop_set = self.adapter.adapt_single(sample)
        coherence_result = self.coherence_measure.compute(prop_set)
        
        evaluation = {
            "coherence_score": coherence_result.score,
            "coherence_details": coherence_result.details,
            "num_propositions": len(prop_set),
            "category": sample.get("category", "unknown")
        }
        
        # Contrastive evaluation if possible
        if "correct_answers" in sample and "incorrect_answers" in sample:
            pos_neg_sets = self.adapter.create_positive_negative_sets(sample)
            
            if "positive" in pos_neg_sets:
                pos_result = self.coherence_measure.compute(pos_neg_sets["positive"])
                evaluation["positive_coherence"] = pos_result.score
            
            if "negative" in pos_neg_sets:
                neg_result = self.coherence_measure.compute(pos_neg_sets["negative"])
                evaluation["negative_coherence"] = neg_result.score
            
            if "positive_coherence" in evaluation and "negative_coherence" in evaluation:
                evaluation["coherence_contrast"] = (
                    evaluation["positive_coherence"] - evaluation["negative_coherence"]
                )
        
        return evaluation
    
    def evaluate_dataset(self, dataset) -> Dict[str, Any]:
        """Evaluate entire TruthfulQA dataset."""
        results = []
        category_scores = {}
        
        for sample in dataset:
            eval_result = self.evaluate_sample(sample)
            results.append(eval_result)
            
            # Aggregate by category
            category = eval_result["category"]
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(eval_result["coherence_score"])
        
        # Calculate summary statistics
        all_scores = [r["coherence_score"] for r in results]
        
        summary = {
            "num_samples": len(results),
            "mean_coherence": sum(all_scores) / len(all_scores) if all_scores else 0,
            "category_means": {
                cat: sum(scores) / len(scores) 
                for cat, scores in category_scores.items()
            },
            "detailed_results": results
        }
        
        # Add contrastive analysis if available
        contrast_scores = [r.get("coherence_contrast") for r in results if "coherence_contrast" in r]
        if contrast_scores:
            summary["mean_coherence_contrast"] = sum(contrast_scores) / len(contrast_scores)
            summary["positive_better_rate"] = sum(1 for c in contrast_scores if c > 0) / len(contrast_scores)
        
        return summary
    
    def analyze_by_category(self, dataset) -> Dict[str, Dict[str, float]]:
        """Analyze coherence performance by TruthfulQA category."""
        category_analysis = {}
        
        for sample in dataset:
            category = sample.get("category", "unknown")
            if category not in category_analysis:
                category_analysis[category] = {"scores": [], "samples": 0}
            
            eval_result = self.evaluate_sample(sample)
            category_analysis[category]["scores"].append(eval_result["coherence_score"])
            category_analysis[category]["samples"] += 1
        
        # Calculate statistics per category
        for category, data in category_analysis.items():
            scores = data["scores"]
            if scores:
                data["mean"] = sum(scores) / len(scores)
                data["min"] = min(scores)
                data["max"] = max(scores)
                # Remove raw scores to save memory
                del data["scores"]
        
        return category_analysis