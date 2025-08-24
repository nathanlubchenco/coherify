"""
TruthfulQA-specific benchmark integration.
Provides specialized adapters and evaluation metrics for TruthfulQA dataset.
"""

import time
from typing import Dict, Any, List, Optional, Tuple

from coherify.core.base import PropositionSet, Proposition
from coherify.benchmarks.adapters import BenchmarkAdapter
from coherify.core.base import CoherenceMeasure
from coherify.reporting import BenchmarkReporter, ModelInfo, ExampleResult, ErrorInfo
from coherify.benchmarks.native_metrics import TruthfulQAMetrics, BenchmarkMetrics

# Try to import performance validation
try:
    from coherify.benchmarks.native_metrics import BenchmarkPerformanceExpectations
    HAS_PERFORMANCE_VALIDATION = True
except ImportError:
    HAS_PERFORMANCE_VALIDATION = False


class TruthfulQAAdapter(BenchmarkAdapter):
    """
    Specialized adapter for TruthfulQA benchmark.

    TruthfulQA has specific structure with questions, best answers,
    correct answers, and incorrect answers that can be leveraged
    for coherence evaluation.
    """

    def __init__(
        self,
        evaluation_mode: str = "generation",
        include_context: bool = True,
        use_correct_answers: bool = False,
    ):
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

    def _adapt_generation_sample(
        self, sample: Dict[str, Any], question: str
    ) -> PropositionSet:
        """Adapt generation-mode sample."""
        if self.use_correct_answers and "correct_answers" in sample:
            # Use provided correct answers
            answers = sample["correct_answers"]
            props = [Proposition(text=ans) for ans in answers]
        elif "best_answer" in sample:
            # Use best answer, segmented
            answer = sample["best_answer"]
            segments = [s.strip() for s in answer.split(".") if s.strip()]
            props = [Proposition(text=seg) for seg in segments]
        elif "answer" in sample:
            # Use generated answer
            answer = sample["answer"]
            segments = [s.strip() for s in answer.split(".") if s.strip()]
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
            "has_incorrect_answers": "incorrect_answers" in sample,
        }

        return PropositionSet(propositions=props, context=context, metadata=metadata)

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
            "num_choices": len(props),
        }

        return PropositionSet(propositions=props, context=context, metadata=metadata)

    def create_positive_negative_sets(
        self, sample: Dict[str, Any]
    ) -> Dict[str, PropositionSet]:
        """
        Create positive (correct) and negative (incorrect) proposition sets.
        Useful for contrastive coherence evaluation.
        """
        question = sample.get("question", "")
        result = {}

        # Positive set (correct answers)
        if "correct_answers" in sample:
            correct_props = [
                Proposition(text=ans, metadata={"valence": "positive"})
                for ans in sample["correct_answers"]
            ]
            result["positive"] = PropositionSet(
                propositions=correct_props,
                context=question,
                metadata={"valence": "positive", "benchmark": "truthfulqa"},
            )

        # Negative set (incorrect answers)
        if "incorrect_answers" in sample:
            incorrect_props = [
                Proposition(text=ans, metadata={"valence": "negative"})
                for ans in sample["incorrect_answers"]
            ]
            result["negative"] = PropositionSet(
                propositions=incorrect_props,
                context=question,
                metadata={"valence": "negative", "benchmark": "truthfulqa"},
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
            "category": sample.get("category", "unknown"),
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

            if (
                "positive_coherence" in evaluation
                and "negative_coherence" in evaluation
            ):
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
            "detailed_results": results,
        }

        # Add contrastive analysis if available
        contrast_scores = [
            r.get("coherence_contrast") for r in results if "coherence_contrast" in r
        ]
        if contrast_scores:
            summary["mean_coherence_contrast"] = sum(contrast_scores) / len(
                contrast_scores
            )
            summary["positive_better_rate"] = sum(
                1 for c in contrast_scores if c > 0
            ) / len(contrast_scores)

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


class EnhancedTruthfulQAEvaluator:
    """
    Enhanced TruthfulQA evaluator with comprehensive reporting capabilities.
    Generates detailed reports with examples, timing, and contextual information.
    """
    
    def __init__(
        self,
        coherence_measure: CoherenceMeasure,
        model_info: Optional[ModelInfo] = None,
        reporter: Optional[BenchmarkReporter] = None,
    ):
        """Initialize enhanced evaluator."""
        self.coherence_measure = coherence_measure
        self.model_info = model_info or ModelInfo()
        self.reporter = reporter or BenchmarkReporter()
        self.adapter = TruthfulQAAdapter()
        
        # Tracking
        self.examples = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def evaluate_sample_with_details(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """Evaluate sample and collect detailed information for reporting."""
        try:
            # Get basic evaluation
            prop_set = self.adapter.adapt_single(sample)
            coherence_result = self.coherence_measure.compute(prop_set)
            
            evaluation = {
                "coherence_score": coherence_result.score,
                "coherence_details": coherence_result.details,
                "num_propositions": len(prop_set),
                "category": sample.get("category", "unknown"),
                "sample_index": sample_idx,
                "input": sample.get("question", ""),
                "output": sample.get("best_answer", ""),
            }
            
            # Create example for reporting
            question = sample.get("question", "")
            best_answer = sample.get("best_answer", "")
            
            # Determine correctness based on coherence and available answers
            is_correct = None
            if "correct_answers" in sample and "incorrect_answers" in sample:
                # Use contrastive evaluation
                pos_neg_sets = self.adapter.create_positive_negative_sets(sample)
                if "positive" in pos_neg_sets and "negative" in pos_neg_sets:
                    pos_coherence = self.coherence_measure.compute(pos_neg_sets["positive"]).score
                    neg_coherence = self.coherence_measure.compute(pos_neg_sets["negative"]).score
                    evaluation["positive_coherence"] = pos_coherence
                    evaluation["negative_coherence"] = neg_coherence
                    evaluation["coherence_contrast"] = pos_coherence - neg_coherence
                    
                    # Consider correct if positive coherence is higher
                    is_correct = pos_coherence > neg_coherence
            else:
                # Use coherence threshold
                is_correct = coherence_result.score > 0.6
            
            example = ExampleResult(
                input_text=question,
                output_text=best_answer,
                expected_output=sample.get("correct_answers", [None])[0] if sample.get("correct_answers") else None,
                coherence_score=coherence_result.score,
                is_correct=is_correct,
                category=sample.get("category"),
                metadata={
                    "has_correct_answers": "correct_answers" in sample,
                    "has_incorrect_answers": "incorrect_answers" in sample,
                    "evaluation_mode": self.adapter.evaluation_mode,
                }
            )
            
            self.examples.append(example)
            return evaluation
            
        except Exception as e:
            error = ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                sample_index=sample_idx,
                timestamp=time.time(),
                context={"sample_keys": list(sample.keys())}
            )
            self.errors.append(error)
            
            # Return minimal evaluation result
            return {
                "coherence_score": 0.0,
                "error": str(e),
                "category": sample.get("category", "unknown"),
                "sample_index": sample_idx,
            }
    
    def evaluate_dataset_with_comprehensive_report(
        self,
        dataset: List[Dict[str, Any]],
        evaluation_config: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate dataset and generate comprehensive report with native metrics.
        
        Args:
            dataset: TruthfulQA dataset samples
            evaluation_config: Optional configuration
            predictions: Optional model predictions (if None, uses best_answer)
        
        Returns both standard evaluation results and saves comprehensive report files.
        """
        self.start_time = time.time()
        self.examples = []
        self.errors = []
        
        print(f"📊 Starting comprehensive TruthfulQA evaluation ({len(dataset)} samples)")
        
        # Run evaluation
        results = []
        category_scores = {}
        coherence_scores = []
        
        # Get predictions if not provided
        if predictions is None:
            predictions = [sample.get("best_answer", "") for sample in dataset]
        
        for idx, sample in enumerate(dataset):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(dataset)} samples ({idx/len(dataset)*100:.1f}%)")
            
            eval_result = self.evaluate_sample_with_details(sample, idx)
            results.append(eval_result)
            
            # Track coherence scores for native metric calculation
            if "coherence_score" in eval_result:
                coherence_scores.append(eval_result["coherence_score"])
            
            # Aggregate by category
            category = eval_result["category"]
            if category not in category_scores:
                category_scores[category] = []
            if "coherence_score" in eval_result:
                category_scores[category].append(eval_result["coherence_score"])
        
        self.end_time = time.time()
        
        # Calculate native TruthfulQA metrics
        print("📏 Calculating native TruthfulQA metrics...")
        native_metrics = TruthfulQAMetrics.calculate_metrics(
            predictions=predictions,
            samples=dataset,
            coherence_scores=coherence_scores if coherence_scores else None,
            coherence_threshold=evaluation_config.get("coherence_threshold", 0.6) if evaluation_config else 0.6
        )
        
        # Validate performance against research expectations
        if HAS_PERFORMANCE_VALIDATION and native_metrics.truthful_score is not None:
            truthful_score = native_metrics.truthful_score
            is_realistic, explanation = BenchmarkPerformanceExpectations.is_performance_realistic(
                "truthfulqa", truthful_score
            )
            
            if not is_realistic:
                print(f"⚠️  Performance Warning: {explanation}")
            elif truthful_score > 0:
                expectations = BenchmarkPerformanceExpectations.get_expectations("truthfulqa")
                best_model_score = expectations.get("best_model", 0)
                human_score = expectations.get("human_performance", 0)
                print(f"ℹ️  Research Context: Human performance {human_score:.1%}, Best model (GPT-3) {best_model_score:.1%}")
                
                # Show expected improvement range
                improvement_range = expectations.get("coherence_improvement", (0, 0))
                if isinstance(improvement_range, tuple) and native_metrics.improvement is not None:
                    expected_min, expected_max = improvement_range
                    actual_improvement = native_metrics.improvement
                    print(f"ℹ️  Coherence improvement: {actual_improvement:+.3f} (expected {expected_min:.1%}-{expected_max:.1%})")
        
        # Calculate summary statistics
        evaluation_summary = {
            "num_samples": len(results),
            "mean_coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0,
            "category_means": {
                cat: sum(scores) / len(scores)
                for cat, scores in category_scores.items()
                if scores
            },
            "detailed_results": results,
            "eval_time": self.end_time - self.start_time,
            
            # Add native metrics
            "native_metrics": {
                "truthful_score": native_metrics.truthful_score,
                "informative_score": native_metrics.informative_score,
                "baseline_accuracy": native_metrics.baseline_accuracy,
                "coherence_filtered_accuracy": native_metrics.coherence_filtered_accuracy,
                "improvement": native_metrics.improvement,
            },
            "benchmark_primary_metric": native_metrics.get_primary_metric(),
        }
        
        # Add contrastive analysis if available
        contrast_scores = [r.get("coherence_contrast") for r in results if "coherence_contrast" in r]
        if contrast_scores:
            evaluation_summary["mean_coherence_contrast"] = sum(contrast_scores) / len(contrast_scores)
            evaluation_summary["positive_better_rate"] = sum(1 for c in contrast_scores if c > 0) / len(contrast_scores)
        
        # Generate comprehensive report
        print("📝 Generating comprehensive report...")
        
        report = self.reporter.create_report(
            benchmark_name="TruthfulQA",
            raw_results=evaluation_summary,
            model_info=self.model_info,
            evaluation_config=evaluation_config,
            start_time=self.start_time,
            end_time=self.end_time,
            examples=self.examples,
            errors=self.errors,
        )
        
        # Save report files
        json_path, md_path = self.reporter.save_report(report)
        
        print(f"✅ Comprehensive evaluation completed!")
        print(f"📄 Report saved: {json_path}")
        print(f"📄 Markdown: {md_path}")
        
        # Add file paths to summary
        evaluation_summary["report_files"] = {
            "json": str(json_path),
            "markdown": str(md_path),
        }
        
        return evaluation_summary
    
    def set_model_info(
        self,
        model_name: str,
        provider: str = None,
        temperature: float = None,
        embedding_model: str = None,
        **kwargs
    ):
        """Set model information for reporting."""
        self.model_info = ModelInfo(
            name=model_name,
            provider=provider,
            temperature=temperature,
            embedding_model=embedding_model,
            parameters=kwargs,
        )
