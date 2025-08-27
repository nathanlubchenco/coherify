"""
FaithBench Adapter

Adapter for FaithBench hallucination detection benchmark that evaluates
faithfulness of AI-generated summaries using coherence analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from coherify.benchmarks.multi_format_adapters import (
    MultiResponseBenchmarkAdapter,
    MultiResponseBenchmarkConfig,
)
from coherify.core.base import Proposition, PropositionSet
from coherify.measures.multi_response import (
    MultiResponseCoherenceMeasure,
    MultiResponseConfig,
)


class FaithBenchLabel(str, Enum):
    """FaithBench hallucination labels with severity mapping."""

    UNWANTED_INTRINSIC = "Unwanted.Intrinsic"  # Severity 3 - Contradicts source
    UNWANTED_EXTRINSIC = "Unwanted.Extrinsic"  # Severity 3 - Adds unsupported info
    UNWANTED = "Unwanted"  # Severity 3 - General unwanted
    QUESTIONABLE = "Questionable"  # Severity 2 - Ambiguous
    BENIGN = "Benign"  # Severity 1 - Acceptable
    CONSISTENT = "Consistent"  # Severity 0 - Faithful

    @property
    def severity(self) -> int:
        """Get severity score for label."""
        severity_map = {
            "Unwanted.Intrinsic": 3,
            "Unwanted.Extrinsic": 3,
            "Unwanted": 3,
            "Questionable": 2,
            "Benign": 1,
            "Consistent": 0,
        }
        return severity_map.get(self.value, 0)


@dataclass
class FaithBenchAnnotation:
    """Single annotation in FaithBench sample."""

    annot_id: int
    annotator_id: str
    annotator_name: str
    label: List[str]
    note: str
    summary_span: str
    summary_start: int
    summary_end: int
    source_span: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None

    @property
    def is_hallucination(self) -> bool:
        """Check if annotation indicates hallucination."""
        return any(
            FaithBenchLabel(lbl).severity >= 2
            for lbl in self.label
            if lbl in [e.value for e in FaithBenchLabel]
        )

    @property
    def max_severity(self) -> int:
        """Get maximum severity across all labels."""
        severities = [
            FaithBenchLabel(lbl).severity
            for lbl in self.label
            if lbl in [e.value for e in FaithBenchLabel]
        ]
        return max(severities) if severities else 0


@dataclass
class FaithBenchMetadata:
    """Metadata for FaithBench sample."""

    summarizer: str
    hhemv1: Optional[float] = None
    hhem_2_1: Optional[float] = None
    trueteacher: Optional[int] = None
    true_nli: Optional[int] = None
    gpt_3_5_turbo: Optional[int] = None
    gpt_4o: Optional[int] = None
    raw_sample_id: Optional[int] = None


@dataclass
class FaithBenchSample:
    """Complete FaithBench sample with annotations."""

    sample_id: int
    source: str
    summary: str
    annotations: List[FaithBenchAnnotation]
    metadata: FaithBenchMetadata

    @property
    def has_hallucination(self) -> bool:
        """Check if sample has any hallucination annotations."""
        return any(ann.is_hallucination for ann in self.annotations)

    @property
    def max_severity(self) -> int:
        """Get maximum severity across all annotations."""
        severities = [ann.max_severity for ann in self.annotations]
        return max(severities) if severities else 0

    def get_aggregated_label(self, strategy: str = "majority") -> bool:
        """Get aggregated hallucination label using specified strategy."""
        if not self.annotations:
            return False

        if strategy == "majority":
            hallucination_votes = sum(
                1 for ann in self.annotations if ann.is_hallucination
            )
            return hallucination_votes > len(self.annotations) / 2
        elif strategy == "worst_case":
            return any(ann.is_hallucination for ann in self.annotations)
        elif strategy == "best_case":
            return all(ann.is_hallucination for ann in self.annotations)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")


@dataclass
class FaithBenchConfig(MultiResponseBenchmarkConfig):
    """Configuration for FaithBench evaluation."""

    aggregation_strategy: str = "majority"  # "majority", "worst_case", "best_case"
    include_span_analysis: bool = True  # Include span-level analysis
    faithfulness_weight: float = 0.7  # Weight for faithfulness vs coherence
    coherence_weight: float = 0.3  # Weight for coherence analysis
    enable_source_coherence: bool = True  # Analyze source-summary coherence
    enable_internal_coherence: bool = True  # Analyze summary internal coherence
    severity_threshold: int = 2  # Minimum severity for hallucination
    enable_span_coherence: bool = True  # Analyze coherence of problem spans
    challenge_level: str = "hard"  # "easy", "medium", "hard", "all"
    enable_challenging_case_filtering: bool = (
        True  # Focus on cases where SOTA models disagree
    )
    model_disagreement_threshold: float = (
        0.3  # Min disagreement between detection models
    )
    min_model_predictions: int = 3  # Minimum number of model predictions required


class FaithBenchAdapter(MultiResponseBenchmarkAdapter):
    """Adapter for FaithBench hallucination detection benchmark."""

    def __init__(self, config: Optional[FaithBenchConfig] = None, provider=None):
        if config is None:
            config = FaithBenchConfig(
                enable_multi_response=True,
                num_responses_per_sample=3,
                temperature_range=(0.1, 0.5),  # Lower for faithful summarization
                reasoning_trace_enabled=True,
            )
        super().__init__("FaithBench", config, provider)

    def filter_challenging_cases(
        self, dataset: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter for challenging cases where SOTA detection models disagree.

        FaithBench specifically includes challenging hallucinations made by 10 modern LLMs
        on which popular, state-of-the-art hallucination detection models disagreed.
        """
        if not self.config.enable_challenging_case_filtering:
            return dataset

        challenging_cases = []

        for sample in dataset:
            difficulty_score = self.evaluate_detection_difficulty(sample)

            if self.config.challenge_level == "all":
                challenging_cases.append(sample)
            elif self.config.challenge_level == "easy" and difficulty_score < 0.3:
                challenging_cases.append(sample)
            elif (
                self.config.challenge_level == "medium"
                and 0.3 <= difficulty_score < 0.7
            ):
                challenging_cases.append(sample)
            elif self.config.challenge_level == "hard" and difficulty_score >= 0.7:
                challenging_cases.append(sample)

        return challenging_cases

    def evaluate_detection_difficulty(self, sample: Dict[str, Any]) -> float:
        """
        Measure how difficult a case is for detection models based on model disagreement.

        Returns:
            float: Difficulty score from 0.0 (easy) to 1.0 (very challenging)
        """
        faithbench_sample = self._parse_faithbench_sample(sample)
        metadata = faithbench_sample.metadata

        # Collect predictions from different detection models in metadata
        model_predictions = []

        # Extract available model predictions (these are from the original FaithBench paper)
        if metadata.hhemv1 is not None:
            model_predictions.append(metadata.hhemv1)
        if metadata.hhem_2_1 is not None:
            model_predictions.append(metadata.hhem_2_1)
        if metadata.trueteacher is not None:
            model_predictions.append(float(metadata.trueteacher))
        if metadata.true_nli is not None:
            model_predictions.append(float(metadata.true_nli))
        if metadata.gpt_3_5_turbo is not None:
            model_predictions.append(float(metadata.gpt_3_5_turbo))
        if metadata.gpt_4o is not None:
            model_predictions.append(float(metadata.gpt_4o))

        if len(model_predictions) < self.config.min_model_predictions:
            # Not enough model predictions - use annotation-based difficulty
            return self._calculate_annotation_difficulty(faithbench_sample)

        # Calculate disagreement among models
        if len(model_predictions) <= 1:
            return 0.5  # Neutral difficulty if only one model

        # Normalize predictions to 0-1 range
        normalized_predictions = []
        for pred in model_predictions:
            if isinstance(pred, (int, float)):
                # Assume predictions are either binary (0,1) or probability scores
                if pred > 1.0:
                    # Likely a score that needs normalization
                    normalized_pred = min(pred / 100.0, 1.0) if pred > 10 else pred
                else:
                    normalized_pred = pred
                normalized_predictions.append(normalized_pred)

        if not normalized_predictions:
            return self._calculate_annotation_difficulty(faithbench_sample)

        # Calculate variance as measure of disagreement
        mean_prediction = sum(normalized_predictions) / len(normalized_predictions)
        variance = sum(
            (pred - mean_prediction) ** 2 for pred in normalized_predictions
        ) / len(normalized_predictions)

        # Convert variance to difficulty score
        # Higher variance = more disagreement = more challenging
        max_possible_variance = 0.25  # Maximum variance for binary predictions
        difficulty_from_disagreement = min(variance / max_possible_variance, 1.0)

        # Combine with annotation-based difficulty
        annotation_difficulty = self._calculate_annotation_difficulty(faithbench_sample)

        # Weighted combination: 70% model disagreement, 30% annotation complexity
        final_difficulty = (
            difficulty_from_disagreement * 0.7 + annotation_difficulty * 0.3
        )

        return min(final_difficulty, 1.0)

    def _calculate_annotation_difficulty(self, sample: FaithBenchSample) -> float:
        """Calculate difficulty based on annotation characteristics."""
        if not sample.annotations:
            return 0.0

        difficulty_factors = []

        # Factor 1: Number of annotations (more = more complex)
        num_annotations = len(sample.annotations)
        annotation_complexity = min(num_annotations / 10.0, 1.0)  # Normalize
        difficulty_factors.append(annotation_complexity)

        # Factor 2: Maximum severity level
        max_severity = sample.max_severity
        severity_difficulty = max_severity / 3.0  # Normalize (severity 0-3)
        difficulty_factors.append(severity_difficulty)

        # Factor 3: Mix of label types (more diverse = harder to detect)
        unique_labels = set()
        for ann in sample.annotations:
            unique_labels.update(ann.label)

        label_diversity = len(unique_labels) / 6.0  # Max 6 possible labels
        difficulty_factors.append(label_diversity)

        # Factor 4: Length and complexity of spans
        span_complexity = 0.0
        total_span_length = 0
        for ann in sample.annotations:
            if ann.summary_span:
                span_length = len(ann.summary_span.split())
                total_span_length += span_length

                # Short spans might be harder to detect (subtle)
                # Very long spans might be easier (obvious)
                if span_length < 5:
                    span_complexity += 0.8  # Short spans are subtle
                elif span_length < 15:
                    span_complexity += 0.6  # Medium spans
                else:
                    span_complexity += 0.3  # Long spans are more obvious

        if sample.annotations:
            span_complexity = span_complexity / len(sample.annotations)
        difficulty_factors.append(span_complexity)

        # Factor 5: Presence of intrinsic vs extrinsic hallucinations
        has_intrinsic = any("Intrinsic" in str(ann.label) for ann in sample.annotations)
        has_extrinsic = any("Extrinsic" in str(ann.label) for ann in sample.annotations)

        # Intrinsic hallucinations (contradictions) are often easier to detect
        # Extrinsic hallucinations (unsupported additions) are harder
        if has_intrinsic and has_extrinsic:
            hallucination_type_difficulty = 0.8  # Mixed types = challenging
        elif has_extrinsic:
            hallucination_type_difficulty = 0.7  # Extrinsic = harder to detect
        elif has_intrinsic:
            hallucination_type_difficulty = 0.4  # Intrinsic = easier to detect
        else:
            hallucination_type_difficulty = 0.5  # Unknown type

        difficulty_factors.append(hallucination_type_difficulty)

        # Average all difficulty factors
        return sum(difficulty_factors) / len(difficulty_factors)

    def get_challenge_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze challenge level distribution in dataset."""
        stats = {
            "total_samples": len(dataset),
            "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
            "avg_difficulty": 0.0,
            "model_disagreement_cases": 0,
            "annotation_based_cases": 0,
        }

        difficulty_scores = []

        for sample in dataset:
            difficulty = self.evaluate_detection_difficulty(sample)
            difficulty_scores.append(difficulty)

            # Categorize difficulty
            if difficulty < 0.3:
                stats["difficulty_distribution"]["easy"] += 1
            elif difficulty < 0.7:
                stats["difficulty_distribution"]["medium"] += 1
            else:
                stats["difficulty_distribution"]["hard"] += 1

            # Check if based on model disagreement vs annotations
            faithbench_sample = self._parse_faithbench_sample(sample)
            model_pred_count = sum(
                1
                for attr in [
                    faithbench_sample.metadata.hhemv1,
                    faithbench_sample.metadata.hhem_2_1,
                    faithbench_sample.metadata.trueteacher,
                    faithbench_sample.metadata.true_nli,
                    faithbench_sample.metadata.gpt_3_5_turbo,
                    faithbench_sample.metadata.gpt_4o,
                ]
                if attr is not None
            )

            if model_pred_count >= self.config.min_model_predictions:
                stats["model_disagreement_cases"] += 1
            else:
                stats["annotation_based_cases"] += 1

        if difficulty_scores:
            stats["avg_difficulty"] = sum(difficulty_scores) / len(difficulty_scores)

        return stats

    def filter_by_model_performance(
        self,
        dataset: List[Dict[str, Any]],
        target_model: str = "gpt_3_5_turbo",
        performance_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Filter cases where a specific model performs below threshold.

        This helps focus on cases that are specifically challenging for certain models.
        """
        challenging_for_model = []

        for sample in dataset:
            faithbench_sample = self._parse_faithbench_sample(sample)
            metadata = faithbench_sample.metadata

            # Get model performance on this sample
            model_performance = None
            if target_model == "gpt_3_5_turbo" and metadata.gpt_3_5_turbo is not None:
                model_performance = metadata.gpt_3_5_turbo
            elif target_model == "gpt_4o" and metadata.gpt_4o is not None:
                model_performance = metadata.gpt_4o
            elif target_model == "trueteacher" and metadata.trueteacher is not None:
                model_performance = metadata.trueteacher
            elif target_model == "true_nli" and metadata.true_nli is not None:
                model_performance = metadata.true_nli

            # Include sample if model performs below threshold
            if model_performance is not None:
                # Normalize performance score
                if isinstance(model_performance, (int, float)):
                    normalized_performance = (
                        min(model_performance, 1.0)
                        if model_performance <= 1.0
                        else model_performance / 100.0
                    )

                    # For hallucination detection, we want cases where model failed
                    # (either failed to detect hallucination or false positive)
                    ground_truth = faithbench_sample.get_aggregated_label(
                        self.config.aggregation_strategy
                    )

                    # Calculate if model was wrong
                    model_prediction = (
                        normalized_performance > 0.5
                    )  # Assuming >0.5 means "no hallucination"
                    model_was_wrong = (model_prediction and ground_truth) or (
                        not model_prediction and not ground_truth
                    )

                    if (
                        model_was_wrong
                        or normalized_performance < performance_threshold
                    ):
                        challenging_for_model.append(sample)
            else:
                # If no model performance data, include based on general difficulty
                difficulty = self.evaluate_detection_difficulty(sample)
                if difficulty >= 0.5:
                    challenging_for_model.append(sample)

        return challenging_for_model

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert FaithBench sample to PropositionSet."""
        faithbench_sample = self._parse_faithbench_sample(sample)

        # Create propositions from source and summary
        props = []

        # Add source document as context propositions
        source_sentences = self._segment_text(faithbench_sample.source)
        for i, sentence in enumerate(source_sentences):
            source_prop = Proposition(
                text=sentence,
                metadata={
                    "type": "source",
                    "sentence_index": i,
                    "sample_id": faithbench_sample.sample_id,
                },
            )
            props.append(source_prop)

        # Add summary sentences as target propositions
        summary_sentences = self._segment_text(faithbench_sample.summary)
        for i, sentence in enumerate(summary_sentences):
            # Check if this sentence overlaps with any hallucination spans
            is_hallucinated = self._check_hallucination_overlap(
                sentence, i, faithbench_sample.annotations, faithbench_sample.summary
            )

            summary_prop = Proposition(
                text=sentence,
                metadata={
                    "type": "summary",
                    "sentence_index": i,
                    "is_hallucinated": is_hallucinated,
                    "sample_id": faithbench_sample.sample_id,
                },
            )
            props.append(summary_prop)

        # Context includes the faithfulness evaluation task
        context = f"Faithfulness evaluation: Summary should be faithful to source. Has hallucination: {faithbench_sample.has_hallucination}"

        return PropositionSet(
            propositions=props,
            context=context,
            metadata={
                "benchmark": "FaithBench",
                "sample_id": faithbench_sample.sample_id,
                "summarizer": faithbench_sample.metadata.summarizer,
                "num_annotations": len(faithbench_sample.annotations),
                "max_severity": faithbench_sample.max_severity,
                "aggregated_label": faithbench_sample.get_aggregated_label(
                    self.config.aggregation_strategy
                ),
            },
        )

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format FaithBench sample as faithfulness evaluation prompt."""
        faithbench_sample = self._parse_faithbench_sample(sample)

        prompt = f"""Faithfulness Evaluation Task: Analyze whether the summary is faithful to the source document.

Source Document:
{faithbench_sample.source}

Summary:
{faithbench_sample.summary}

"""

        if self.config.reasoning_trace_enabled:
            prompt += """Please analyze the faithfulness step by step:
1. Identify key claims in the summary
2. Check each claim against the source document
3. Flag any claims that contradict the source (intrinsic hallucinations)
4. Flag any claims that add unsupported information (extrinsic hallucinations)
5. Provide overall faithfulness assessment

Classification: FAITHFUL or HALLUCINATED
Explanation:"""
        else:
            prompt += """Classify the summary as:
FAITHFUL - Summary accurately reflects source content
HALLUCINATED - Summary contains contradictions or unsupported claims

Answer:"""

        return prompt

    def evaluate_responses(
        self, sample: Dict[str, Any], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate FaithBench responses for faithfulness consistency."""
        faithbench_sample = self._parse_faithbench_sample(sample)
        ground_truth = faithbench_sample.get_aggregated_label(
            self.config.aggregation_strategy
        )

        response_analysis = []
        predicted_labels = []
        faithfulness_scores = []

        for i, response in enumerate(responses):
            # Extract predicted faithfulness label
            predicted_faithful = self._extract_faithfulness_label(response)
            is_correct = (
                not predicted_faithful
            ) == ground_truth  # ground_truth=True means hallucinated

            predicted_labels.append(predicted_faithful)

            # Analyze faithfulness reasoning
            faithfulness_analysis = self._analyze_faithfulness_reasoning(
                response, faithbench_sample
            )
            source_usage = self._analyze_source_usage(response, faithbench_sample)

            analysis = {
                "response_index": i,
                "predicted_faithful": predicted_faithful,
                "is_correct": is_correct,
                "faithfulness_analysis": faithfulness_analysis,
                "source_usage": source_usage,
                "response_length": len(response),
            }
            response_analysis.append(analysis)

            # Score faithfulness consistency
            faithfulness_scores.append(
                faithfulness_analysis.get("consistency_score", 0.0)
            )

        # Check prediction consistency across responses
        unique_predictions = set(predicted_labels)
        is_prediction_consistent = len(unique_predictions) <= 1

        # Majority vote
        if predicted_labels:
            faithful_count = sum(1 for label in predicted_labels if label)
            majority_faithful = faithful_count > len(predicted_labels) / 2
        else:
            majority_faithful = True

        # Compute overall scores
        correct_count = sum(
            1 for analysis in response_analysis if analysis["is_correct"]
        )
        accuracy = correct_count / len(responses) if responses else 0.0
        avg_faithfulness_consistency = (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else 0.0
        )

        return {
            "ground_truth_hallucinated": ground_truth,
            "response_analysis": response_analysis,
            "predicted_labels": predicted_labels,
            "is_prediction_consistent": is_prediction_consistent,
            "majority_faithful": majority_faithful,
            "accuracy": accuracy,
            "faithfulness_consistency": avg_faithfulness_consistency,
            "faithbench_score": accuracy
            * avg_faithfulness_consistency,  # Combined score
            "max_severity": faithbench_sample.max_severity,
            "num_annotations": len(faithbench_sample.annotations),
            "span_analysis": (
                self._analyze_span_consistency(faithbench_sample, responses)
                if self.config.include_span_analysis
                else None
            ),
        }

    def _parse_faithbench_sample(self, sample: Dict[str, Any]) -> FaithBenchSample:
        """Parse raw FaithBench sample into structured format."""
        # Parse annotations
        annotations = []
        for ann_data in sample.get("annotations", []):
            annotation = FaithBenchAnnotation(
                annot_id=ann_data.get("annot_id", 0),
                annotator_id=ann_data.get("annotator_id", ""),
                annotator_name=ann_data.get("annotator_name", ""),
                label=ann_data.get("label", []),
                note=ann_data.get("note", ""),
                summary_span=ann_data.get("summary_span", ""),
                summary_start=ann_data.get("summary_start", 0),
                summary_end=ann_data.get("summary_end", 0),
                source_span=ann_data.get("source_span"),
                source_start=ann_data.get("source_start"),
                source_end=ann_data.get("source_end"),
            )
            annotations.append(annotation)

        # Parse metadata
        metadata_dict = sample.get("metadata", {})
        metadata = FaithBenchMetadata(
            summarizer=metadata_dict.get("summarizer", "unknown"),
            hhemv1=metadata_dict.get("hhemv1"),
            hhem_2_1=metadata_dict.get("hhem-2.1"),
            trueteacher=metadata_dict.get("trueteacher"),
            true_nli=metadata_dict.get("true_nli"),
            gpt_3_5_turbo=metadata_dict.get("gpt_3.5_turbo"),
            gpt_4o=metadata_dict.get("gpt_4o"),
            raw_sample_id=metadata_dict.get("raw_sample_id"),
        )

        return FaithBenchSample(
            sample_id=sample.get("sample_id", 0),
            source=sample.get("source", ""),
            summary=sample.get("summary", ""),
            annotations=annotations,
            metadata=metadata,
        )

    def _segment_text(self, text: str) -> List[str]:
        """Segment text into sentences."""
        # Simple sentence segmentation
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in ".!?" and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences

    def _check_hallucination_overlap(
        self,
        sentence: str,
        sentence_idx: int,
        annotations: List[FaithBenchAnnotation],
        full_summary: str,
    ) -> bool:
        """Check if sentence overlaps with any hallucination spans."""
        # Find sentence boundaries in full summary
        sentences = self._segment_text(full_summary)
        if sentence_idx >= len(sentences):
            return False

        # Estimate sentence position in full text
        sentence_start = sum(len(sentences[i]) + 1 for i in range(sentence_idx))
        sentence_end = sentence_start + len(sentence)

        # Check overlap with annotation spans
        for annotation in annotations:
            if annotation.is_hallucination:
                # Check if there's overlap between sentence and annotation span
                overlap_start = max(sentence_start, annotation.summary_start)
                overlap_end = min(sentence_end, annotation.summary_end)
                if overlap_start < overlap_end:
                    return True

        return False

    def _extract_faithfulness_label(self, response: str) -> bool:
        """Extract faithfulness label from response (True = faithful, False = hallucinated)."""
        response_upper = response.upper()

        # Look for explicit labels
        if "FAITHFUL" in response_upper and "HALLUCINATED" not in response_upper:
            return True
        elif "HALLUCINATED" in response_upper or "HALLUCINATION" in response_upper:
            return False

        # Look for other indicators
        faithful_indicators = ["ACCURATE", "CORRECT", "CONSISTENT", "SUPPORTED"]
        hallucinated_indicators = [
            "CONTRADICTS",
            "UNSUPPORTED",
            "ADDS INFORMATION",
            "INCONSISTENT",
        ]

        faithful_count = sum(
            1 for indicator in faithful_indicators if indicator in response_upper
        )
        hallucinated_count = sum(
            1 for indicator in hallucinated_indicators if indicator in response_upper
        )

        if faithful_count > hallucinated_count:
            return True
        elif hallucinated_count > faithful_count:
            return False

        # Default to faithful if unclear
        return True

    def _analyze_faithfulness_reasoning(
        self, response: str, sample: FaithBenchSample
    ) -> Dict[str, Any]:
        """Analyze quality of faithfulness reasoning in response."""
        reasoning_indicators = [
            "source document",
            "source",
            "document",
            "original text",
            "claims",
            "contradicts",
            "supports",
            "accurate",
            "consistent",
            "adds information",
            "unsupported",
            "faithful",
            "hallucination",
        ]

        response_lower = response.lower()

        # Count reasoning indicators
        indicator_count = sum(
            1 for indicator in reasoning_indicators if indicator in response_lower
        )

        # Check for specific analysis patterns
        has_claim_analysis = any(
            pattern in response_lower
            for pattern in ["claim", "statement", "assertion", "fact"]
        )

        has_source_comparison = any(
            pattern in response_lower
            for pattern in [
                "source says",
                "document states",
                "according to",
                "source mentions",
            ]
        )

        has_contradiction_analysis = any(
            pattern in response_lower
            for pattern in [
                "contradicts",
                "conflicts with",
                "disagrees",
                "inconsistent",
            ]
        )

        return {
            "reasoning_indicator_count": indicator_count,
            "has_claim_analysis": has_claim_analysis,
            "has_source_comparison": has_source_comparison,
            "has_contradiction_analysis": has_contradiction_analysis,
            "reasoning_density": (
                indicator_count / len(response.split()) if response else 0
            ),
            "consistency_score": min(
                1.0,
                (
                    indicator_count
                    + int(has_claim_analysis) * 2
                    + int(has_source_comparison) * 2
                    + int(has_contradiction_analysis) * 2
                )
                / 10,
            ),
        }

    def _analyze_source_usage(
        self, response: str, sample: FaithBenchSample
    ) -> Dict[str, Any]:
        """Analyze how well response uses source document information."""
        source_words = set(sample.source.lower().split())
        response_words = set(response.lower().split())

        # Simple overlap metric
        overlap = len(source_words.intersection(response_words))
        source_coverage = overlap / len(source_words) if source_words else 0.0

        # Check for explicit source references
        source_references = 0
        for sentence in response.split("."):
            if any(ref in sentence.lower() for ref in ["source", "document", "text"]):
                source_references += 1

        return {
            "source_word_overlap": overlap,
            "source_coverage": source_coverage,
            "source_references": source_references,
            "uses_source_explicitly": source_references > 0,
        }

    def _analyze_span_consistency(
        self, sample: FaithBenchSample, responses: List[str]
    ) -> Dict[str, Any]:
        """Analyze consistency in identifying problematic spans."""
        if not self.config.include_span_analysis:
            return {}

        span_mentions = []
        for response in responses:
            # Simple heuristic: look for quoted text or specific mentions
            mentioned_spans = []
            for annotation in sample.annotations:
                if annotation.summary_span.lower() in response.lower():
                    mentioned_spans.append(annotation.summary_span)
            span_mentions.append(mentioned_spans)

        # Compute consistency in span identification
        all_spans = set()
        for spans in span_mentions:
            all_spans.update(spans)

        if not all_spans:
            consistency_score = 1.0  # No spans mentioned consistently
        else:
            span_agreement = {}
            for span in all_spans:
                agreement = sum(1 for spans in span_mentions if span in spans)
                span_agreement[span] = agreement / len(responses)

            consistency_score = sum(span_agreement.values()) / len(span_agreement)

        return {
            "span_mentions_per_response": span_mentions,
            "total_unique_spans_mentioned": len(all_spans),
            "span_consistency_score": consistency_score,
            "most_consistent_spans": (
                [span for span, agreement in span_agreement.items() if agreement > 0.5]
                if "span_agreement" in locals()
                else []
            ),
        }


class FaithfulnessCoherence(MultiResponseCoherenceMeasure):
    """Coherence measure specialized for faithfulness evaluation."""

    def __init__(self, config: Optional[MultiResponseConfig] = None, provider=None):
        if config is None:
            from coherify.measures.hybrid import HybridCoherence

            base_measure = HybridCoherence()
            config = MultiResponseConfig(
                num_responses=3,
                temperature_range=(0.1, 0.4),  # Lower for faithful summarization
                consistency_threshold=0.8,
            )
        else:
            from coherify.measures.hybrid import HybridCoherence

            base_measure = HybridCoherence()

        super().__init__(base_measure, config, provider)

    def evaluate_source_summary_coherence(
        self, source: str, summary: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate coherence between source document and summary.

        Core functionality for FaithBench-style faithfulness evaluation.
        """
        # Create proposition sets
        source_sentences = self._segment_text(source)
        summary_sentences = self._segment_text(summary)

        source_props = [
            Proposition(text=sent, metadata={"type": "source"})
            for sent in source_sentences
        ]
        summary_props = [
            Proposition(text=sent, metadata={"type": "summary"})
            for sent in summary_sentences
        ]

        # Evaluate source-summary coherence
        combined_props = source_props + summary_props
        combined_set = PropositionSet(
            propositions=combined_props,
            context=context or "Source-summary faithfulness evaluation",
        )

        source_summary_result = self.base_measure.compute(combined_set)

        # Evaluate summary internal coherence
        if len(summary_props) > 1:
            summary_only_set = PropositionSet(
                propositions=summary_props, context="Summary internal coherence"
            )
            summary_coherence_result = self.base_measure.compute(summary_only_set)
            summary_coherence = summary_coherence_result.score
        else:
            summary_coherence = 1.0

        # Combined faithfulness score (emphasize source-summary coherence)
        faithfulness_score = source_summary_result.score * 0.8 + summary_coherence * 0.2

        return {
            "source_summary_coherence": source_summary_result.score,
            "summary_internal_coherence": summary_coherence,
            "overall_faithfulness": faithfulness_score,
            "num_source_sentences": len(source_sentences),
            "num_summary_sentences": len(summary_sentences),
            "faithfulness_verdict": (
                "faithful"
                if faithfulness_score > self.config.consistency_threshold
                else "potentially_hallucinated"
            ),
        }

    def _segment_text(self, text: str) -> List[str]:
        """Simple text segmentation into sentences."""
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in ".!?" and len(current.strip()) > 10:
                sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        return sentences
