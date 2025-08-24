"""
SelfCheckGPT-specific benchmark integration.
Provides specialized adapters for self-consistency evaluation using coherence measures.
"""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

from coherify.core.base import PropositionSet, Proposition
from coherify.benchmarks.adapters import BenchmarkAdapter
from coherify.core.base import CoherenceMeasure
from coherify.benchmarks.native_metrics import SelfCheckGPTMetrics


class SelfCheckGPTAdapter(BenchmarkAdapter):
    """
    Specialized adapter for SelfCheckGPT-style evaluation.

    SelfCheckGPT evaluates consistency by comparing multiple generations
    from the same prompt. This adapter helps structure that data for
    coherence evaluation.
    """

    def __init__(
        self,
        consistency_mode: str = "multi_sample",
        segment_responses: bool = True,
        min_samples: int = 2,
    ):
        """
        Initialize SelfCheckGPT adapter.

        Args:
            consistency_mode: "multi_sample" or "sentence_level"
            segment_responses: Whether to segment responses into sentences
            min_samples: Minimum number of samples required
        """
        super().__init__("selfcheckgpt")
        self.consistency_mode = consistency_mode
        self.segment_responses = segment_responses
        self.min_samples = min_samples

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert SelfCheckGPT sample to PropositionSet."""
        prompt = sample.get("prompt", sample.get("question", ""))

        if self.consistency_mode == "multi_sample":
            return self._adapt_multi_sample(sample, prompt)
        elif self.consistency_mode == "sentence_level":
            return self._adapt_sentence_level(sample, prompt)
        else:
            raise ValueError(f"Unknown consistency mode: {self.consistency_mode}")

    def _adapt_multi_sample(
        self, sample: Dict[str, Any], prompt: str
    ) -> PropositionSet:
        """Adapt multi-sample consistency evaluation."""
        # Get multiple generated responses
        responses = []

        # Try different keys for responses
        if "sampled_answers" in sample:
            responses = sample["sampled_answers"]
        elif "generations" in sample:
            responses = sample["generations"]
        elif "responses" in sample:
            responses = sample["responses"]
        elif "original_answer" in sample:
            # Single response case
            responses = [sample["original_answer"]]
        else:
            raise ValueError("No responses found in sample")

        if len(responses) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, got {len(responses)}"
            )

        # Create propositions from responses
        props = []
        for i, response in enumerate(responses):
            if self.segment_responses:
                # Segment each response and mark with sample index
                segments = [s.strip() for s in response.split(".") if s.strip()]
                for j, segment in enumerate(segments):
                    prop_metadata = {
                        "sample_index": i,
                        "segment_index": j,
                        "full_response": response,
                    }
                    props.append(Proposition(text=segment, metadata=prop_metadata))
            else:
                # Use full response as proposition
                prop_metadata = {"sample_index": i, "full_response": response}
                props.append(Proposition(text=response, metadata=prop_metadata))

        metadata = {
            "benchmark": "selfcheckgpt",
            "consistency_mode": "multi_sample",
            "num_samples": len(responses),
            "segmented": self.segment_responses,
        }

        return PropositionSet(propositions=props, context=prompt, metadata=metadata)

    def _adapt_sentence_level(
        self, sample: Dict[str, Any], prompt: str
    ) -> PropositionSet:
        """Adapt sentence-level consistency evaluation."""
        # Extract the main response to check
        main_response = sample.get("original_answer", sample.get("main_response", ""))

        if not main_response:
            raise ValueError("No main response found for sentence-level evaluation")

        # Segment main response into sentences
        sentences = [s.strip() for s in main_response.split(".") if s.strip()]

        # Create propositions with sentence-level metadata
        props = []
        for i, sentence in enumerate(sentences):
            prop_metadata = {
                "sentence_index": i,
                "main_response": main_response,
                "evaluation_target": True,  # Mark as target for evaluation
            }
            props.append(Proposition(text=sentence, metadata=prop_metadata))

        metadata = {
            "benchmark": "selfcheckgpt",
            "consistency_mode": "sentence_level",
            "num_sentences": len(sentences),
            "main_response": main_response,
        }

        return PropositionSet(propositions=props, context=prompt, metadata=metadata)

    def create_consistency_groups(
        self, sample: Dict[str, Any]
    ) -> Dict[str, PropositionSet]:
        """
        Create proposition sets grouped by semantic content.
        Useful for analyzing consistency within topics.
        """
        prop_set = self.adapt_single(sample)

        if self.consistency_mode != "multi_sample":
            return {"all": prop_set}

        # Group propositions by sample
        sample_groups = defaultdict(list)
        for prop in prop_set.propositions:
            sample_idx = prop.metadata.get("sample_index", 0)
            sample_groups[sample_idx].append(prop)

        # Create proposition sets for each sample group
        groups = {}
        for sample_idx, props in sample_groups.items():
            groups[f"sample_{sample_idx}"] = PropositionSet(
                propositions=props,
                context=prop_set.context,
                metadata={**prop_set.metadata, "group": f"sample_{sample_idx}"},
            )

        return groups


class SelfCheckGPTEvaluator:
    """
    Specialized evaluator for SelfCheckGPT using coherence measures.
    Provides consistency-focused evaluation metrics.
    """

    def __init__(self, coherence_measure: CoherenceMeasure, consistency_method: str = "bertscore"):
        self.coherence_measure = coherence_measure
        self.adapter = SelfCheckGPTAdapter()
        self.consistency_method = consistency_method

    def evaluate_consistency(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate consistency of multiple generations using SelfCheckGPT methods."""
        # Overall coherence across all samples
        prop_set = self.adapter.adapt_single(sample)
        overall_result = self.coherence_measure.compute(prop_set)

        evaluation = {
            "overall_coherence": overall_result.score,
            "overall_details": overall_result.details,
            "num_propositions": len(prop_set),
            "num_samples": prop_set.metadata.get("num_samples", 1),
        }

        # Extract main response and sampled responses for consistency checking
        main_response = ""
        sampled_responses = []
        
        # Try different keys for responses
        if "original_answer" in sample:
            main_response = sample["original_answer"]
        elif "main_response" in sample:
            main_response = sample["main_response"]
        
        if "sampled_answers" in sample:
            sampled_responses = sample["sampled_answers"]
        elif "generations" in sample:
            sampled_responses = sample["generations"]
        elif "responses" in sample:
            sampled_responses = sample["responses"]
        
        # Apply SelfCheckGPT consistency checking methods
        if main_response and sampled_responses:
            consistency_scores = self._calculate_consistency_scores(
                main_response, sampled_responses, sample.get("question", sample.get("prompt", ""))
            )
            evaluation.update(consistency_scores)

        # Per-sample coherence
        sample_groups = self.adapter.create_consistency_groups(sample)
        sample_coherences = {}

        for group_name, group_set in sample_groups.items():
            if len(group_set) > 1:  # Need multiple propositions for coherence
                group_result = self.coherence_measure.compute(group_set)
                sample_coherences[group_name] = group_result.score

        if sample_coherences:
            evaluation["sample_coherences"] = sample_coherences
            evaluation["mean_sample_coherence"] = np.mean(
                list(sample_coherences.values())
            )
            evaluation["sample_coherence_std"] = np.std(
                list(sample_coherences.values())
            )

        # Consistency metrics
        if len(sample_coherences) > 1:
            coherence_values = list(sample_coherences.values())
            evaluation["coherence_consistency"] = 1.0 - np.std(
                coherence_values
            )  # Higher when more consistent
            evaluation["min_sample_coherence"] = min(coherence_values)
            evaluation["max_sample_coherence"] = max(coherence_values)

        return evaluation
    
    def _calculate_consistency_scores(
        self, main_response: str, sampled_responses: List[str], question: str = ""
    ) -> Dict[str, float]:
        """Calculate consistency scores using various SelfCheckGPT methods."""
        consistency_scores = {}
        
        # BERTScore consistency
        if self.consistency_method == "bertscore" or self.consistency_method == "all":
            bertscore = SelfCheckGPTMetrics.check_consistency_bertscore(
                main_response, sampled_responses
            )
            consistency_scores["bertscore_consistency"] = bertscore
        
        # NLI consistency
        if self.consistency_method == "nli" or self.consistency_method == "all":
            nli_score = SelfCheckGPTMetrics.check_consistency_nli(
                main_response, sampled_responses
            )
            consistency_scores["nli_consistency"] = nli_score
        
        # N-gram consistency
        if self.consistency_method == "ngram" or self.consistency_method == "all":
            ngram_score = SelfCheckGPTMetrics.check_consistency_ngram(
                main_response, sampled_responses
            )
            consistency_scores["ngram_consistency"] = ngram_score
        
        # QA-based consistency
        if self.consistency_method == "qa" or self.consistency_method == "all":
            qa_score = SelfCheckGPTMetrics.check_consistency_qa_based(
                main_response, sampled_responses, question
            )
            consistency_scores["qa_consistency"] = qa_score
        
        # Calculate overall consistency score as average
        if consistency_scores:
            consistency_scores["overall_consistency"] = np.mean(list(consistency_scores.values()))
        
        return consistency_scores

    def evaluate_sentence_reliability(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate reliability of individual sentences based on consistency.
        """
        # Switch adapter to sentence-level mode
        sentence_adapter = SelfCheckGPTAdapter(consistency_mode="sentence_level")
        prop_set = sentence_adapter.adapt_single(sample)

        sentence_scores = []
        for prop in prop_set.propositions:
            # Create mini proposition set with just this sentence and context
            mini_set = PropositionSet(propositions=[prop], context=prop_set.context)
            # Score would be 1.0 for single proposition, so we need different approach
            # For now, we'll score based on sentence length and complexity as proxy
            sentence_score = self._score_sentence_reliability(prop.text)
            sentence_scores.append(
                {
                    "sentence": prop.text,
                    "score": sentence_score,
                    "index": prop.metadata.get("sentence_index", 0),
                }
            )

        overall_result = self.coherence_measure.compute(prop_set)

        return {
            "overall_coherence": overall_result.score,
            "sentence_scores": sentence_scores,
            "mean_sentence_score": np.mean([s["score"] for s in sentence_scores]),
            "num_sentences": len(sentence_scores),
        }

    def _score_sentence_reliability(self, sentence: str) -> float:
        """
        Simple heuristic for sentence reliability.
        In practice, this would use comparison with other generations.
        """
        # Simple heuristics for demonstration
        words = sentence.split()

        # Longer sentences might be more specific/reliable
        length_score = min(len(words) / 20.0, 1.0)

        # Sentences with specific terms might be more reliable
        specific_terms = ["the", "is", "was", "will", "can", "should", "must"]
        specificity_score = sum(
            1 for word in words if word.lower() in specific_terms
        ) / len(words)

        # Combine scores
        return (length_score + specificity_score) / 2.0

    def compare_generation_strategies(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare coherence across different generation strategies or models.
        """
        strategy_results = defaultdict(list)

        for sample in samples:
            strategy = sample.get("strategy", sample.get("model", "unknown"))
            eval_result = self.evaluate_consistency(sample)
            strategy_results[strategy].append(eval_result["overall_coherence"])

        comparison = {}
        for strategy, scores in strategy_results.items():
            comparison[strategy] = {
                "mean_coherence": np.mean(scores),
                "std_coherence": np.std(scores),
                "num_samples": len(scores),
                "scores": scores,
            }

        return comparison

    def analyze_consistency_patterns(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in consistency across multiple generations.
        """
        evaluation = self.evaluate_consistency(sample)

        # Extract patterns from the evaluation
        patterns = {
            "high_consistency": evaluation.get("coherence_consistency", 0) > 0.8,
            "low_consistency": evaluation.get("coherence_consistency", 0) < 0.3,
            "outlier_samples": [],
        }

        # Identify outlier samples
        if "sample_coherences" in evaluation:
            mean_coherence = evaluation["mean_sample_coherence"]
            std_coherence = evaluation["sample_coherence_std"]

            for sample_name, coherence in evaluation["sample_coherences"].items():
                if abs(coherence - mean_coherence) > 2 * std_coherence:
                    patterns["outlier_samples"].append(
                        {
                            "sample": sample_name,
                            "coherence": coherence,
                            "deviation": abs(coherence - mean_coherence),
                        }
                    )

        return patterns
