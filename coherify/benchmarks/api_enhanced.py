"""
API-enhanced benchmark adapters using external providers.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.base import Proposition, PropositionSet
from ..providers.base import ModelResponse
from ..providers.manager import ModelProvider, get_provider
from .adapters import QABenchmarkAdapter


@dataclass
class APIBenchmarkConfig:
    """Configuration for API-enhanced benchmark evaluation."""

    provider_name: Optional[str] = None
    model_name: Optional[str] = None
    use_model_generation: bool = True
    temperature_range: List[float] = None
    num_generations_per_prompt: int = 3
    enable_answer_expansion: bool = True
    enable_confidence_scoring: bool = True

    def __post_init__(self):
        if self.temperature_range is None:
            self.temperature_range = [0.3, 0.7, 1.0]


class APIEnhancedQAAdapter(QABenchmarkAdapter):
    """QA benchmark adapter enhanced with external API providers."""

    def __init__(
        self,
        benchmark_name: str,
        config: Optional[APIBenchmarkConfig] = None,
        provider: Optional[ModelProvider] = None,
        **kwargs,
    ):
        """
        Initialize API-enhanced QA adapter.

        Args:
            benchmark_name: Name of the benchmark
            config: API configuration
            provider: External model provider
            **kwargs: Additional arguments for base QABenchmarkAdapter
        """
        super().__init__(benchmark_name, **kwargs)
        self.config = config or APIBenchmarkConfig()
        self.provider = provider or get_provider(self.config.provider_name)

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert QA sample with API-enhanced answer generation."""
        # Get base adaptation
        base_prop_set = super().adapt_single(sample)

        # Enhance with API-generated responses
        if self.config.use_model_generation:
            enhanced_prop_set = self._enhance_with_api_generation(sample, base_prop_set)
            return enhanced_prop_set

        return base_prop_set

    def _enhance_with_api_generation(
        self, sample: Dict[str, Any], base_prop_set: PropositionSet
    ) -> PropositionSet:
        """Enhance proposition set with API-generated content."""
        question = sample[self.question_key]
        original_answer = sample.get(self.answer_key, "")

        enhanced_propositions = list(base_prop_set.propositions)
        metadata = dict(base_prop_set.metadata)

        try:
            # Generate additional responses using API
            if self.config.num_generations_per_prompt > 0:
                api_responses = self._generate_api_responses(question)

                # Convert API responses to propositions
                for i, response in enumerate(api_responses):
                    # Create proposition from API response
                    api_prop = Proposition(
                        text=response.text,
                        metadata={
                            "source": "api_generated",
                            "provider": self.provider.provider_name,
                            "model": response.model_name,
                            "temperature": response.temperature,
                            "generation_index": i,
                            "confidence": response.confidence,
                        },
                    )
                    enhanced_propositions.append(api_prop)

                # Update metadata
                metadata.update(
                    {
                        "api_enhanced": True,
                        "api_generations": len(api_responses),
                        "provider": self.provider.provider_name,
                        "original_answer": original_answer,
                    }
                )

            # Add answer expansion if enabled
            if self.config.enable_answer_expansion and original_answer:
                expanded_answer = self._expand_answer(question, original_answer)
                if expanded_answer:
                    expanded_prop = Proposition(
                        text=expanded_answer.text,
                        metadata={
                            "source": "api_expanded",
                            "original_answer": original_answer,
                            "provider": self.provider.provider_name,
                        },
                    )
                    enhanced_propositions.append(expanded_prop)

            # Add confidence scoring if enabled
            if self.config.enable_confidence_scoring:
                self._add_confidence_scores(enhanced_propositions, question)

        except Exception as e:
            print(
                f"Warning: API enhancement failed for question '{question[:50]}...': {e}"
            )
            # Return base proposition set on failure
            return base_prop_set

        return PropositionSet(
            propositions=enhanced_propositions,
            context=base_prop_set.context,
            metadata=metadata,
        )

    def _generate_api_responses(self, question: str) -> List[ModelResponse]:
        """Generate multiple API responses for a question."""
        responses = []

        try:
            # Generate with different temperatures
            for temp in self.config.temperature_range:
                for i in range(self.config.num_generations_per_prompt):
                    response = self.provider.generate_text(
                        prompt=f"Answer this question accurately and concisely: {question}",
                        max_tokens=200,
                        temperature=temp,
                    )
                    responses.append(response)

                    # Add small delay to avoid rate limiting
                    time.sleep(0.1)

        except Exception as e:
            print(f"Warning: API response generation failed: {e}")

        return responses

    def _expand_answer(
        self, question: str, original_answer: str
    ) -> Optional[ModelResponse]:
        """Expand and elaborate on the original answer."""
        try:
            expansion_prompt = f"""
Given this question and answer, provide a more detailed and comprehensive response that maintains the same accuracy while adding helpful context and explanation.

Question: {question}
Original Answer: {original_answer}

Expanded Answer:"""

            response = self.provider.generate_text(
                prompt=expansion_prompt, max_tokens=300, temperature=0.5
            )

            return response

        except Exception as e:
            print(f"Warning: Answer expansion failed: {e}")
            return None

    def _add_confidence_scores(self, propositions: List[Proposition], question: str):
        """Add confidence scores to propositions."""
        if not self.config.enable_confidence_scoring:
            return

        try:
            for prop in propositions:
                if prop.metadata.get("source") in ["api_generated", "api_expanded"]:
                    # Get confidence score from API
                    confidence_score = self._get_answer_confidence(question, prop.text)
                    prop.metadata["api_confidence"] = confidence_score

        except Exception as e:
            print(f"Warning: Confidence scoring failed: {e}")

    def _get_answer_confidence(self, question: str, answer: str) -> float:
        """Get confidence score for an answer."""
        try:
            confidence_prompt = f"""
Rate the confidence/reliability of this answer to the question on a scale from 0.0 to 1.0, where:
- 0.0 means completely unreliable/incorrect
- 0.5 means uncertain/partially correct
- 1.0 means highly reliable/correct

Question: {question}
Answer: {answer}

Confidence score (respond with just a number):"""

            response = self.provider.generate_text(
                prompt=confidence_prompt, max_tokens=5, temperature=0.1
            )

            try:
                score = float(response.text.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5  # Default moderate confidence

        except Exception as e:
            print(f"Warning: Confidence scoring failed: {e}")
            return 0.5

    def adapt_dataset_with_api_enhancement(
        self,
        dataset,
        batch_size: int = 10,
        progress_callback: Optional[callable] = None,
    ) -> List[PropositionSet]:
        """Adapt entire dataset with API enhancement and batching."""
        results = []
        total_samples = len(dataset)

        for i in range(0, total_samples, batch_size):
            batch = dataset[i : i + batch_size]
            batch_results = []

            for sample in batch:
                try:
                    prop_set = self.adapt_single(sample)
                    batch_results.append(prop_set)
                except Exception as e:
                    print(f"Warning: Failed to process sample {i}: {e}")
                    # Create fallback proposition set
                    fallback_prop_set = super().adapt_single(sample)
                    batch_results.append(fallback_prop_set)

            results.extend(batch_results)

            # Progress callback
            if progress_callback:
                progress = len(results) / total_samples
                progress_callback(progress, len(results), total_samples)

            # Rate limiting between batches
            if i + batch_size < total_samples:
                time.sleep(1.0)

        return results


class APIBenchmarkEvaluator:
    """Evaluator for API-enhanced benchmark performance."""

    def __init__(
        self,
        adapter: APIEnhancedQAAdapter,
        coherence_measures: List,
        config: Optional[APIBenchmarkConfig] = None,
    ):
        """
        Initialize API benchmark evaluator.

        Args:
            adapter: API-enhanced benchmark adapter
            coherence_measures: List of coherence measures to evaluate
            config: API configuration
        """
        self.adapter = adapter
        self.coherence_measures = coherence_measures
        self.config = config or APIBenchmarkConfig()

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample with API enhancement."""
        # Adapt sample to proposition set
        prop_set = self.adapter.adapt_single(sample)

        # Evaluate with coherence measures
        results = {
            "sample": sample,
            "proposition_set": prop_set,
            "coherence_scores": {},
            "api_metadata": prop_set.metadata,
            "evaluation_time": 0.0,
        }

        start_time = time.time()

        for measure in self.coherence_measures:
            try:
                coherence_result = measure.compute(prop_set)
                results["coherence_scores"][measure.__class__.__name__] = {
                    "score": coherence_result.score,
                    "details": coherence_result.details,
                }
            except Exception as e:
                print(
                    f"Warning: Coherence evaluation failed for {measure.__class__.__name__}: {e}"
                )
                results["coherence_scores"][measure.__class__.__name__] = {
                    "score": 0.0,
                    "error": str(e),
                }

        results["evaluation_time"] = time.time() - start_time

        return results

    def evaluate_dataset(
        self,
        dataset,
        sample_limit: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate entire dataset with API enhancement."""
        if sample_limit:
            dataset = dataset[:sample_limit]

        results = {
            "dataset_size": len(dataset),
            "sample_results": [],
            "aggregate_scores": {},
            "api_statistics": {},
            "total_evaluation_time": 0.0,
        }

        start_time = time.time()

        for i, sample in enumerate(dataset):
            sample_result = self.evaluate_sample(sample)
            results["sample_results"].append(sample_result)

            if progress_callback:
                progress = (i + 1) / len(dataset)
                progress_callback(progress, i + 1, len(dataset))

        results["total_evaluation_time"] = time.time() - start_time

        # Compute aggregate statistics
        results["aggregate_scores"] = self._compute_aggregate_scores(
            results["sample_results"]
        )
        results["api_statistics"] = self._compute_api_statistics(
            results["sample_results"]
        )

        return results

    def _compute_aggregate_scores(
        self, sample_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute aggregate coherence scores."""
        aggregate = {}

        # Collect scores by measure
        measure_scores = {}
        for result in sample_results:
            for measure_name, measure_result in result["coherence_scores"].items():
                if "score" in measure_result:
                    if measure_name not in measure_scores:
                        measure_scores[measure_name] = []
                    measure_scores[measure_name].append(measure_result["score"])

        # Compute statistics
        for measure_name, scores in measure_scores.items():
            if scores:
                import numpy as np

                aggregate[measure_name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores),
                }

        return aggregate

    def _compute_api_statistics(
        self, sample_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute API usage statistics."""
        stats = {
            "samples_with_api_enhancement": 0,
            "total_api_generations": 0,
            "average_generations_per_sample": 0.0,
            "providers_used": set(),
            "total_api_time": 0.0,
        }

        api_enhanced_samples = 0
        total_generations = 0

        for result in sample_results:
            metadata = result.get("api_metadata", {})

            if metadata.get("api_enhanced"):
                api_enhanced_samples += 1
                generations = metadata.get("api_generations", 0)
                total_generations += generations

                if "provider" in metadata:
                    stats["providers_used"].add(metadata["provider"])

        stats["samples_with_api_enhancement"] = api_enhanced_samples
        stats["total_api_generations"] = total_generations

        if api_enhanced_samples > 0:
            stats["average_generations_per_sample"] = (
                total_generations / api_enhanced_samples
            )

        stats["providers_used"] = list(stats["providers_used"])

        return stats

    def create_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Create a comprehensive evaluation report."""
        report = []
        report.append("=== API-Enhanced Benchmark Evaluation Report ===")
        report.append(f"Dataset Size: {evaluation_results['dataset_size']}")
        report.append(
            f"Total Evaluation Time: {evaluation_results['total_evaluation_time']:.2f}s"
        )

        # API Statistics
        api_stats = evaluation_results["api_statistics"]
        report.append("\n--- API Enhancement Statistics ---")
        report.append(
            f"Samples with API Enhancement: {api_stats['samples_with_api_enhancement']}"
        )
        report.append(f"Total API Generations: {api_stats['total_api_generations']}")
        report.append(
            f"Avg Generations per Sample: {api_stats['average_generations_per_sample']:.1f}"
        )
        report.append(f"Providers Used: {', '.join(api_stats['providers_used'])}")

        # Coherence Scores
        aggregate_scores = evaluation_results["aggregate_scores"]
        report.append("\n--- Coherence Scores ---")

        for measure_name, scores in aggregate_scores.items():
            report.append(f"{measure_name}:")
            report.append(f"  Mean: {scores['mean']:.3f} ± {scores['std']:.3f}")
            report.append(f"  Range: {scores['min']:.3f} - {scores['max']:.3f}")
            report.append(f"  Samples: {scores['count']}")

        return "\n".join(report)
