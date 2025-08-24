"""
Multi-Format Benchmark Adapters

Enhanced adapters that support multiple response generation and
coherence evaluation across different benchmark formats.
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import abstractmethod
from dataclasses import dataclass
import re

from coherify.core.base import PropositionSet, Proposition
from coherify.benchmarks.adapters import BenchmarkAdapter


@dataclass
class MultiResponseBenchmarkConfig:
    """Configuration for multi-response benchmark evaluation."""

    enable_multi_response: bool = True
    num_responses_per_sample: int = 3
    temperature_range: Tuple[float, float] = (0.3, 0.8)
    use_self_consistency: bool = True
    use_temperature_variance: bool = True
    reasoning_trace_enabled: bool = False
    max_response_length: int = 512


class MultiResponseBenchmarkAdapter(BenchmarkAdapter):
    """Base adapter for multi-response benchmark evaluation."""

    def __init__(
        self, benchmark_name: str, config: MultiResponseBenchmarkConfig, provider=None
    ):
        super().__init__(benchmark_name)
        self.config = config
        self.provider = provider

    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format sample into a prompt for response generation."""

    @abstractmethod
    def evaluate_responses(
        self, sample: Dict[str, Any], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate generated responses against ground truth."""

    def adapt_single_with_multi_response(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt single sample with multi-response generation and evaluation.

        Returns both PropositionSet and multi-response analysis.
        """
        # Standard single-response adaptation
        prop_set = self.adapt_single(sample)

        result = {
            "proposition_set": prop_set,
            "sample": sample,
            "benchmark_name": self.benchmark_name,
        }

        if not self.config.enable_multi_response or not self.provider:
            return result

        # Generate multiple responses
        prompt = self.format_prompt(sample)

        try:
            responses = self._generate_multiple_responses(prompt)

            # Evaluate responses
            evaluation = self.evaluate_responses(sample, responses)

            # Create proposition sets from responses
            response_prop_sets = []
            for response in responses:
                try:
                    response_prop_set = PropositionSet.from_qa_pair(prompt, response)
                    response_prop_sets.append(response_prop_set)
                except Exception:
                    # Fallback for malformed responses
                    props = [Proposition(text=response)]
                    response_prop_set = PropositionSet(
                        propositions=props, context=prompt
                    )
                    response_prop_sets.append(response_prop_set)

            result.update(
                {
                    "multi_response_enabled": True,
                    "generated_responses": responses,
                    "response_proposition_sets": response_prop_sets,
                    "response_evaluation": evaluation,
                    "prompt": prompt,
                }
            )

        except Exception as e:
            result.update(
                {"multi_response_enabled": False, "multi_response_error": str(e)}
            )

        return result

    def _generate_multiple_responses(self, prompt: str) -> List[str]:
        """Generate multiple responses using different temperatures."""
        if not self.provider:
            return [
                f"Mock response {i+1} to: {prompt[:50]}..."
                for i in range(self.config.num_responses_per_sample)
            ]

        responses = []
        temperatures = self._get_temperature_sequence()

        for temp in temperatures:
            try:
                response = self.provider.generate(
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=self.config.max_response_length,
                )
                responses.append(response)
            except Exception:
                # Fallback to previous response or default
                if responses:
                    responses.append(responses[-1])
                else:
                    responses.append(f"Error generating response at temperature {temp}")

        return responses

    def _get_temperature_sequence(self) -> List[float]:
        """Get sequence of temperatures for multi-response generation."""
        import numpy as np

        return list(
            np.linspace(
                self.config.temperature_range[0],
                self.config.temperature_range[1],
                self.config.num_responses_per_sample,
            )
        )


class GSM8KAdapter(MultiResponseBenchmarkAdapter):
    """Adapter for GSM8K mathematical reasoning benchmark with multi-response support."""

    def __init__(
        self, config: Optional[MultiResponseBenchmarkConfig] = None, provider=None
    ):
        if config is None:
            config = MultiResponseBenchmarkConfig(
                num_responses_per_sample=3,
                reasoning_trace_enabled=True,
                max_response_length=1024,
            )
        super().__init__("GSM8K", config, provider)

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert GSM8K sample to PropositionSet."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Extract the numerical answer if possible
        numerical_answer = self._extract_numerical_answer(answer)

        # Create propositions from the solution steps
        solution_steps = self._extract_solution_steps(answer)

        props = []
        if solution_steps:
            for step in solution_steps:
                props.append(Proposition(text=step))
        else:
            # Fallback: treat entire answer as single proposition
            props.append(Proposition(text=answer))

        # Add final answer as a proposition
        if numerical_answer:
            props.append(Proposition(text=f"The final answer is {numerical_answer}"))

        return PropositionSet(propositions=props, context=question)

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format GSM8K sample as a reasoning prompt."""
        question = sample.get("question", "")

        if self.config.reasoning_trace_enabled:
            return f"""Solve this math problem step by step:

Question: {question}

Please show your work and provide the final numerical answer."""
        else:
            return f"""Solve this math problem:

{question}

Answer:"""

    def evaluate_responses(
        self, sample: Dict[str, Any], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate GSM8K responses for mathematical consistency."""
        ground_truth = sample.get("answer", "")
        ground_truth_number = self._extract_numerical_answer(ground_truth)

        response_analysis = []
        correct_count = 0

        for i, response in enumerate(responses):
            response_number = self._extract_numerical_answer(response)

            is_correct = False
            if ground_truth_number is not None and response_number is not None:
                # Allow for small floating point differences
                is_correct = (
                    abs(float(response_number) - float(ground_truth_number)) < 1e-6
                )

            if is_correct:
                correct_count += 1

            analysis = {
                "response_index": i,
                "extracted_answer": response_number,
                "is_correct": is_correct,
                "response_length": len(response),
                "has_reasoning_steps": self._has_reasoning_steps(response),
            }
            response_analysis.append(analysis)

        return {
            "ground_truth_answer": ground_truth_number,
            "response_analysis": response_analysis,
            "accuracy": correct_count / len(responses) if responses else 0.0,
            "self_consistency_rate": (
                correct_count / len(responses) if responses else 0.0
            ),
            "unanimous_correct": correct_count == len(responses) and correct_count > 0,
        }

    def _extract_numerical_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        # Look for patterns like "The answer is X" or "= X"
        patterns = [
            r"(?:answer is|equals?|=)\s*(\d+(?:\.\d+)?)",
            r"(?:^|[^\d])(\d+(?:\.\d+)?)(?:\s*$|[^\d])",
            r"####\s*(\d+(?:\.\d+)?)",  # GSM8K specific format
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Return last match
                except ValueError:
                    continue

        return None

    def _extract_solution_steps(self, text: str) -> List[str]:
        """Extract individual solution steps from answer text."""
        # Split on common step delimiters
        steps = []

        # Try to split on numbered steps or line breaks
        lines = text.split("\n")
        current_step = ""

        for line in lines:
            line = line.strip()
            if not line:
                if current_step:
                    steps.append(current_step)
                    current_step = ""
                continue

            # Check if this looks like a new step (numbered, bullet, etc.)
            if re.match(r"^[\d\-\*\+•]\s*", line):
                if current_step:
                    steps.append(current_step)
                current_step = line
            else:
                if current_step:
                    current_step += " " + line
                else:
                    current_step = line

        if current_step:
            steps.append(current_step)

        return [
            step for step in steps if len(step.strip()) > 5
        ]  # Filter very short steps

    def _has_reasoning_steps(self, response: str) -> bool:
        """Check if response contains reasoning steps."""
        indicators = [
            "first",
            "then",
            "next",
            "step",
            "calculate",
            "multiply",
            "divide",
            "add",
            "subtract",
            "because",
            "so",
            "therefore",
            "thus",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in indicators)


class HellaSwagAdapter(MultiResponseBenchmarkAdapter):
    """Adapter for HellaSwag commonsense reasoning benchmark."""

    def __init__(
        self, config: Optional[MultiResponseBenchmarkConfig] = None, provider=None
    ):
        if config is None:
            config = MultiResponseBenchmarkConfig(
                num_responses_per_sample=4,  # Match number of choices
                temperature_range=(0.2, 0.6),  # Lower for consistency
                reasoning_trace_enabled=False,
            )
        super().__init__("HellaSwag", config, provider)

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert HellaSwag sample to PropositionSet."""
        context = sample.get("ctx", "")
        endings = sample.get("endings", [])

        # Create propositions from context and all possible endings
        props = [Proposition(text=context)]

        for i, ending in enumerate(endings):
            complete_text = context + " " + ending
            props.append(Proposition(text=complete_text))

        return PropositionSet(propositions=props, context="Commonsense reasoning task")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format HellaSwag sample as multiple choice prompt."""
        context = sample.get("ctx", "")
        endings = sample.get("endings", [])

        prompt = f"""Complete the following scenario with the most plausible continuation:

Context: {context}

Choose the best continuation:
"""

        for i, ending in enumerate(endings):
            prompt += f"{chr(65+i)}. {ending}\n"

        prompt += "\nProvide your answer as a single letter (A, B, C, or D) and explain your reasoning."

        return prompt

    def evaluate_responses(
        self, sample: Dict[str, Any], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate HellaSwag responses for consistency in choice selection."""
        correct_label = sample.get("label", 0)
        correct_letter = chr(65 + correct_label)  # Convert 0,1,2,3 to A,B,C,D

        response_analysis = []
        predicted_choices = []

        for i, response in enumerate(responses):
            predicted_choice = self._extract_choice(response)
            is_correct = predicted_choice == correct_letter

            predicted_choices.append(predicted_choice)

            analysis = {
                "response_index": i,
                "predicted_choice": predicted_choice,
                "is_correct": is_correct,
                "confidence_indicators": self._extract_confidence_indicators(response),
            }
            response_analysis.append(analysis)

        # Check consistency across responses
        unique_choices = set(choice for choice in predicted_choices if choice)
        is_consistent = len(unique_choices) <= 1

        # Majority vote
        if predicted_choices:
            choice_counts = {}
            for choice in predicted_choices:
                if choice:
                    choice_counts[choice] = choice_counts.get(choice, 0) + 1
            majority_choice = (
                max(choice_counts.items(), key=lambda x: x[1])[0]
                if choice_counts
                else None
            )
        else:
            majority_choice = None

        correct_count = sum(
            1 for analysis in response_analysis if analysis["is_correct"]
        )

        return {
            "correct_answer": correct_letter,
            "response_analysis": response_analysis,
            "predicted_choices": predicted_choices,
            "is_consistent": is_consistent,
            "majority_choice": majority_choice,
            "accuracy": correct_count / len(responses) if responses else 0.0,
            "consistency_score": (
                1.0
                if is_consistent
                else len(choice_counts) / len(responses) if choice_counts else 0.0
            ),
        }

    def _extract_choice(self, response: str) -> Optional[str]:
        """Extract choice (A, B, C, D) from response."""
        # Look for explicit choice indicators
        patterns = [
            r"\b([ABCD])\b",
            r"(?:choice|answer|option)\s*:?\s*([ABCD])",
            r"^([ABCD])[.)]",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[0].upper()

        return None

    def _extract_confidence_indicators(self, response: str) -> List[str]:
        """Extract words indicating confidence level."""
        confidence_words = [
            "certain",
            "definitely",
            "clearly",
            "obviously",
            "sure",
            "probably",
            "likely",
            "possibly",
            "maybe",
            "uncertain",
        ]

        response_lower = response.lower()
        found_indicators = []

        for word in confidence_words:
            if word in response_lower:
                found_indicators.append(word)

        return found_indicators


class MMLUAdapter(MultiResponseBenchmarkAdapter):
    """Adapter for MMLU (Massive Multitask Language Understanding) benchmark."""

    def __init__(
        self, config: Optional[MultiResponseBenchmarkConfig] = None, provider=None
    ):
        if config is None:
            config = MultiResponseBenchmarkConfig(
                num_responses_per_sample=3,
                temperature_range=(0.1, 0.5),  # Lower for factual consistency
                reasoning_trace_enabled=True,
            )
        super().__init__("MMLU", config, provider)

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert MMLU sample to PropositionSet."""
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        subject = sample.get("subject", "")

        # Create propositions from question and choices
        props = [Proposition(text=question)]

        for i, choice in enumerate(choices):
            choice_text = f"Option {chr(65+i)}: {choice}"
            props.append(Proposition(text=choice_text))

        context = f"Multiple choice question in {subject}"
        return PropositionSet(propositions=props, context=context)

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format MMLU sample as domain-specific multiple choice prompt."""
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        subject = sample.get("subject", "unknown")

        prompt = f"""Answer the following {subject} question:

Question: {question}

Options:
"""

        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"

        prompt += "\nProvide your answer as a single letter (A, B, C, or D) and explain your reasoning."

        return prompt

    def evaluate_responses(
        self, sample: Dict[str, Any], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate MMLU responses for domain knowledge consistency."""
        correct_answer_idx = sample.get("answer", 0)
        correct_letter = chr(65 + correct_answer_idx)
        subject = sample.get("subject", "unknown")

        response_analysis = []
        predicted_choices = []

        for i, response in enumerate(responses):
            predicted_choice = self._extract_choice_from_response(response)
            is_correct = predicted_choice == correct_letter

            predicted_choices.append(predicted_choice)

            # Analyze reasoning quality
            reasoning_quality = self._assess_reasoning_quality(response, subject)

            analysis = {
                "response_index": i,
                "predicted_choice": predicted_choice,
                "is_correct": is_correct,
                "reasoning_quality": reasoning_quality,
                "mentions_subject_concepts": self._mentions_subject_concepts(
                    response, subject
                ),
            }
            response_analysis.append(analysis)

        # Cross-domain consistency analysis
        consistency_analysis = self._analyze_cross_domain_consistency(
            predicted_choices, responses
        )

        correct_count = sum(
            1 for analysis in response_analysis if analysis["is_correct"]
        )

        return {
            "subject": subject,
            "correct_answer": correct_letter,
            "response_analysis": response_analysis,
            "consistency_analysis": consistency_analysis,
            "accuracy": correct_count / len(responses) if responses else 0.0,
            "subject_coherence": self._compute_subject_coherence(responses, subject),
        }

    def _extract_choice_from_response(self, response: str) -> Optional[str]:
        """Extract choice from MMLU response."""
        # Similar to HellaSwag but more lenient
        patterns = [
            r"(?:answer|choice|option)\s*:?\s*([ABCD])",
            r"\b([ABCD])\b",
            r"^([ABCD])[.)]",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[0].upper()

        return None

    def _assess_reasoning_quality(self, response: str, subject: str) -> Dict[str, Any]:
        """Assess quality of reasoning in response."""
        reasoning_indicators = {
            "science": ["because", "due to", "therefore", "as a result", "evidence"],
            "math": ["calculate", "solve", "equation", "formula", "proof"],
            "history": ["during", "period", "era", "caused by", "led to"],
            "literature": ["theme", "author", "style", "symbolism", "character"],
        }

        response_lower = response.lower()

        # Get subject-specific indicators
        subject_key = (
            subject.lower() if subject.lower() in reasoning_indicators else "science"
        )
        indicators = reasoning_indicators.get(
            subject_key, reasoning_indicators["science"]
        )

        indicator_count = sum(
            1 for indicator in indicators if indicator in response_lower
        )

        return {
            "indicator_count": indicator_count,
            "has_reasoning": indicator_count > 0,
            "response_length": len(response),
            "reasoning_density": (
                indicator_count / len(response.split()) if response else 0
            ),
        }

    def _mentions_subject_concepts(self, response: str, subject: str) -> bool:
        """Check if response mentions subject-specific concepts."""
        concept_maps = {
            "biology": ["cell", "organism", "evolution", "gene", "protein"],
            "chemistry": ["molecule", "atom", "reaction", "bond", "element"],
            "physics": ["force", "energy", "mass", "velocity", "field"],
            "mathematics": ["function", "derivative", "integral", "theorem", "proof"],
            "history": ["century", "war", "empire", "revolution", "period"],
            "literature": ["novel", "poem", "author", "character", "theme"],
        }

        subject_concepts = concept_maps.get(subject.lower(), [])
        response_lower = response.lower()

        return any(concept in response_lower for concept in subject_concepts)

    def _analyze_cross_domain_consistency(
        self, choices: List[str], responses: List[str]
    ) -> Dict[str, Any]:
        """Analyze consistency of reasoning across domain knowledge."""
        unique_choices = set(choice for choice in choices if choice)

        # Check if reasoning approaches are similar across responses
        reasoning_similarity = self._compute_reasoning_similarity(responses)

        return {
            "choice_consistency": len(unique_choices) <= 1,
            "num_unique_choices": len(unique_choices),
            "reasoning_similarity": reasoning_similarity,
            "high_confidence": reasoning_similarity > 0.7,
        }

    def _compute_reasoning_similarity(self, responses: List[str]) -> float:
        """Compute similarity in reasoning approaches across responses."""
        if len(responses) <= 1:
            return 1.0

        # Simple similarity based on shared reasoning words
        reasoning_words = [
            "because",
            "since",
            "therefore",
            "thus",
            "however",
            "although",
            "evidence",
            "shows",
            "indicates",
            "suggests",
            "implies",
        ]

        response_word_sets = []
        for response in responses:
            response_lower = response.lower()
            words_present = set(
                word for word in reasoning_words if word in response_lower
            )
            response_word_sets.append(words_present)

        # Compute pairwise Jaccard similarities
        similarities = []
        for i in range(len(response_word_sets)):
            for j in range(i + 1, len(response_word_sets)):
                set1, set2 = response_word_sets[i], response_word_sets[j]
                if not set1 and not set2:
                    similarity = 1.0
                elif not set1 or not set2:
                    similarity = 0.0
                else:
                    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                similarities.append(similarity)

        import numpy as np
        return float(np.mean(similarities)) if similarities else 1.0

    def _compute_subject_coherence(self, responses: List[str], subject: str) -> float:
        """Compute coherence score specific to the subject domain."""
        # This is a simplified coherence measure based on domain-specific vocabulary
        domain_vocab = {
            "biology": [
                "biological",
                "organism",
                "cellular",
                "genetic",
                "evolutionary",
            ],
            "chemistry": ["chemical", "molecular", "atomic", "compound", "reaction"],
            "physics": [
                "physical",
                "mechanical",
                "electromagnetic",
                "quantum",
                "thermal",
            ],
            "mathematics": [
                "mathematical",
                "numerical",
                "algebraic",
                "geometric",
                "statistical",
            ],
        }

        subject_words = domain_vocab.get(subject.lower(), [])
        if not subject_words:
            return 0.5  # Neutral coherence for unknown subjects

        coherence_scores = []
        for response in responses:
            response_lower = response.lower()
            word_count = sum(1 for word in subject_words if word in response_lower)
            # Normalize by response length
            normalized_score = word_count / len(response.split()) if response else 0
            coherence_scores.append(
                min(1.0, normalized_score * 10)
            )  # Scale up and cap at 1.0

        import numpy as np
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
