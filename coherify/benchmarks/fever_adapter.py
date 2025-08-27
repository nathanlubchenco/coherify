"""
FEVER Benchmark Adapter

Adapter for FEVER (Fact Extraction and VERification) benchmark with
evidence-based coherence evaluation for fact-checking tasks.
"""

import re
from dataclasses import dataclass
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


@dataclass
class FEVERConfig(MultiResponseBenchmarkConfig):
    """Configuration for FEVER benchmark evaluation."""

    max_evidence_sentences: int = 5
    enable_evidence_retrieval: bool = False  # Enable actual Wikipedia retrieval
    use_provided_evidence: bool = True  # Use evidence from dataset
    evidence_coherence_weight: float = 0.6
    claim_coherence_weight: float = 0.4
    enable_multi_hop_reasoning: bool = True
    cross_document_analysis: bool = True


@dataclass
class EvidenceSet:
    """Represents a set of evidence sentences for a claim."""

    claim_id: str
    evidence_sentences: List[str]
    sources: List[Dict[str, Any]]  # Wikipedia page info
    annotation_id: Optional[str] = None
    evidence_score: Optional[float] = None
    evidence_type: str = "single"  # "single", "multi_sentence", "multi_page"
    requires_composition: bool = (
        False  # True if evidence must be composed across sentences
    )


@dataclass
class FEVERSample:
    """Processed FEVER sample with evidence and claim."""

    claim_id: str
    claim: str
    label: str  # SUPPORTS, REFUTES, NOT ENOUGH INFO
    evidence_sets: List[EvidenceSet]
    original_data: Dict[str, Any]


class FEVERAdapter(MultiResponseBenchmarkAdapter):
    """Adapter for FEVER fact-checking benchmark with evidence-based coherence."""

    def __init__(self, config: Optional[FEVERConfig] = None, provider=None):
        if config is None:
            config = FEVERConfig(
                enable_multi_response=True,
                num_responses_per_sample=3,
                temperature_range=(0.1, 0.6),  # Lower for factual tasks
                reasoning_trace_enabled=True,
            )
        super().__init__("FEVER", config, provider)

        # Evidence retrieval setup (placeholder for now)
        self.evidence_retriever = None
        if config.enable_evidence_retrieval:
            # Would initialize actual Wikipedia retrieval here
            pass

    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert FEVER sample to PropositionSet with evidence."""
        fever_sample = self._parse_fever_sample(sample)

        # Create propositions from claim and evidence
        props = []

        # Add claim as primary proposition
        claim_prop = Proposition(
            text=fever_sample.claim,
            metadata={"type": "claim", "label": fever_sample.label},
        )
        props.append(claim_prop)

        # Add evidence sentences as supporting propositions
        for evidence_set in fever_sample.evidence_sets:
            for i, evidence_sentence in enumerate(evidence_set.evidence_sentences):
                evidence_prop = Proposition(
                    text=evidence_sentence,
                    metadata={
                        "type": "evidence",
                        "evidence_set_id": evidence_set.annotation_id or "unknown",
                        "sentence_index": i,
                        "sources": evidence_set.sources,
                    },
                )
                props.append(evidence_prop)

        # Context includes the fact-checking task description
        context = f"Fact-checking task: Verify claim against evidence. Label: {fever_sample.label}"

        return PropositionSet(
            propositions=props,
            context=context,
            metadata={
                "benchmark": "FEVER",
                "claim_id": fever_sample.claim_id,
                "num_evidence_sets": len(fever_sample.evidence_sets),
                "requires_multi_hop": self._requires_multi_hop_reasoning(fever_sample),
            },
        )

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format FEVER sample as fact-checking prompt."""
        fever_sample = self._parse_fever_sample(sample)

        prompt = f"""Fact-checking task: Verify the following claim against the provided evidence.

Claim: {fever_sample.claim}

Evidence:
"""

        # Add evidence from all evidence sets
        evidence_count = 1
        for evidence_set in fever_sample.evidence_sets:
            for evidence_sentence in evidence_set.evidence_sentences:
                prompt += f"{evidence_count}. {evidence_sentence}\n"
                evidence_count += 1

        if self.config.reasoning_trace_enabled:
            prompt += """
Please analyze the evidence step by step and determine if the claim is:
- SUPPORTS: The evidence supports the claim
- REFUTES: The evidence contradicts the claim
- NOT ENOUGH INFO: Insufficient evidence to verify the claim

Provide your reasoning and final verdict."""
        else:
            prompt += """
Based on the evidence, classify the claim as:
SUPPORTS, REFUTES, or NOT ENOUGH INFO

Answer:"""

        return prompt

    def retrieve_evidence_chain(
        self, claim: str, evidence_data: List[List[Any]]
    ) -> Dict[str, Any]:
        """
        Retrieve and analyze multi-sentence, multi-page evidence chains for FEVER claims.

        According to Thorne et al. (2018):
        - 31.75% of claims require multiple sentences as evidence
        - 16.82% require evidence composition across sentences
        - 12.15% require evidence from multiple Wikipedia pages
        """
        if not evidence_data:
            return {
                "single_sentence_evidence": [],
                "multi_sentence_evidence": [],
                "cross_page_evidence": {},
                "evidence_type": "none",
                "requires_composition": False,
                "complexity_analysis": {
                    "level": "none",
                    "reason": "No evidence provided",
                },
            }

        # Analyze evidence structure
        all_evidence_sentences = []
        evidence_by_page = {}
        multi_sentence_groups = []

        for evidence_group in evidence_data:
            if not evidence_group:
                continue

            group_sentences = []
            group_pages = set()

            for evidence_item in evidence_group:
                if len(evidence_item) >= 4:
                    ann_id, ev_id, page, sent_id = evidence_item[:4]

                    if page and sent_id is not None:
                        # In real implementation, would retrieve actual text from Wikipedia
                        # For now, create structured evidence representation
                        evidence_text = self._retrieve_evidence_sentence(
                            page, sent_id, claim
                        )

                        sentence_info = {
                            "text": evidence_text,
                            "page": page,
                            "sentence_id": sent_id,
                            "evidence_id": ev_id,
                            "annotation_id": ann_id,
                        }

                        all_evidence_sentences.append(sentence_info)
                        group_sentences.append(sentence_info)
                        group_pages.add(page)

                        # Group by page
                        if page not in evidence_by_page:
                            evidence_by_page[page] = []
                        evidence_by_page[page].append(sentence_info)

            if len(group_sentences) > 1:
                multi_sentence_groups.append(
                    {
                        "sentences": group_sentences,
                        "pages": list(group_pages),
                        "requires_composition": len(group_sentences) > 1,
                    }
                )

        # Determine evidence complexity
        unique_pages = set(sent["page"] for sent in all_evidence_sentences)
        requires_multi_page = len(unique_pages) > 1
        requires_multi_sentence = any(
            len(group["sentences"]) > 1 for group in multi_sentence_groups
        )
        requires_composition = len(all_evidence_sentences) > 1

        # Classify evidence type
        if requires_multi_page:
            evidence_type = "multi_page"
            complexity_level = "high"
            complexity_reason = (
                f"Requires evidence from {len(unique_pages)} Wikipedia pages"
            )
        elif requires_multi_sentence:
            evidence_type = "multi_sentence"
            complexity_level = "medium"
            complexity_reason = (
                f"Requires composition across {len(all_evidence_sentences)} sentences"
            )
        elif len(all_evidence_sentences) == 1:
            evidence_type = "single"
            complexity_level = "low"
            complexity_reason = "Single sentence evidence"
        else:
            evidence_type = "single"
            complexity_level = "low"
            complexity_reason = "Basic single-sentence verification"

        return {
            "single_sentence_evidence": [
                sent["text"]
                for sent in all_evidence_sentences
                if len(all_evidence_sentences) == 1
            ],
            "multi_sentence_evidence": [
                [sent["text"] for sent in group["sentences"]]
                for group in multi_sentence_groups
            ],
            "cross_page_evidence": {
                page: [sent["text"] for sent in sentences]
                for page, sentences in evidence_by_page.items()
            },
            "evidence_type": evidence_type,
            "requires_composition": requires_composition,
            "complexity_analysis": {
                "level": complexity_level,
                "reason": complexity_reason,
                "num_sentences": len(all_evidence_sentences),
                "num_pages": len(unique_pages),
                "requires_multi_page": requires_multi_page,
                "requires_multi_sentence": requires_multi_sentence,
            },
            "structured_evidence": {
                "all_sentences": all_evidence_sentences,
                "by_page": evidence_by_page,
                "multi_sentence_groups": multi_sentence_groups,
            },
        }

    def _retrieve_evidence_sentence(
        self, page: str, sentence_id: int, claim: str
    ) -> str:
        """
        Retrieve actual evidence sentence from Wikipedia page.

        In a full implementation, this would:
        1. Query Wikipedia API for the page
        2. Extract sentence by ID
        3. Return the actual evidence text

        For now, creates contextually relevant mock evidence.
        """
        # Mock evidence based on page name and claim context
        if isinstance(sentence_id, int) and sentence_id >= 0:
            # Create more realistic evidence that could plausibly support/refute claims
            evidence_templates = [
                f"According to the Wikipedia article on {page}, sentence {sentence_id} states relevant information about the topic discussed in the claim.",
                f"The {page} article (sentence {sentence_id}) provides factual information that can be used to verify the claim.",
                f"From {page}: This sentence contains specific details that either support or contradict the stated claim.",
                f"Wikipedia's {page} page (sentence {sentence_id}) offers authoritative information on this subject matter.",
            ]

            # Select template based on hash of page name for consistency
            template_index = hash(page) % len(evidence_templates)
            base_text = evidence_templates[template_index]

            # Add more specific mock content based on page name
            if any(
                term in page.lower()
                for term in ["person", "people", "actor", "writer", "politician"]
            ):
                base_text = f"According to {page}'s Wikipedia page, biographical information in sentence {sentence_id} indicates relevant facts about this person."
            elif any(
                term in page.lower()
                for term in ["place", "city", "country", "location"]
            ):
                base_text = f"The Wikipedia article about {page} contains geographical information in sentence {sentence_id} that relates to the claim."
            elif any(
                term in page.lower() for term in ["event", "war", "battle", "year"]
            ):
                base_text = f"Historical information from the {page} Wikipedia page (sentence {sentence_id}) provides context for verifying this claim."

            return base_text
        else:
            return (
                f"Evidence from {page}: Information relevant to the claim verification."
            )

    def evaluate_with_evidence_composition(
        self, claim: str, evidence_chain: Dict[str, Any], responses: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate claims requiring evidence composition from multiple sources.

        This method specifically handles the complex FEVER cases that require:
        - Multi-sentence reasoning
        - Cross-page evidence synthesis
        - Compositional evidence evaluation
        """
        complexity_analysis = evidence_chain.get("complexity_analysis", {})
        evidence_type = evidence_chain.get("evidence_type", "single")

        # Base evaluation metrics
        evaluation = {
            "claim": claim,
            "evidence_type": evidence_type,
            "complexity_level": complexity_analysis.get("level", "unknown"),
            "requires_composition": evidence_chain.get("requires_composition", False),
            "num_evidence_sentences": complexity_analysis.get("num_sentences", 0),
            "num_pages": complexity_analysis.get("num_pages", 0),
        }

        # Evaluate compositional complexity
        if evidence_type == "multi_page":
            # Cross-document coherence analysis
            cross_page_evidence = evidence_chain.get("cross_page_evidence", {})
            page_coherence_scores = []

            pages = list(cross_page_evidence.keys())
            for i, page1 in enumerate(pages):
                for page2 in pages[i + 1 :]:
                    # Simple coherence check between evidence from different pages
                    evidence1 = " ".join(cross_page_evidence[page1])
                    evidence2 = " ".join(cross_page_evidence[page2])

                    coherence_score = self._calculate_evidence_coherence(
                        evidence1, evidence2
                    )
                    page_coherence_scores.append(coherence_score)

            evaluation["cross_page_coherence"] = {
                "mean_coherence": (
                    sum(page_coherence_scores) / len(page_coherence_scores)
                    if page_coherence_scores
                    else 0.0
                ),
                "coherence_scores": page_coherence_scores,
                "pages_analyzed": len(pages),
            }

        elif evidence_type == "multi_sentence":
            # Multi-sentence composition analysis
            multi_sentence_groups = evidence_chain.get("structured_evidence", {}).get(
                "multi_sentence_groups", []
            )
            composition_scores = []

            for group in multi_sentence_groups:
                sentences = [sent["text"] for sent in group["sentences"]]
                if len(sentences) > 1:
                    # Check if sentences form coherent evidence chain
                    chain_coherence = self._evaluate_evidence_chain_coherence(sentences)
                    composition_scores.append(chain_coherence)

            evaluation["multi_sentence_composition"] = {
                "mean_composition_score": (
                    sum(composition_scores) / len(composition_scores)
                    if composition_scores
                    else 0.0
                ),
                "composition_scores": composition_scores,
                "num_groups": len(multi_sentence_groups),
            }

        # If responses provided, evaluate how well they handle evidence composition
        if responses:
            response_evaluations = []

            for i, response in enumerate(responses):
                resp_eval = {
                    "response_index": i,
                    "handles_composition": self._response_handles_composition(
                        response, evidence_chain
                    ),
                    "evidence_integration_score": self._score_evidence_integration(
                        response, evidence_chain
                    ),
                    "reasoning_complexity": self._assess_compositional_reasoning(
                        response, evidence_type
                    ),
                }
                response_evaluations.append(resp_eval)

            evaluation["response_analysis"] = response_evaluations
            evaluation["mean_integration_score"] = (
                sum(resp["evidence_integration_score"] for resp in response_evaluations)
                / len(response_evaluations)
                if response_evaluations
                else 0.0
            )

        # Overall composition effectiveness score
        composition_effectiveness = self._calculate_composition_effectiveness(
            evaluation
        )
        evaluation["composition_effectiveness"] = composition_effectiveness

        return evaluation

    def _calculate_evidence_coherence(self, evidence1: str, evidence2: str) -> float:
        """Calculate coherence between two evidence pieces."""
        if not evidence1 or not evidence2:
            return 0.0

        # Simple word overlap metric (in real implementation would use embeddings)
        words1 = set(evidence1.lower().split())
        words2 = set(evidence2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _evaluate_evidence_chain_coherence(self, sentences: List[str]) -> float:
        """Evaluate coherence across a chain of evidence sentences."""
        if len(sentences) <= 1:
            return 1.0

        coherence_scores = []
        for i in range(len(sentences) - 1):
            score = self._calculate_evidence_coherence(sentences[i], sentences[i + 1])
            coherence_scores.append(score)

        return (
            sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        )

    def _response_handles_composition(
        self, response: str, evidence_chain: Dict[str, Any]
    ) -> bool:
        """Check if response properly handles evidence composition."""
        evidence_type = evidence_chain.get("evidence_type", "single")
        response_lower = response.lower()

        composition_indicators = [
            "multiple",
            "several",
            "various",
            "different",
            "across",
            "combining",
            "together",
            "both",
            "all",
            "each",
            "first",
            "second",
            "also",
            "furthermore",
            "additionally",
            "from different",
            "multiple sources",
            "various sources",
        ]

        has_composition_language = any(
            indicator in response_lower for indicator in composition_indicators
        )

        if evidence_type == "multi_page":
            # Check for cross-page reasoning
            cross_page_indicators = [
                "page",
                "article",
                "source",
                "according to",
                "from",
            ]
            has_cross_page_reasoning = (
                sum(
                    1
                    for indicator in cross_page_indicators
                    if indicator in response_lower
                )
                >= 2
            )
            return has_composition_language and has_cross_page_reasoning
        elif evidence_type == "multi_sentence":
            # Check for multi-sentence synthesis
            return has_composition_language and len(response.split(".")) >= 2
        else:
            return True  # Single evidence doesn't require composition

    def _score_evidence_integration(
        self, response: str, evidence_chain: Dict[str, Any]
    ) -> float:
        """Score how well response integrates evidence from the chain."""
        evidence_sentences = []

        # Collect all evidence sentences
        if "structured_evidence" in evidence_chain:
            all_sentences = evidence_chain["structured_evidence"].get(
                "all_sentences", []
            )
            evidence_sentences = [sent["text"] for sent in all_sentences]

        if not evidence_sentences:
            return 0.0

        # Count evidence references in response
        evidence_mentions = 0
        response_words = set(response.lower().split())

        for evidence in evidence_sentences:
            evidence_words = set(evidence.lower().split())
            # Filter common words
            common_words = {
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            }
            evidence_words_filtered = evidence_words - common_words

            if evidence_words_filtered:
                overlap = len(evidence_words_filtered.intersection(response_words))
                if overlap >= 2:  # At least 2 meaningful words overlap
                    evidence_mentions += 1

        return (
            evidence_mentions / len(evidence_sentences) if evidence_sentences else 0.0
        )

    def _assess_compositional_reasoning(
        self, response: str, evidence_type: str
    ) -> Dict[str, Any]:
        """Assess the complexity of compositional reasoning in response."""
        response_lower = response.lower()

        reasoning_patterns = {
            "sequential": ["first", "then", "next", "finally", "subsequently"],
            "comparative": [
                "however",
                "but",
                "while",
                "whereas",
                "although",
                "despite",
            ],
            "causal": [
                "because",
                "since",
                "therefore",
                "thus",
                "as a result",
                "consequently",
            ],
            "aggregative": [
                "overall",
                "in total",
                "combined",
                "together",
                "collectively",
            ],
            "synthetic": [
                "integrating",
                "combining",
                "synthesizing",
                "considering all",
            ],
        }

        pattern_counts = {}
        for pattern_type, patterns in reasoning_patterns.items():
            count = sum(1 for pattern in patterns if pattern in response_lower)
            pattern_counts[pattern_type] = count

        total_patterns = sum(pattern_counts.values())
        complexity_score = min(total_patterns / 5, 1.0)  # Normalize to 0-1

        expected_complexity = {
            "single": 0.2,
            "multi_sentence": 0.5,
            "multi_page": 0.8,
        }.get(evidence_type, 0.5)

        meets_expected_complexity = complexity_score >= expected_complexity

        return {
            "pattern_counts": pattern_counts,
            "complexity_score": complexity_score,
            "expected_complexity": expected_complexity,
            "meets_expected_complexity": meets_expected_complexity,
            "reasoning_quality": (
                "high"
                if complexity_score >= 0.7
                else "medium" if complexity_score >= 0.4 else "low"
            ),
        }

    def _calculate_composition_effectiveness(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall effectiveness of evidence composition handling."""
        effectiveness_factors = []

        # Cross-page coherence factor
        if "cross_page_coherence" in evaluation:
            effectiveness_factors.append(
                evaluation["cross_page_coherence"]["mean_coherence"]
            )

        # Multi-sentence composition factor
        if "multi_sentence_composition" in evaluation:
            effectiveness_factors.append(
                evaluation["multi_sentence_composition"]["mean_composition_score"]
            )

        # Response integration factor
        if "mean_integration_score" in evaluation:
            effectiveness_factors.append(evaluation["mean_integration_score"])

        # Reasoning complexity factor
        if "response_analysis" in evaluation:
            complexity_scores = [
                resp["reasoning_complexity"]["complexity_score"]
                for resp in evaluation["response_analysis"]
            ]
            if complexity_scores:
                effectiveness_factors.append(
                    sum(complexity_scores) / len(complexity_scores)
                )

        return (
            sum(effectiveness_factors) / len(effectiveness_factors)
            if effectiveness_factors
            else 0.0
        )

    def evaluate_responses(
        self, sample: Dict[str, Any], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate FEVER responses for fact-checking consistency."""
        fever_sample = self._parse_fever_sample(sample)
        ground_truth_label = fever_sample.label

        response_analysis = []
        predicted_labels = []
        evidence_consistency_scores = []

        for i, response in enumerate(responses):
            # Extract predicted label
            predicted_label = self._extract_fever_label(response)
            is_correct = predicted_label == ground_truth_label

            predicted_labels.append(predicted_label)

            # Analyze evidence usage and reasoning quality
            evidence_analysis = self._analyze_evidence_usage(response, fever_sample)
            reasoning_quality = self._assess_reasoning_quality(response, fever_sample)

            analysis = {
                "response_index": i,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "evidence_analysis": evidence_analysis,
                "reasoning_quality": reasoning_quality,
                "response_length": len(response),
            }
            response_analysis.append(analysis)

            # Score evidence consistency (how well response uses provided evidence)
            evidence_consistency_scores.append(
                evidence_analysis.get("consistency_score", 0.0)
            )

        # Check label consistency across responses
        unique_labels = set(label for label in predicted_labels if label)
        is_label_consistent = len(unique_labels) <= 1

        # Majority vote
        if predicted_labels:
            label_counts = {}
            for label in predicted_labels:
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
            majority_label = (
                max(label_counts.items(), key=lambda x: x[1])[0]
                if label_counts
                else None
            )
        else:
            majority_label = None

        # Compute overall scores
        correct_count = sum(
            1 for analysis in response_analysis if analysis["is_correct"]
        )
        accuracy = correct_count / len(responses) if responses else 0.0
        avg_evidence_consistency = (
            sum(evidence_consistency_scores) / len(evidence_consistency_scores)
            if evidence_consistency_scores
            else 0.0
        )

        return {
            "ground_truth_label": ground_truth_label,
            "response_analysis": response_analysis,
            "predicted_labels": predicted_labels,
            "is_label_consistent": is_label_consistent,
            "majority_label": majority_label,
            "accuracy": accuracy,
            "evidence_consistency": avg_evidence_consistency,
            "fever_score": accuracy
            * avg_evidence_consistency,  # Simplified FEVER-like score
            "requires_multi_hop": self._requires_multi_hop_reasoning(fever_sample),
            "evidence_coherence_analysis": self._analyze_evidence_coherence(
                fever_sample, responses
            ),
        }

    def _parse_fever_sample(self, sample: Dict[str, Any]) -> FEVERSample:
        """Parse raw FEVER sample into structured format with enhanced evidence chain analysis."""
        claim_id = str(sample.get("id", "unknown"))
        claim = sample.get("claim", "")
        label = sample.get("label", "NOT ENOUGH INFO")

        # Parse evidence using enhanced evidence chain retrieval
        evidence_sets = []
        evidence_data = sample.get("evidence", [])

        if evidence_data and label != "NOT ENOUGH INFO":
            # Use enhanced evidence chain analysis
            self.retrieve_evidence_chain(claim, evidence_data)

            # Convert the structured evidence chain back to EvidenceSet objects
            for evidence_group in evidence_data:
                if evidence_group:  # Skip empty evidence groups
                    evidence_sentences = []
                    sources = []
                    annotation_id = None

                    for evidence_item in evidence_group:
                        if len(evidence_item) >= 4:
                            ann_id, ev_id, page, sent_id = evidence_item[:4]

                            if annotation_id is None:
                                annotation_id = (
                                    str(ann_id) if ann_id is not None else "unknown"
                                )

                            # Use enhanced evidence retrieval
                            if page and sent_id is not None:
                                evidence_sentence = self._retrieve_evidence_sentence(
                                    page, sent_id, claim
                                )
                                evidence_sentences.append(evidence_sentence)
                                sources.append(
                                    {
                                        "page": page,
                                        "sentence_id": sent_id,
                                        "evidence_id": ev_id,
                                    }
                                )

                    if evidence_sentences:
                        # Determine evidence type and composition requirements
                        unique_pages = set(src["page"] for src in sources)
                        evidence_type = (
                            "multi_page"
                            if len(unique_pages) > 1
                            else (
                                "multi_sentence"
                                if len(evidence_sentences) > 1
                                else "single"
                            )
                        )
                        requires_composition = (
                            len(evidence_sentences) > 1 or len(unique_pages) > 1
                        )

                        evidence_set = EvidenceSet(
                            claim_id=claim_id,
                            evidence_sentences=evidence_sentences,
                            sources=sources,
                            annotation_id=annotation_id,
                            evidence_type=evidence_type,
                            requires_composition=requires_composition,
                        )
                        evidence_sets.append(evidence_set)

        # Handle NOT ENOUGH INFO case or missing evidence
        if not evidence_sets:
            evidence_sets = [
                EvidenceSet(
                    claim_id=claim_id,
                    evidence_sentences=["No sufficient evidence available."],
                    sources=[{"page": "N/A", "sentence_id": "N/A"}],
                    annotation_id="no_evidence",
                    evidence_type="none",
                    requires_composition=False,
                )
            ]

        return FEVERSample(
            claim_id=claim_id,
            claim=claim,
            label=label,
            evidence_sets=evidence_sets,
            original_data=sample,
        )

    def _extract_fever_label(self, response: str) -> Optional[str]:
        """Extract FEVER label from response."""
        response_upper = response.upper()

        # Look for explicit labels
        patterns = [
            r"\b(SUPPORTS|REFUTES|NOT ENOUGH INFO)\b",
            r"(?:VERDICT|ANSWER|CLASSIFICATION):\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)",
            r"(?:CLAIM IS|CLAIM:)\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                return matches[-1]  # Return last match

        # Fuzzy matching for common variations
        if "SUPPORT" in response_upper and "NOT" not in response_upper:
            return "SUPPORTS"
        elif (
            "REFUTE" in response_upper
            or "CONTRADICT" in response_upper
            or "FALSE" in response_upper
        ):
            return "REFUTES"
        elif (
            "NOT ENOUGH" in response_upper
            or "INSUFFICIENT" in response_upper
            or "UNCLEAR" in response_upper
        ):
            return "NOT ENOUGH INFO"

        return None

    def _analyze_evidence_usage(
        self, response: str, fever_sample: FEVERSample
    ) -> Dict[str, Any]:
        """Analyze how well the response uses provided evidence."""
        evidence_mentions = 0
        evidence_references = []

        # Count evidence sentence references
        for evidence_set in fever_sample.evidence_sets:
            for i, evidence_sentence in enumerate(evidence_set.evidence_sentences):
                # Look for key phrases from evidence in response
                evidence_words = set(evidence_sentence.lower().split())
                response_words = set(response.lower().split())

                # Simple overlap metric
                overlap = len(evidence_words.intersection(response_words))
                if overlap > 3:  # Threshold for considering evidence mentioned
                    evidence_mentions += 1
                    evidence_references.append(
                        {
                            "evidence_index": i,
                            "overlap_words": overlap,
                            "evidence_set_id": evidence_set.annotation_id,
                        }
                    )

        # Calculate consistency score
        total_evidence = sum(
            len(es.evidence_sentences) for es in fever_sample.evidence_sets
        )
        consistency_score = evidence_mentions / max(total_evidence, 1)

        return {
            "evidence_mentions": evidence_mentions,
            "total_evidence": total_evidence,
            "consistency_score": min(1.0, consistency_score),
            "evidence_references": evidence_references,
            "uses_evidence": evidence_mentions > 0,
        }

    def _assess_reasoning_quality(
        self, response: str, fever_sample: FEVERSample
    ) -> Dict[str, Any]:
        """Assess quality of reasoning in fact-checking response."""
        reasoning_indicators = [
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
            "proves",
            "contradicts",
            "according to",
            "based on",
            "given that",
            "considering",
        ]

        response_lower = response.lower()

        # Count reasoning indicators
        indicator_count = sum(
            1 for indicator in reasoning_indicators if indicator in response_lower
        )

        # Check for step-by-step reasoning
        has_structured_reasoning = any(
            pattern in response_lower
            for pattern in [
                "first",
                "second",
                "next",
                "then",
                "finally",
                "step 1",
                "step 2",
                "analysis:",
                "evidence analysis",
                "reasoning:",
            ]
        )

        # Check for evidence evaluation
        has_evidence_evaluation = any(
            pattern in response_lower
            for pattern in [
                "evidence shows",
                "evidence indicates",
                "according to evidence",
                "evidence supports",
                "evidence contradicts",
                "evidence suggests",
            ]
        )

        return {
            "reasoning_indicator_count": indicator_count,
            "has_structured_reasoning": has_structured_reasoning,
            "has_evidence_evaluation": has_evidence_evaluation,
            "reasoning_density": (
                indicator_count / len(response.split()) if response else 0
            ),
            "reasoning_quality_score": min(
                1.0,
                (
                    indicator_count
                    + int(has_structured_reasoning) * 2
                    + int(has_evidence_evaluation) * 2
                )
                / 10,
            ),
        }

    def _requires_multi_hop_reasoning(self, fever_sample: FEVERSample) -> bool:
        """Determine if sample requires multi-hop reasoning."""
        # Simple heuristic: multiple evidence sets or multiple sources
        if len(fever_sample.evidence_sets) > 1:
            return True

        # Check if evidence comes from multiple sources
        all_sources = []
        for evidence_set in fever_sample.evidence_sets:
            all_sources.extend([src.get("page", "") for src in evidence_set.sources])

        unique_sources = set(all_sources)
        return len(unique_sources) > 1

    def _analyze_evidence_coherence(
        self, fever_sample: FEVERSample, responses: List[str]
    ) -> Dict[str, Any]:
        """Analyze coherence of evidence usage across responses."""
        if not responses:
            return {"coherence_score": 0.0, "analysis": "No responses to analyze"}

        evidence_usage_patterns = []

        for response in responses:
            usage = self._analyze_evidence_usage(response, fever_sample)
            evidence_usage_patterns.append(usage["evidence_references"])

        # Compute consistency in evidence usage
        if len(evidence_usage_patterns) <= 1:
            consistency_score = 1.0
        else:
            # Simple consistency metric: how often the same evidence is referenced
            all_referenced_indices = set()
            for pattern in evidence_usage_patterns:
                for ref in pattern:
                    all_referenced_indices.add(ref["evidence_index"])

            if not all_referenced_indices:
                consistency_score = 1.0  # No evidence used consistently
            else:
                # Count how many responses reference each piece of evidence
                reference_counts = {}
                for idx in all_referenced_indices:
                    count = sum(
                        1
                        for pattern in evidence_usage_patterns
                        if any(ref["evidence_index"] == idx for ref in pattern)
                    )
                    reference_counts[idx] = count

                # Consistency is average reference rate
                avg_reference_rate = sum(reference_counts.values()) / (
                    len(reference_counts) * len(responses)
                )
                consistency_score = avg_reference_rate

        return {
            "coherence_score": consistency_score,
            "evidence_usage_patterns": evidence_usage_patterns,
            "total_unique_evidence_referenced": (
                len(all_referenced_indices)
                if "all_referenced_indices" in locals()
                else 0
            ),
            "analysis": f"Evidence coherence across {len(responses)} responses",
        }


class EvidenceBasedCoherence(MultiResponseCoherenceMeasure):
    """Coherence measure specifically designed for evidence-based fact-checking."""

    def __init__(self, config: Optional[MultiResponseConfig] = None, provider=None):
        if config is None:
            from coherify.measures.hybrid import HybridCoherence

            base_measure = HybridCoherence()
            config = MultiResponseConfig(
                num_responses=3,
                temperature_range=(0.1, 0.5),  # Lower for factual consistency
                consistency_threshold=0.8,
            )
        else:
            from coherify.measures.hybrid import HybridCoherence

            base_measure = HybridCoherence()

        super().__init__(base_measure, config, provider)

    def evaluate_evidence_coherence(
        self, claim: str, evidence_sentences: List[str], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate coherence between claim and evidence, and across evidence sentences.

        This is the core functionality for FEVER-style fact-checking coherence.
        """
        # Create proposition sets
        claim_props = [Proposition(text=claim, metadata={"type": "claim"})]
        evidence_props = [
            Proposition(text=sent, metadata={"type": "evidence"})
            for sent in evidence_sentences
        ]

        # Evaluate claim-evidence coherence
        claim_evidence_props = claim_props + evidence_props
        claim_evidence_set = PropositionSet(
            propositions=claim_evidence_props,
            context=context or "Fact-checking evaluation",
        )

        claim_evidence_result = self.base_measure.compute(claim_evidence_set)

        # Evaluate evidence-only coherence (internal consistency)
        if len(evidence_props) > 1:
            evidence_only_set = PropositionSet(
                propositions=evidence_props, context="Evidence consistency evaluation"
            )
            evidence_consistency_result = self.base_measure.compute(evidence_only_set)
            evidence_consistency = evidence_consistency_result.score
        else:
            evidence_consistency = (
                1.0  # Single evidence sentence is trivially consistent
            )

        # Combined coherence score
        overall_coherence = (
            claim_evidence_result.score * 0.7 + evidence_consistency * 0.3
        )

        return {
            "claim_evidence_coherence": claim_evidence_result.score,
            "evidence_consistency": evidence_consistency,
            "overall_coherence": overall_coherence,
            "claim": claim,
            "num_evidence_sentences": len(evidence_sentences),
            "coherence_verdict": (
                "coherent"
                if overall_coherence > self.config.consistency_threshold
                else "incoherent"
            ),
        }

    def evaluate_multi_response_fact_checking(
        self, claim: str, evidence_sentences: List[str], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate multiple fact-checking responses and evaluate their coherence.

        This combines multi-response generation with evidence-based evaluation.
        """
        if not self.provider:
            return {
                "error": "Provider required for multi-response evaluation",
                "single_evaluation": self.evaluate_evidence_coherence(
                    claim, evidence_sentences, context
                ),
            }

        # Format as fact-checking prompt
        evidence_text = "\n".join(
            f"{i+1}. {sent}" for i, sent in enumerate(evidence_sentences)
        )
        prompt = f"""Fact-checking task: Verify the following claim against the evidence.

Claim: {claim}

Evidence:
{evidence_text}

Classify as SUPPORTS, REFUTES, or NOT ENOUGH INFO and explain your reasoning."""

        # Generate multiple responses
        multi_result = self.compute_multi_response(prompt, context)

        # Evaluate evidence coherence
        evidence_coherence = self.evaluate_evidence_coherence(
            claim, evidence_sentences, context
        )

        # Extract fact-checking verdicts from responses
        verdicts = []
        for response in multi_result.responses:
            # Simple verdict extraction
            response_upper = response.upper()
            if "SUPPORTS" in response_upper:
                verdicts.append("SUPPORTS")
            elif "REFUTES" in response_upper:
                verdicts.append("REFUTES")
            elif "NOT ENOUGH INFO" in response_upper:
                verdicts.append("NOT ENOUGH INFO")
            else:
                verdicts.append("UNCLEAR")

        # Verdict consistency
        unique_verdicts = set(verdicts)
        verdict_consistency = (
            1.0 if len(unique_verdicts) <= 1 else len(unique_verdicts) / len(verdicts)
        )

        return {
            "multi_response_result": multi_result,
            "evidence_coherence": evidence_coherence,
            "fact_checking_verdicts": verdicts,
            "verdict_consistency": verdict_consistency,
            "is_verdict_consistent": len(unique_verdicts) <= 1,
            "majority_verdict": (
                max(set(verdicts), key=verdicts.count) if verdicts else "UNCLEAR"
            ),
            "overall_confidence": multi_result.confidence_score
            * evidence_coherence["overall_coherence"],
        }
