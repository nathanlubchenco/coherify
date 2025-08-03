"""
FEVER Benchmark Adapter

Adapter for FEVER (Fact Extraction and VERification) benchmark with 
evidence-based coherence evaluation for fact-checking tasks.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import re
import json

from coherify.core.base import PropositionSet, Proposition
from coherify.benchmarks.multi_format_adapters import MultiResponseBenchmarkAdapter, MultiResponseBenchmarkConfig
from coherify.measures.multi_response import MultiResponseCoherenceMeasure, MultiResponseConfig


@dataclass
class FEVERConfig(MultiResponseBenchmarkConfig):
    """Configuration for FEVER benchmark evaluation."""
    max_evidence_sentences: int = 5
    enable_evidence_retrieval: bool = False  # Enable actual Wikipedia retrieval
    use_provided_evidence: bool = True       # Use evidence from dataset
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
    
    def __init__(self, 
                 config: Optional[FEVERConfig] = None,
                 provider = None):
        if config is None:
            config = FEVERConfig(
                enable_multi_response=True,
                num_responses_per_sample=3,
                temperature_range=(0.1, 0.6),  # Lower for factual tasks
                reasoning_trace_enabled=True
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
            metadata={"type": "claim", "label": fever_sample.label}
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
                        "sources": evidence_set.sources
                    }
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
                "requires_multi_hop": self._requires_multi_hop_reasoning(fever_sample)
            }
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
    
    def evaluate_responses(self, 
                         sample: Dict[str, Any], 
                         responses: List[str]) -> Dict[str, Any]:
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
                "response_length": len(response)
            }
            response_analysis.append(analysis)
            
            # Score evidence consistency (how well response uses provided evidence)
            evidence_consistency_scores.append(evidence_analysis.get("consistency_score", 0.0))
        
        # Check label consistency across responses
        unique_labels = set(label for label in predicted_labels if label)
        is_label_consistent = len(unique_labels) <= 1
        
        # Majority vote
        if predicted_labels:
            label_counts = {}
            for label in predicted_labels:
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
            majority_label = max(label_counts.items(), key=lambda x: x[1])[0] if label_counts else None
        else:
            majority_label = None
        
        # Compute overall scores
        correct_count = sum(1 for analysis in response_analysis if analysis["is_correct"])
        accuracy = correct_count / len(responses) if responses else 0.0
        avg_evidence_consistency = sum(evidence_consistency_scores) / len(evidence_consistency_scores) if evidence_consistency_scores else 0.0
        
        return {
            "ground_truth_label": ground_truth_label,
            "response_analysis": response_analysis,
            "predicted_labels": predicted_labels,
            "is_label_consistent": is_label_consistent,
            "majority_label": majority_label,
            "accuracy": accuracy,
            "evidence_consistency": avg_evidence_consistency,
            "fever_score": accuracy * avg_evidence_consistency,  # Simplified FEVER-like score
            "requires_multi_hop": self._requires_multi_hop_reasoning(fever_sample),
            "evidence_coherence_analysis": self._analyze_evidence_coherence(fever_sample, responses)
        }
    
    def _parse_fever_sample(self, sample: Dict[str, Any]) -> FEVERSample:
        """Parse raw FEVER sample into structured format."""
        claim_id = str(sample.get("id", "unknown"))
        claim = sample.get("claim", "")
        label = sample.get("label", "NOT ENOUGH INFO")
        
        # Parse evidence
        evidence_sets = []
        evidence_data = sample.get("evidence", [])
        
        if evidence_data and label != "NOT ENOUGH INFO":
            for evidence_group in evidence_data:
                if evidence_group:  # Skip empty evidence groups
                    evidence_sentences = []
                    sources = []
                    annotation_id = None
                    
                    for evidence_item in evidence_group:
                        if len(evidence_item) >= 4:
                            ann_id, ev_id, page, sent_id = evidence_item[:4]
                            
                            if annotation_id is None:
                                annotation_id = str(ann_id) if ann_id is not None else "unknown"
                            
                            # In a real implementation, would retrieve actual sentence from Wikipedia
                            # For now, create a mock evidence sentence
                            if page and sent_id is not None:
                                evidence_sentence = f"[Evidence from {page}] Sentence {sent_id}: Mock evidence text related to the claim."
                                evidence_sentences.append(evidence_sentence)
                                sources.append({
                                    "page": page,
                                    "sentence_id": sent_id,
                                    "evidence_id": ev_id
                                })
                    
                    if evidence_sentences:
                        evidence_set = EvidenceSet(
                            claim_id=claim_id,
                            evidence_sentences=evidence_sentences,
                            sources=sources,
                            annotation_id=annotation_id
                        )
                        evidence_sets.append(evidence_set)
        
        # Handle NOT ENOUGH INFO case or missing evidence
        if not evidence_sets:
            evidence_sets = [EvidenceSet(
                claim_id=claim_id,
                evidence_sentences=["No sufficient evidence available."],
                sources=[{"page": "N/A", "sentence_id": "N/A"}],
                annotation_id="no_evidence"
            )]
        
        return FEVERSample(
            claim_id=claim_id,
            claim=claim,
            label=label,
            evidence_sets=evidence_sets,
            original_data=sample
        )
    
    def _extract_fever_label(self, response: str) -> Optional[str]:
        """Extract FEVER label from response."""
        response_upper = response.upper()
        
        # Look for explicit labels
        patterns = [
            r'\b(SUPPORTS|REFUTES|NOT ENOUGH INFO)\b',
            r'(?:VERDICT|ANSWER|CLASSIFICATION):\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)',
            r'(?:CLAIM IS|CLAIM:)\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                return matches[-1]  # Return last match
        
        # Fuzzy matching for common variations
        if "SUPPORT" in response_upper and "NOT" not in response_upper:
            return "SUPPORTS"
        elif "REFUTE" in response_upper or "CONTRADICT" in response_upper or "FALSE" in response_upper:
            return "REFUTES"
        elif "NOT ENOUGH" in response_upper or "INSUFFICIENT" in response_upper or "UNCLEAR" in response_upper:
            return "NOT ENOUGH INFO"
        
        return None
    
    def _analyze_evidence_usage(self, response: str, fever_sample: FEVERSample) -> Dict[str, Any]:
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
                    evidence_references.append({
                        "evidence_index": i,
                        "overlap_words": overlap,
                        "evidence_set_id": evidence_set.annotation_id
                    })
        
        # Calculate consistency score
        total_evidence = sum(len(es.evidence_sentences) for es in fever_sample.evidence_sets)
        consistency_score = evidence_mentions / max(total_evidence, 1)
        
        return {
            "evidence_mentions": evidence_mentions,
            "total_evidence": total_evidence,
            "consistency_score": min(1.0, consistency_score),
            "evidence_references": evidence_references,
            "uses_evidence": evidence_mentions > 0
        }
    
    def _assess_reasoning_quality(self, response: str, fever_sample: FEVERSample) -> Dict[str, Any]:
        """Assess quality of reasoning in fact-checking response."""
        reasoning_indicators = [
            "because", "since", "therefore", "thus", "however", "although",
            "evidence", "shows", "indicates", "suggests", "proves", "contradicts",
            "according to", "based on", "given that", "considering"
        ]
        
        response_lower = response.lower()
        
        # Count reasoning indicators
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        
        # Check for step-by-step reasoning
        has_structured_reasoning = any(pattern in response_lower for pattern in [
            "first", "second", "next", "then", "finally",
            "step 1", "step 2", "analysis:",
            "evidence analysis", "reasoning:"
        ])
        
        # Check for evidence evaluation
        has_evidence_evaluation = any(pattern in response_lower for pattern in [
            "evidence shows", "evidence indicates", "according to evidence",
            "evidence supports", "evidence contradicts", "evidence suggests"
        ])
        
        return {
            "reasoning_indicator_count": indicator_count,
            "has_structured_reasoning": has_structured_reasoning,
            "has_evidence_evaluation": has_evidence_evaluation,
            "reasoning_density": indicator_count / len(response.split()) if response else 0,
            "reasoning_quality_score": min(1.0, (indicator_count + 
                                                int(has_structured_reasoning) * 2 + 
                                                int(has_evidence_evaluation) * 2) / 10)
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
    
    def _analyze_evidence_coherence(self, 
                                  fever_sample: FEVERSample, 
                                  responses: List[str]) -> Dict[str, Any]:
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
                    count = sum(1 for pattern in evidence_usage_patterns 
                              if any(ref["evidence_index"] == idx for ref in pattern))
                    reference_counts[idx] = count
                
                # Consistency is average reference rate
                avg_reference_rate = sum(reference_counts.values()) / (len(reference_counts) * len(responses))
                consistency_score = avg_reference_rate
        
        return {
            "coherence_score": consistency_score,
            "evidence_usage_patterns": evidence_usage_patterns,
            "total_unique_evidence_referenced": len(all_referenced_indices) if 'all_referenced_indices' in locals() else 0,
            "analysis": f"Evidence coherence across {len(responses)} responses"
        }


class EvidenceBasedCoherence(MultiResponseCoherenceMeasure):
    """Coherence measure specifically designed for evidence-based fact-checking."""
    
    def __init__(self, 
                 config: Optional[MultiResponseConfig] = None,
                 provider = None):
        if config is None:
            from coherify.measures.hybrid import HybridCoherence
            base_measure = HybridCoherence()
            config = MultiResponseConfig(
                num_responses=3,
                temperature_range=(0.1, 0.5),  # Lower for factual consistency
                consistency_threshold=0.8
            )
        else:
            from coherify.measures.hybrid import HybridCoherence
            base_measure = HybridCoherence()
        
        super().__init__(base_measure, config, provider)
    
    def evaluate_evidence_coherence(self, 
                                  claim: str,
                                  evidence_sentences: List[str],
                                  context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate coherence between claim and evidence, and across evidence sentences.
        
        This is the core functionality for FEVER-style fact-checking coherence.
        """
        # Create proposition sets
        claim_props = [Proposition(text=claim, metadata={"type": "claim"})]
        evidence_props = [Proposition(text=sent, metadata={"type": "evidence"}) 
                         for sent in evidence_sentences]
        
        # Evaluate claim-evidence coherence
        claim_evidence_props = claim_props + evidence_props
        claim_evidence_set = PropositionSet(
            propositions=claim_evidence_props,
            context=context or "Fact-checking evaluation"
        )
        
        claim_evidence_result = self.base_measure.compute(claim_evidence_set)
        
        # Evaluate evidence-only coherence (internal consistency)
        if len(evidence_props) > 1:
            evidence_only_set = PropositionSet(
                propositions=evidence_props,
                context="Evidence consistency evaluation"
            )
            evidence_consistency_result = self.base_measure.compute(evidence_only_set)
            evidence_consistency = evidence_consistency_result.score
        else:
            evidence_consistency = 1.0  # Single evidence sentence is trivially consistent
        
        # Combined coherence score
        overall_coherence = (claim_evidence_result.score * 0.7 + evidence_consistency * 0.3)
        
        return {
            "claim_evidence_coherence": claim_evidence_result.score,
            "evidence_consistency": evidence_consistency,
            "overall_coherence": overall_coherence,
            "claim": claim,
            "num_evidence_sentences": len(evidence_sentences),
            "coherence_verdict": "coherent" if overall_coherence > self.config.consistency_threshold else "incoherent"
        }
    
    def evaluate_multi_response_fact_checking(self,
                                            claim: str,
                                            evidence_sentences: List[str],
                                            context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate multiple fact-checking responses and evaluate their coherence.
        
        This combines multi-response generation with evidence-based evaluation.
        """
        if not self.provider:
            return {
                "error": "Provider required for multi-response evaluation",
                "single_evaluation": self.evaluate_evidence_coherence(claim, evidence_sentences, context)
            }
        
        # Format as fact-checking prompt
        evidence_text = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(evidence_sentences))
        prompt = f"""Fact-checking task: Verify the following claim against the evidence.

Claim: {claim}

Evidence:
{evidence_text}

Classify as SUPPORTS, REFUTES, or NOT ENOUGH INFO and explain your reasoning."""
        
        # Generate multiple responses
        multi_result = self.compute_multi_response(prompt, context)
        
        # Evaluate evidence coherence
        evidence_coherence = self.evaluate_evidence_coherence(claim, evidence_sentences, context)
        
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
        verdict_consistency = 1.0 if len(unique_verdicts) <= 1 else len(unique_verdicts) / len(verdicts)
        
        return {
            "multi_response_result": multi_result,
            "evidence_coherence": evidence_coherence,
            "fact_checking_verdicts": verdicts,
            "verdict_consistency": verdict_consistency,
            "is_verdict_consistent": len(unique_verdicts) <= 1,
            "majority_verdict": max(set(verdicts), key=verdicts.count) if verdicts else "UNCLEAR",
            "overall_confidence": multi_result.confidence_score * evidence_coherence["overall_coherence"]
        }