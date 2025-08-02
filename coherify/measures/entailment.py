"""
Entailment-based coherence measure using Natural Language Inference models.
This approach evaluates logical relationships between propositions.
"""

import time
from typing import Optional, Dict, List, Tuple
import numpy as np

from coherify.core.base import CoherenceMeasure, CoherenceResult, PropositionSet, Proposition
from coherify.core.types import NLIModel


class EntailmentCoherence(CoherenceMeasure):
    """
    Entailment-based coherence using NLI models.
    
    Evaluates coherence by checking logical relationships (entailment/contradiction)
    between propositions. High coherence means many entailments and few contradictions.
    """
    
    def __init__(self, 
                 nli_model: Optional[NLIModel] = None,
                 entailment_weight: float = 1.0,
                 contradiction_weight: float = -2.0,
                 neutral_weight: float = 0.0):
        """
        Initialize entailment coherence measure.
        
        Args:
            nli_model: NLI model for entailment prediction. If None, uses default.
            entailment_weight: Weight for entailment relationships (positive)
            contradiction_weight: Weight for contradiction relationships (negative)
            neutral_weight: Weight for neutral relationships
        """
        self.nli_model = nli_model or self._get_default_nli_model()
        self.entailment_weight = entailment_weight
        self.contradiction_weight = contradiction_weight
        self.neutral_weight = neutral_weight
    
    def _get_default_nli_model(self) -> NLIModel:
        """Get default NLI model using transformers."""
        try:
            from transformers import pipeline
            
            # Use a lightweight, fast NLI model
            model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Fallback model
                return_all_scores=True
            )
            
            # Try to use a proper NLI model
            try:
                model = pipeline(
                    "text-classification",
                    model="facebook/bart-large-mnli",
                    return_all_scores=True
                )
            except:
                # Fallback to a smaller model
                model = pipeline(
                    "text-classification", 
                    model="cross-encoder/nli-deberta-v3-base",
                    return_all_scores=True
                )
            
            return HuggingFaceNLIWrapper(model)
            
        except ImportError:
            raise ImportError(
                "transformers is required for default NLI model. "
                "Install with: pip install transformers"
            )
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute entailment-based coherence for a proposition set."""
        start_time = time.time()
        
        if len(prop_set) < 2:
            return CoherenceResult(
                score=1.0,  # Single proposition is coherent with itself
                measure_name="EntailmentCoherence",
                details={"num_propositions": len(prop_set), "reason": "insufficient_propositions"},
                computation_time=time.time() - start_time
            )
        
        props = prop_set.propositions
        n = len(props)
        
        # Initialize counters
        entailments = 0
        contradictions = 0
        neutrals = 0
        
        # Store detailed results
        pairwise_results = []
        
        # Check all directed pairs for entailment relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    premise = props[i].text
                    hypothesis = props[j].text
                    
                    # Get entailment prediction
                    label = self.nli_model.predict(premise, hypothesis)
                    pairwise_results.append({
                        "premise_idx": i,
                        "hypothesis_idx": j,
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": label
                    })
                    
                    # Count relationships
                    if label == "entailment":
                        entailments += 1
                    elif label == "contradiction":
                        contradictions += 1
                    else:  # neutral
                        neutrals += 1
        
        # Calculate coherence score
        total_pairs = n * (n - 1)  # All directed pairs
        
        if total_pairs == 0:
            score = 1.0
        else:
            # Weighted score based on relationship types
            weighted_score = (
                entailments * self.entailment_weight +
                contradictions * self.contradiction_weight +
                neutrals * self.neutral_weight
            )
            
            # Normalize to reasonable range
            max_possible_score = total_pairs * self.entailment_weight
            min_possible_score = total_pairs * self.contradiction_weight
            
            if max_possible_score == min_possible_score:
                score = 0.5
            else:
                # Scale to [0, 1] range
                score = (weighted_score - min_possible_score) / (max_possible_score - min_possible_score)
        
        computation_time = time.time() - start_time
        
        return CoherenceResult(
            score=score,
            measure_name="EntailmentCoherence",
            details={
                "entailments": entailments,
                "contradictions": contradictions,
                "neutrals": neutrals,
                "total_pairs": total_pairs,
                "entailment_rate": entailments / total_pairs if total_pairs > 0 else 0,
                "contradiction_rate": contradictions / total_pairs if total_pairs > 0 else 0,
                "neutral_rate": neutrals / total_pairs if total_pairs > 0 else 0,
                "weights": {
                    "entailment": self.entailment_weight,
                    "contradiction": self.contradiction_weight,
                    "neutral": self.neutral_weight
                },
                "pairwise_results": pairwise_results,
                "num_propositions": n
            },
            computation_time=computation_time
        )
    
    def compute_pairwise(self, prop1: Proposition, prop2: Proposition) -> float:
        """Compute entailment score between two propositions."""
        # Check both directions
        label1 = self.nli_model.predict(prop1.text, prop2.text)
        label2 = self.nli_model.predict(prop2.text, prop1.text)
        
        # Score based on relationship types
        score = 0.0
        for label in [label1, label2]:
            if label == "entailment":
                score += self.entailment_weight
            elif label == "contradiction":
                score += self.contradiction_weight
            else:  # neutral
                score += self.neutral_weight
        
        # Normalize to [0, 1] range
        max_score = 2 * self.entailment_weight
        min_score = 2 * self.contradiction_weight
        
        if max_score == min_score:
            return 0.5
        
        return (score - min_score) / (max_score - min_score)


class HuggingFaceNLIWrapper:
    """Wrapper for HuggingFace NLI models to match NLIModel protocol."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
        # Map model-specific labels to standard labels
        self.label_mapping = {
            # Standard NLI labels
            "ENTAILMENT": "entailment",
            "CONTRADICTION": "contradiction", 
            "NEUTRAL": "neutral",
            # MNLI labels
            "entailment": "entailment",
            "contradiction": "contradiction",
            "neutral": "neutral",
            # Some models use numbers
            "0": "entailment",
            "1": "neutral", 
            "2": "contradiction"
        }
    
    def predict(self, premise: str, hypothesis: str) -> str:
        """Predict entailment relationship between premise and hypothesis."""
        # Format input for NLI model
        if hasattr(self.pipeline.model.config, 'model_type') and 'bart' in self.pipeline.model.config.model_type:
            # BART-based models expect specific format
            input_text = f"{premise} </s></s> {hypothesis}"
        else:
            # Standard format for most NLI models
            input_text = f"{premise} [SEP] {hypothesis}"
        
        try:
            # Get prediction
            result = self.pipeline(input_text)
            
            # Handle different output formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Multiple scores returned
                    best_result = max(result[0], key=lambda x: x['score'])
                else:
                    # Single result
                    best_result = result[0]
            else:
                best_result = result
            
            # Extract label
            label = best_result['label'].upper()
            
            # Map to standard labels
            return self.label_mapping.get(label, "neutral")
            
        except Exception as e:
            # Fallback to neutral if prediction fails
            print(f"Warning: NLI prediction failed: {e}")
            return "neutral"


class SimpleNLIModel:
    """Simple rule-based NLI model for testing/fallback."""
    
    def predict(self, premise: str, hypothesis: str) -> str:
        """Simple heuristic-based entailment prediction."""
        premise = premise.lower()
        hypothesis = hypothesis.lower()
        
        # Simple keyword-based heuristics
        if premise == hypothesis:
            return "entailment"
        
        # Check for obvious contradictions
        contradiction_pairs = [
            ("yes", "no"), ("true", "false"), ("good", "bad"),
            ("hot", "cold"), ("big", "small"), ("up", "down")
        ]
        
        for word1, word2 in contradiction_pairs:
            if (word1 in premise and word2 in hypothesis) or (word2 in premise and word1 in hypothesis):
                return "contradiction"
        
        # Check for potential entailment (hypothesis is subset of premise concepts)
        hypothesis_words = set(hypothesis.split())
        premise_words = set(premise.split())
        
        if hypothesis_words.issubset(premise_words):
            return "entailment"
        
        return "neutral"