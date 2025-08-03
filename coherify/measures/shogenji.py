"""
Traditional Shogenji coherence measure with proper probability estimation.
Implements the classical philosophical coherence measure: C_S(S) = P(H1 ∧ H2 ∧ ... ∧ Hn) / ∏P(Hi)
"""

import time
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from coherify.core.base import (
    CoherenceMeasure, CoherenceResult, PropositionSet, 
    Proposition, ProbabilityEstimator
)
from coherify.utils.caching import cached_computation, get_default_computation_cache


class ModelBasedProbabilityEstimator(ProbabilityEstimator):
    """
    Probability estimator using language model likelihood.
    
    This approach uses a language model to estimate P(proposition | context)
    by computing the likelihood of the proposition text given the context.
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 max_length: int = 512,
                 use_cache: bool = True):
        """
        Initialize model-based probability estimator.
        
        Args:
            model_name: HuggingFace model name for probability estimation
            max_length: Maximum sequence length for model input
            use_cache: Whether to cache probability computations
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_cache = use_cache
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def estimate_probability(self, proposition: Proposition, 
                           context: Optional[str] = None) -> float:
        """Estimate probability of a single proposition given context."""
        if self.use_cache:
            return self._cached_estimate_probability(
                proposition.text, context or ""
            )
        else:
            return self._compute_probability(proposition.text, context or "")
    
    @cached_computation()
    def _cached_estimate_probability(self, text: str, context: str) -> float:
        """Cached version of probability estimation."""
        return self._compute_probability(text, context)
    
    def _compute_probability(self, text: str, context: str) -> float:
        """Compute probability using model likelihood."""
        try:
            # Prepare input
            if context:
                full_text = f"{context} {text}"
            else:
                full_text = text
            
            # Tokenize
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get model likelihood
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Convert loss to probability (higher loss = lower probability)
                # Use negative log likelihood approximation
                probability = torch.exp(-loss).item()
                
                # Clamp to reasonable range
                probability = max(min(probability, 0.999), 0.001)
                
            return probability
            
        except Exception as e:
            # Fallback to neutral probability
            print(f"Warning: Probability estimation failed: {e}")
            return 0.5
    
    def estimate_joint_probability(self, propositions: List[Proposition],
                                 context: Optional[str] = None) -> float:
        """Estimate joint probability of multiple propositions."""
        if len(propositions) == 0:
            return 1.0
        elif len(propositions) == 1:
            return self.estimate_probability(propositions[0], context)
        
        # For joint probability, concatenate propositions and compute likelihood
        combined_text = " ".join([p.text for p in propositions])
        
        if self.use_cache:
            return self._cached_estimate_joint_probability(combined_text, context or "")
        else:
            return self._compute_probability(combined_text, context or "")
    
    @cached_computation()
    def _cached_estimate_joint_probability(self, combined_text: str, context: str) -> float:
        """Cached version of joint probability estimation."""
        return self._compute_probability(combined_text, context)


class ConfidenceBasedProbabilityEstimator(ProbabilityEstimator):
    """
    Probability estimator using model confidence scores.
    
    This approach uses a classification model to estimate the confidence
    that each proposition is true given the context.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_cache: bool = True):
        """
        Initialize confidence-based estimator.
        
        Args:
            model_name: Model for confidence estimation
            use_cache: Whether to cache computations
        """
        self.model_name = model_name
        self.use_cache = use_cache
        
        # Initialize NLI pipeline for confidence estimation
        try:
            from coherify.utils.transformers_utils import create_pipeline_with_suppressed_warnings
            self.classifier = create_pipeline_with_suppressed_warnings(
                "text-classification",
                "facebook/bart-large-mnli",
                return_all_scores=True
            )
        except:
            # Fallback to a simpler model
            self.classifier = create_pipeline_with_suppressed_warnings(
                "text-classification",
                "cross-encoder/nli-deberta-v3-base",
                return_all_scores=True
            )
    
    def estimate_probability(self, proposition: Proposition, 
                           context: Optional[str] = None) -> float:
        """Estimate probability using NLI confidence."""
        if self.use_cache:
            return self._cached_estimate_probability(
                proposition.text, context or ""
            )
        else:
            return self._compute_confidence(proposition.text, context or "")
    
    @cached_computation()
    def _cached_estimate_probability(self, text: str, context: str) -> float:
        """Cached version of probability estimation."""
        return self._compute_confidence(text, context)
    
    def _compute_confidence(self, text: str, context: str) -> float:
        """Compute confidence using NLI model."""
        try:
            if context:
                # Format as entailment question
                input_text = f"{context} </s></s> {text}"
            else:
                # Use text alone
                input_text = text
            
            # Get classification scores
            from coherify.utils.transformers_utils import safe_pipeline_call
            results = safe_pipeline_call(self.classifier, input_text)
            
            # Extract entailment/positive confidence
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    scores = results[0]
                else:
                    scores = results
                
                # Look for entailment or positive labels
                for score_dict in scores:
                    label = score_dict.get('label', '').upper()
                    if 'ENTAILMENT' in label or 'POSITIVE' in label or label == '0':
                        return score_dict.get('score', 0.5)
                
                # If no specific label found, use highest score
                if scores:
                    return max(scores, key=lambda x: x.get('score', 0))['score']
            
            return 0.5
            
        except Exception as e:
            print(f"Warning: Confidence estimation failed: {e}")
            return 0.5
    
    def estimate_joint_probability(self, propositions: List[Proposition],
                                 context: Optional[str] = None) -> float:
        """Estimate joint probability as product of individual probabilities."""
        if len(propositions) == 0:
            return 1.0
        
        # Simple independence assumption for joint probability
        individual_probs = [
            self.estimate_probability(prop, context) 
            for prop in propositions
        ]
        
        # Use geometric mean to avoid overly small probabilities
        return np.exp(np.mean(np.log(individual_probs)))


class EnsembleProbabilityEstimator(ProbabilityEstimator):
    """
    Ensemble probability estimator combining multiple approaches.
    """
    
    def __init__(self, 
                 estimators: Optional[List[ProbabilityEstimator]] = None,
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble estimator.
        
        Args:
            estimators: List of probability estimators to combine
            weights: Weights for each estimator (default: equal weights)
        """
        if estimators is None:
            # Default ensemble
            estimators = [
                ConfidenceBasedProbabilityEstimator(),
                # Note: ModelBasedProbabilityEstimator is expensive, disabled by default
            ]
        
        self.estimators = estimators
        
        if weights is None:
            weights = [1.0 / len(estimators)] * len(estimators)
        
        if len(weights) != len(estimators):
            raise ValueError("Number of weights must match number of estimators")
        
        self.weights = weights
    
    def estimate_probability(self, proposition: Proposition, 
                           context: Optional[str] = None) -> float:
        """Estimate probability as weighted average of ensemble."""
        probabilities = []
        
        for estimator in self.estimators:
            try:
                prob = estimator.estimate_probability(proposition, context)
                probabilities.append(prob)
            except Exception as e:
                print(f"Warning: Estimator failed: {e}")
                probabilities.append(0.5)  # Neutral fallback
        
        # Weighted average
        weighted_prob = sum(p * w for p, w in zip(probabilities, self.weights))
        return max(min(weighted_prob, 0.999), 0.001)
    
    def estimate_joint_probability(self, propositions: List[Proposition],
                                 context: Optional[str] = None) -> float:
        """Estimate joint probability as weighted average of ensemble."""
        probabilities = []
        
        for estimator in self.estimators:
            try:
                prob = estimator.estimate_joint_probability(propositions, context)
                probabilities.append(prob)
            except Exception as e:
                print(f"Warning: Estimator failed: {e}")
                probabilities.append(0.5)
        
        # Weighted average
        weighted_prob = sum(p * w for p, w in zip(probabilities, self.weights))
        return max(min(weighted_prob, 0.999), 0.001)


class ShogunjiCoherence(CoherenceMeasure):
    """
    Traditional Shogenji coherence measure: C_S(S) = P(H1 ∧ H2 ∧ ... ∧ Hn) / ∏P(Hi)
    
    This implementation provides proper probability estimation using language models
    to compute the classical philosophical coherence measure.
    """
    
    def __init__(self, 
                 probability_estimator: Optional[ProbabilityEstimator] = None,
                 smoothing: float = 1e-6,
                 use_log_space: bool = True):
        """
        Initialize Shogenji coherence measure.
        
        Args:
            probability_estimator: Method for estimating probabilities
            smoothing: Small value to avoid division by zero
            use_log_space: Whether to compute in log space for numerical stability
        """
        if probability_estimator is None:
            # Default to ensemble estimator for robustness
            probability_estimator = EnsembleProbabilityEstimator()
        
        self.estimator = probability_estimator
        self.smoothing = smoothing
        self.use_log_space = use_log_space
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute traditional Shogenji coherence."""
        start_time = time.time()
        
        if len(prop_set) < 2:
            return CoherenceResult(
                score=1.0,  # Single proposition is perfectly coherent
                measure_name="ShogunjiCoherence",
                details={"num_propositions": len(prop_set), "reason": "insufficient_propositions"},
                computation_time=time.time() - start_time
            )
        
        # Estimate individual probabilities
        individual_probs = []
        for prop in prop_set.propositions:
            if prop.probability is None:
                prop.probability = self.estimator.estimate_probability(
                    prop, context=prop_set.context
                )
            
            # Apply smoothing
            prob = max(prop.probability + self.smoothing, self.smoothing)
            individual_probs.append(prob)
        
        # Estimate joint probability
        joint_prob = self.estimator.estimate_joint_probability(
            prop_set.propositions, context=prop_set.context
        )
        joint_prob = max(joint_prob + self.smoothing, self.smoothing)
        
        # Compute Shogenji coherence score
        if self.use_log_space:
            # Compute in log space for numerical stability
            log_joint = np.log(joint_prob)
            log_product_individual = np.sum(np.log(individual_probs))
            log_score = log_joint - log_product_individual
            score = np.exp(log_score)
        else:
            # Direct computation
            product_individual = np.prod(individual_probs)
            score = joint_prob / product_individual
        
        # Additional analysis
        independence_baseline = np.prod(individual_probs)
        coherence_gain = joint_prob / independence_baseline
        
        computation_time = time.time() - start_time
        
        return CoherenceResult(
            score=score,
            measure_name="ShogunjiCoherence",
            details={
                "joint_probability": joint_prob,
                "individual_probabilities": individual_probs,
                "product_individual": independence_baseline,
                "coherence_gain": coherence_gain,
                "log_space_computation": self.use_log_space,
                "smoothing": self.smoothing,
                "num_propositions": len(prop_set.propositions),
                "interpretation": self._interpret_score(score)
            },
            computation_time=computation_time
        )
    
    def compute_pairwise(self, prop1: Proposition, prop2: Proposition) -> float:
        """Compute Shogenji coherence for a pair of propositions."""
        prop_set = PropositionSet(propositions=[prop1, prop2])
        result = self.compute(prop_set)
        return result.score
    
    def _interpret_score(self, score: float) -> str:
        """Provide interpretation of Shogenji coherence score."""
        if score > 1.2:
            return "Strong positive coherence - propositions strongly support each other"
        elif score > 1.05:
            return "Moderate positive coherence - propositions somewhat support each other"
        elif score > 0.95:
            return "Near independence - propositions neither support nor contradict"
        elif score > 0.8:
            return "Moderate negative coherence - propositions somewhat contradict each other"
        else:
            return "Strong negative coherence - propositions strongly contradict each other"