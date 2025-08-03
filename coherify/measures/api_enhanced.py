"""
API-enhanced coherence measures using external providers.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from ..core.base import CoherenceMeasure, CoherenceResult, PropositionSet
from ..providers.manager import get_provider, ModelProvider
from ..providers.base import ModelResponse
from .semantic import SemanticCoherence
from .entailment import EntailmentCoherence
from .hybrid import HybridCoherence


@dataclass
class APICoherenceConfig:
    """Configuration for API-enhanced coherence measures."""
    provider_name: Optional[str] = None
    model_name: Optional[str] = None
    use_temperature_variance: bool = True
    temperature_range: List[float] = None
    variance_weight: float = 0.3
    enable_reasoning_trace: bool = True
    
    def __post_init__(self):
        if self.temperature_range is None:
            self.temperature_range = [0.3, 0.7, 1.0]


class APIEnhancedSemanticCoherence(SemanticCoherence):
    """Semantic coherence enhanced with external API providers."""
    
    def __init__(
        self,
        config: Optional[APICoherenceConfig] = None,
        provider: Optional[ModelProvider] = None,
        **kwargs
    ):
        """
        Initialize API-enhanced semantic coherence.
        
        Args:
            config: API configuration
            provider: External model provider
            **kwargs: Additional arguments for base SemanticCoherence
        """
        super().__init__(**kwargs)
        self.config = config or APICoherenceConfig()
        self.provider = provider or get_provider(self.config.provider_name)
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute coherence using API-enhanced semantic analysis."""
        # Get base semantic coherence
        base_result = super().compute(prop_set)
        
        # Enhance with API-based analysis
        api_enhancement = self._compute_api_enhancement(prop_set)
        
        # Combine results
        enhanced_score = self._combine_scores(base_result.score, api_enhancement)
        
        return CoherenceResult(
            score=enhanced_score,
            measure_name="APIEnhancedSemantic",
            details={
                **base_result.details,
                "api_enhancement": api_enhancement,
                "base_score": base_result.score,
                "provider": self.provider.provider_name,
                "model": self.provider.model_name
            }
        )
    
    def _compute_api_enhancement(self, prop_set: PropositionSet) -> Dict[str, Any]:
        """Compute API-based coherence enhancement."""
        propositions = [p.text for p in prop_set.propositions]
        
        enhancement_data = {
            "embeddings_coherence": 0.0,
            "temperature_variance": 0.0,
            "reasoning_analysis": None
        }
        
        try:
            # Enhanced embeddings using API provider
            embeddings = []
            for prop_text in propositions:
                embedding = self.provider.embed_text(prop_text)
                embeddings.append(embedding)
            
            if embeddings:
                # Compute enhanced semantic coherence
                enhancement_data["embeddings_coherence"] = self._compute_embeddings_coherence(embeddings)
            
            # Temperature variance analysis
            if self.config.use_temperature_variance:
                enhancement_data["temperature_variance"] = self._compute_temperature_variance(prop_set)
            
            # Reasoning analysis for supported models
            if self.config.enable_reasoning_trace:
                enhancement_data["reasoning_analysis"] = self._compute_reasoning_analysis(prop_set)
                
        except Exception as e:
            print(f"Warning: API enhancement failed: {e}")
        
        return enhancement_data
    
    def _compute_embeddings_coherence(self, embeddings: List[List[float]]) -> float:
        """Compute coherence from API-generated embeddings."""
        if len(embeddings) < 2:
            return 1.0
        
        embeddings_array = np.array(embeddings)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings_array)):
            for j in range(i + 1, len(embeddings_array)):
                sim = np.dot(embeddings_array[i], embeddings_array[j]) / (
                    np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
                )
                similarities.append(max(0, sim))  # Ensure non-negative
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_temperature_variance(self, prop_set: PropositionSet) -> float:
        """Compute coherence based on temperature variance."""
        if not hasattr(self.provider, 'generate_with_temperatures'):
            return 0.0
        
        # Create a summary prompt from propositions
        props_text = " ".join([p.text for p in prop_set.propositions])
        prompt = f"Summarize the key ideas in this text concisely: {props_text}"
        
        try:
            # Generate responses with different temperatures
            responses = self.provider.generate_with_temperatures(
                prompt=prompt,
                temperatures=self.config.temperature_range,
                max_tokens=100
            )
            
            if len(responses) < 2:
                return 0.0
            
            # Analyze variance in responses
            response_texts = [r.text for r in responses]
            variance_score = self._analyze_response_variance(response_texts)
            
            # Lower variance indicates higher coherence
            return 1.0 - min(1.0, variance_score)
            
        except Exception as e:
            print(f"Warning: Temperature variance analysis failed: {e}")
            return 0.0
    
    def _analyze_response_variance(self, responses: List[str]) -> float:
        """Analyze variance in model responses."""
        if len(responses) < 2:
            return 0.0
        
        # Simple variance measure based on response diversity
        unique_responses = set(responses)
        diversity_ratio = len(unique_responses) / len(responses)
        
        # Compute length variance
        lengths = [len(r.split()) for r in responses]
        length_variance = np.var(lengths) / (np.mean(lengths) + 1e-6)
        
        # Combine measures
        return (diversity_ratio + length_variance) / 2.0
    
    def _compute_reasoning_analysis(self, prop_set: PropositionSet) -> Optional[Dict[str, Any]]:
        """Compute reasoning-based coherence analysis."""
        if not self.provider.model_name or not self.provider.model_name.startswith("o"):
            return None
        
        props_text = " ".join([p.text for p in prop_set.propositions])
        prompt = f"""
Analyze the logical coherence of these statements. Consider:
1. Internal consistency
2. Logical flow between ideas
3. Factual consistency
4. Overall coherence

Statements: {props_text}

Provide a coherence score (0.0-1.0) and brief reasoning.
"""
        
        try:
            response = self.provider.generate_text(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            # Extract reasoning trace if available
            reasoning_trace = response.metadata.get("reasoning_trace")
            
            # Try to extract score from response
            score = self._extract_score_from_text(response.text)
            
            return {
                "reasoning_score": score,
                "reasoning_text": response.text,
                "reasoning_trace": reasoning_trace,
                "has_trace": reasoning_trace is not None
            }
            
        except Exception as e:
            print(f"Warning: Reasoning analysis failed: {e}")
            return None
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numerical score from text response."""
        import re
        
        # Look for patterns like "0.8", "score: 0.7", etc.
        patterns = [
            r'score[:\s]+([0-1]?\.\d+)',
            r'coherence[:\s]+([0-1]?\.\d+)',
            r'([0-1]?\.\d+)/1\.0',
            r'([0-1]?\.\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        
        return 0.5  # Default moderate score
    
    def _combine_scores(self, base_score: float, enhancement_data: Dict[str, Any]) -> float:
        """Combine base coherence score with API enhancements."""
        enhanced_score = base_score
        
        # Factor in embeddings coherence
        embeddings_score = enhancement_data.get("embeddings_coherence", 0.0)
        if embeddings_score > 0:
            enhanced_score = 0.7 * enhanced_score + 0.3 * embeddings_score
        
        # Factor in temperature variance
        if self.config.use_temperature_variance:
            variance_score = enhancement_data.get("temperature_variance", 0.0)
            if variance_score > 0:
                enhanced_score = (1 - self.config.variance_weight) * enhanced_score + \
                               self.config.variance_weight * variance_score
        
        # Factor in reasoning analysis
        reasoning_data = enhancement_data.get("reasoning_analysis")
        if reasoning_data and reasoning_data.get("reasoning_score") is not None:
            reasoning_score = reasoning_data["reasoning_score"]
            enhanced_score = 0.8 * enhanced_score + 0.2 * reasoning_score
        
        return max(0.0, min(1.0, enhanced_score))


class APIEnhancedEntailmentCoherence(EntailmentCoherence):
    """Entailment coherence enhanced with external API providers."""
    
    def __init__(
        self,
        config: Optional[APICoherenceConfig] = None,
        provider: Optional[ModelProvider] = None,
        **kwargs
    ):
        """Initialize API-enhanced entailment coherence."""
        super().__init__(**kwargs)
        self.config = config or APICoherenceConfig()
        self.provider = provider or get_provider(self.config.provider_name)
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute coherence using API-enhanced entailment analysis."""
        # Get base entailment coherence
        base_result = super().compute(prop_set)
        
        # Enhance with API-based entailment analysis
        api_enhancement = self._compute_api_entailment_enhancement(prop_set)
        
        # Combine results
        enhanced_score = self._combine_entailment_scores(base_result.score, api_enhancement)
        
        return CoherenceResult(
            score=enhanced_score,
            measure_name="APIEnhancedEntailment",
            details={
                **base_result.details,
                "api_enhancement": api_enhancement,
                "base_score": base_result.score,
                "provider": self.provider.provider_name
            }
        )
    
    def _compute_api_entailment_enhancement(self, prop_set: PropositionSet) -> Dict[str, Any]:
        """Compute API-based entailment enhancement."""
        propositions = [p.text for p in prop_set.propositions]
        
        if len(propositions) < 2:
            return {"api_entailment_scores": [], "average_entailment": 1.0}
        
        entailment_scores = []
        
        try:
            # Compute pairwise entailment using API
            for i in range(len(propositions)):
                for j in range(i + 1, len(propositions)):
                    entailment_result = self.provider.classify_entailment(
                        premise=propositions[i],
                        hypothesis=propositions[j]
                    )
                    
                    # Convert entailment classification to coherence score
                    coherence_score = self._entailment_to_coherence(entailment_result)
                    entailment_scores.append(coherence_score)
                    
        except Exception as e:
            print(f"Warning: API entailment analysis failed: {e}")
        
        return {
            "api_entailment_scores": entailment_scores,
            "average_entailment": np.mean(entailment_scores) if entailment_scores else 0.5
        }
    
    def _entailment_to_coherence(self, entailment_result: Dict[str, float]) -> float:
        """Convert entailment classification to coherence score."""
        # High entailment or neutral = coherent
        # High contradiction = incoherent
        entailment_score = entailment_result.get("entailment", 0.0)
        neutral_score = entailment_result.get("neutral", 0.0)
        contradiction_score = entailment_result.get("contradiction", 0.0)
        
        # Coherence is high when there's entailment or neutral relationship
        # and low when there's contradiction
        coherence = (entailment_score + 0.7 * neutral_score) * (1 - contradiction_score)
        return max(0.0, min(1.0, coherence))
    
    def _combine_entailment_scores(self, base_score: float, api_enhancement: Dict[str, Any]) -> float:
        """Combine base and API entailment scores."""
        api_score = api_enhancement.get("average_entailment", 0.5)
        
        # Weight combination: 60% base, 40% API
        return 0.6 * base_score + 0.4 * api_score


class APIEnhancedHybridCoherence(HybridCoherence):
    """Hybrid coherence enhanced with external API providers."""
    
    def __init__(
        self,
        config: Optional[APICoherenceConfig] = None,
        provider: Optional[ModelProvider] = None,
        semantic_weight: float = 0.5,
        entailment_weight: float = 0.5,
        **kwargs
    ):
        """Initialize API-enhanced hybrid coherence."""
        self.config = config or APICoherenceConfig()
        self.provider = provider or get_provider(self.config.provider_name)
        
        # Initialize enhanced component measures
        semantic_measure = APIEnhancedSemanticCoherence(
            config=self.config,
            provider=self.provider
        )
        entailment_measure = APIEnhancedEntailmentCoherence(
            config=self.config,
            provider=self.provider
        )
        
        super().__init__(
            semantic_measure=semantic_measure,
            entailment_measure=entailment_measure,
            semantic_weight=semantic_weight,
            entailment_weight=entailment_weight,
            **kwargs
        )
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute API-enhanced hybrid coherence."""
        result = super().compute(prop_set)
        result.measure_name = "APIEnhancedHybrid"
        result.details["provider"] = self.provider.provider_name
        result.details["model"] = self.provider.model_name
        result.details["api_config"] = {
            "use_temperature_variance": self.config.use_temperature_variance,
            "temperature_range": self.config.temperature_range,
            "variance_weight": self.config.variance_weight,
            "enable_reasoning_trace": self.config.enable_reasoning_trace
        }
        
        return result