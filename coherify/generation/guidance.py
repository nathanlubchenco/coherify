"""
Coherence-guided generation strategies and real-time guidance systems.
Provides high-level interfaces for coherence-guided text generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass
import time
from collections import deque

from coherify.core.base import CoherenceMeasure, PropositionSet, Proposition, CoherenceResult
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.hybrid import HybridCoherence
from coherify.generation.beam_search import CoherenceGuidedBeamSearch, BeamSearchResult
from coherify.generation.filtering import CoherenceFilter, MultiStageFilter


@dataclass
class GenerationGuidance:
    """Guidance information for generation process."""
    coherence_score: float
    coherence_trend: str  # "improving", "declining", "stable"
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class GenerationSession:
    """Represents a complete generation session."""
    context: str
    generated_text: str
    guidance_history: List[GenerationGuidance]
    final_coherence: float
    session_time: float
    token_count: int
    metadata: Dict[str, Any]


class CoherenceGuidedGenerator:
    """
    High-level generator that combines beam search and filtering for coherent generation.
    """
    
    def __init__(self,
                 coherence_measure: Optional[CoherenceMeasure] = None,
                 beam_search_config: Optional[Dict[str, Any]] = None,
                 filter_config: Optional[Dict[str, Any]] = None,
                 guidance_enabled: bool = True):
        """
        Initialize coherence-guided generator.
        
        Args:
            coherence_measure: Coherence measure for evaluation
            beam_search_config: Configuration for beam search
            filter_config: Configuration for filtering
            guidance_enabled: Whether to provide real-time guidance
        """
        self.coherence_measure = coherence_measure or HybridCoherence()
        self.guidance_enabled = guidance_enabled
        
        # Initialize beam search
        beam_config = beam_search_config or {}
        self.beam_search = CoherenceGuidedBeamSearch(
            coherence_measure=self.coherence_measure,
            **beam_config
        )
        
        # Initialize filter
        filter_config = filter_config or {}
        self.filter = CoherenceFilter(
            coherence_measure=self.coherence_measure,
            **filter_config
        )
        
        # Session tracking
        self.current_session: Optional[GenerationSession] = None
        self.session_history: List[GenerationSession] = []
    
    def generate(self,
                context: str,
                prompt: str = "",
                max_length: int = 100,
                num_candidates: int = 5,
                return_guidance: bool = False) -> Union[str, Tuple[str, List[GenerationGuidance]]]:
        """
        Generate coherent text using beam search and filtering.
        
        Args:
            context: Context for coherence evaluation
            prompt: Starting prompt
            max_length: Maximum generation length
            num_candidates: Number of candidates to generate
            return_guidance: Whether to return guidance information
            
        Returns:
            Generated text, optionally with guidance history
        """
        start_time = time.time()
        guidance_history = []
        
        # Start session
        if self.guidance_enabled:
            self.current_session = GenerationSession(
                context=context,
                generated_text="",
                guidance_history=[],
                final_coherence=0.0,
                session_time=0.0,
                token_count=0,
                metadata={"prompt": prompt, "max_length": max_length}
            )
        
        # Configure beam search for this generation
        self.beam_search.max_length = max_length
        
        # Create generation function
        def generation_function(current_text: str, beam_size: int) -> List[Tuple[str, float]]:
            return self._mock_generation_function(current_text, beam_size, context)
        
        # Perform beam search
        beam_result = self.beam_search.search(
            context=context,
            generation_function=generation_function,
            prompt=prompt
        )
        
        # Generate multiple candidates
        candidates = []
        for i in range(min(num_candidates, len(beam_result.all_candidates))):
            if i < len(beam_result.all_candidates):
                candidate = beam_result.all_candidates[i]
                candidates.append(candidate.text)
        
        if not candidates:
            candidates = [beam_result.best_candidate.text if beam_result.best_candidate else prompt]
        
        # Filter candidates
        filter_result = self.filter.filter_candidates(context, candidates)
        
        # Select best candidate
        if filter_result.passed_candidates:
            best_text = filter_result.passed_candidates[0]
            final_coherence = filter_result.coherence_scores[0]
        else:
            best_text = candidates[0] if candidates else prompt
            final_coherence = 0.0
        
        # Generate guidance if enabled
        if self.guidance_enabled:
            guidance = self._generate_guidance(context, best_text, beam_result, filter_result)
            guidance_history.append(guidance)
            
            # Complete session
            if self.current_session:
                self.current_session.generated_text = best_text
                self.current_session.guidance_history = guidance_history
                self.current_session.final_coherence = final_coherence
                self.current_session.session_time = time.time() - start_time
                self.current_session.token_count = len(best_text.split())
                
                self.session_history.append(self.current_session)
                self.current_session = None
        
        if return_guidance:
            return best_text, guidance_history
        else:
            return best_text
    
    def generate_iterative(self,
                          context: str,
                          prompt: str = "",
                          max_iterations: int = 5,
                          improvement_threshold: float = 0.05) -> GenerationSession:
        """
        Generate text iteratively, refining based on coherence feedback.
        
        Args:
            context: Context for coherence evaluation
            prompt: Starting prompt
            max_iterations: Maximum refinement iterations
            improvement_threshold: Minimum improvement to continue
            
        Returns:
            Complete generation session with refinement history
        """
        start_time = time.time()
        current_text = prompt
        guidance_history = []
        
        for iteration in range(max_iterations):
            # Generate with current text as base
            generation_result, guidance = self.generate(
                context=context,
                prompt=current_text,
                max_length=50,  # Shorter for iterative refinement
                return_guidance=True
            )
            
            # Evaluate improvement
            if guidance:
                current_guidance = guidance[0]
                guidance_history.extend(guidance)
                
                # Check for improvement
                if iteration > 0:
                    prev_coherence = guidance_history[-2].coherence_score
                    current_coherence = current_guidance.coherence_score
                    improvement = current_coherence - prev_coherence
                    
                    if improvement < improvement_threshold:
                        break  # Converged
                
                current_text = generation_result
            else:
                break
        
        # Create session record
        session = GenerationSession(
            context=context,
            generated_text=current_text,
            guidance_history=guidance_history,
            final_coherence=guidance_history[-1].coherence_score if guidance_history else 0.0,
            session_time=time.time() - start_time,
            token_count=len(current_text.split()),
            metadata={
                "iterations": iteration + 1,
                "improvement_threshold": improvement_threshold,
                "converged": iteration < max_iterations - 1
            }
        )
        
        self.session_history.append(session)
        return session
    
    def _mock_generation_function(self, current_text: str, beam_size: int, context: str) -> List[Tuple[str, float]]:
        """Mock generation function for demonstration."""
        # Context-aware word selection
        context_words = {
            "machine learning": ["algorithm", "data", "model", "training", "prediction", "accuracy"],
            "artificial intelligence": ["neural", "network", "learning", "intelligent", "system", "autonomous"],
            "data science": ["analysis", "visualization", "statistics", "insights", "patterns", "correlation"],
            "technology": ["innovation", "digital", "software", "development", "solution", "platform"]
        }
        
        # Select relevant words based on context
        relevant_words = []
        context_lower = context.lower()
        for topic, words in context_words.items():
            if any(word in context_lower for word in topic.split()):
                relevant_words.extend(words)
        
        # Default word bank
        if not relevant_words:
            relevant_words = ["system", "process", "method", "approach", "solution", "result"]
        
        # Add common words
        common_words = [".", "and", "the", "is", "can", "will", "with", "for", "that", "this"]
        all_words = relevant_words + common_words
        
        # Generate candidates with log probabilities
        import random
        random.seed(len(current_text))  # Deterministic based on current state
        
        candidates = []
        for word in random.sample(all_words, min(beam_size, len(all_words))):
            # Simple scoring based on relevance and position
            if word in relevant_words:
                log_prob = random.uniform(-0.5, -0.2)
            elif word in common_words:
                log_prob = random.uniform(-0.8, -0.5)
            else:
                log_prob = random.uniform(-1.5, -1.0)
            
            candidates.append((word, log_prob))
        
        return candidates
    
    def _generate_guidance(self,
                          context: str,
                          generated_text: str,
                          beam_result: BeamSearchResult,
                          filter_result) -> GenerationGuidance:
        """Generate guidance based on generation results."""
        # Evaluate current coherence
        coherence_score = self._evaluate_coherence(context, generated_text)
        
        # Determine trend
        trend = "stable"
        if len(self.session_history) > 0:
            last_coherence = self.session_history[-1].final_coherence
            if coherence_score > last_coherence + 0.1:
                trend = "improving"
            elif coherence_score < last_coherence - 0.1:
                trend = "declining"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            coherence_score, beam_result, filter_result
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(beam_result, filter_result)
        
        return GenerationGuidance(
            coherence_score=coherence_score,
            coherence_trend=trend,
            recommendations=recommendations,
            confidence=confidence,
            metadata={
                "beam_candidates": len(beam_result.all_candidates),
                "filter_passed": len(filter_result.passed_candidates),
                "filter_rate": filter_result.filter_statistics.get("pass_rate", 0)
            }
        )
    
    def _evaluate_coherence(self, context: str, text: str) -> float:
        """Evaluate coherence between context and generated text."""
        if not text.strip():
            return 0.0
        
        prop_set = PropositionSet.from_qa_pair(context, text)
        if len(prop_set.propositions) <= 1:
            return 1.0
        
        result = self.coherence_measure.compute(prop_set)
        return result.score
    
    def _generate_recommendations(self,
                                coherence_score: float,
                                beam_result: BeamSearchResult,
                                filter_result) -> List[str]:
        """Generate recommendations based on generation quality."""
        recommendations = []
        
        if coherence_score < 0.3:
            recommendations.append("Consider revising the generation approach - coherence is low")
            recommendations.append("Try increasing coherence weight in beam search")
        elif coherence_score < 0.5:
            recommendations.append("Generation quality is moderate - consider minor adjustments")
        else:
            recommendations.append("Good coherence achieved - continue current approach")
        
        # Beam search specific recommendations
        if beam_result.beam_statistics:
            coherence_stats = beam_result.beam_statistics.get("coherence_stats", {})
            if coherence_stats.get("std", 0) > 0.3:
                recommendations.append("High coherence variance - consider tuning beam search parameters")
        
        # Filter specific recommendations
        pass_rate = filter_result.filter_statistics.get("pass_rate", 0)
        if pass_rate < 0.2:
            recommendations.append("Low filter pass rate - consider lowering coherence threshold")
        elif pass_rate > 0.8:
            recommendations.append("High filter pass rate - consider raising coherence threshold")
        
        return recommendations
    
    def _calculate_confidence(self, beam_result: BeamSearchResult, filter_result) -> float:
        """Calculate confidence in generation quality."""
        factors = []
        
        # Coherence consistency
        if beam_result.beam_statistics:
            coherence_stats = beam_result.beam_statistics.get("coherence_stats", {})
            coherence_std = coherence_stats.get("std", 1.0)
            consistency_factor = max(0, 1.0 - coherence_std)
            factors.append(consistency_factor)
        
        # Filter effectiveness
        pass_rate = filter_result.filter_statistics.get("pass_rate", 0)
        filter_factor = 1.0 - abs(pass_rate - 0.5) * 2  # Optimal around 50% pass rate
        factors.append(max(0, filter_factor))
        
        # Beam search coverage
        if beam_result.total_steps > 0:
            coverage_factor = min(1.0, beam_result.coherence_evaluations / beam_result.total_steps)
            factors.append(coverage_factor)
        
        return np.mean(factors) if factors else 0.5
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics across all generation sessions."""
        if not self.session_history:
            return {"no_sessions": True}
        
        coherence_scores = [s.final_coherence for s in self.session_history]
        session_times = [s.session_time for s in self.session_history]
        token_counts = [s.token_count for s in self.session_history]
        
        analytics = {
            "total_sessions": len(self.session_history),
            "performance_metrics": {
                "avg_coherence": np.mean(coherence_scores),
                "coherence_trend": "improving" if len(coherence_scores) > 1 and coherence_scores[-1] > coherence_scores[0] else "stable",
                "avg_session_time": np.mean(session_times),
                "avg_token_count": np.mean(token_counts)
            },
            "quality_distribution": {
                "high_quality": sum(1 for s in coherence_scores if s > 0.7) / len(coherence_scores),
                "medium_quality": sum(1 for s in coherence_scores if 0.4 <= s <= 0.7) / len(coherence_scores),
                "low_quality": sum(1 for s in coherence_scores if s < 0.4) / len(coherence_scores)
            }
        }
        
        return analytics


class StreamingCoherenceGuide:
    """
    Real-time coherence guidance for streaming generation.
    
    Provides live feedback during text generation process.
    """
    
    def __init__(self,
                 coherence_measure: Optional[CoherenceMeasure] = None,
                 window_size: int = 50,
                 guidance_frequency: int = 10):
        """
        Initialize streaming guidance system.
        
        Args:
            coherence_measure: Coherence measure for evaluation
            window_size: Size of sliding window for evaluation
            guidance_frequency: How often to provide guidance (tokens)
        """
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.window_size = window_size
        self.guidance_frequency = guidance_frequency
        
        # Streaming state
        self.context = ""
        self.generated_tokens = deque(maxlen=window_size)
        self.coherence_history = deque(maxlen=100)
        self.token_count = 0
        self.guidance_count = 0
    
    def start_session(self, context: str):
        """Start a new streaming guidance session."""
        self.context = context
        self.generated_tokens.clear()
        self.coherence_history.clear()
        self.token_count = 0
        self.guidance_count = 0
    
    def add_token(self, token: str) -> Optional[GenerationGuidance]:
        """
        Add a generated token and optionally return guidance.
        
        Args:
            token: Newly generated token
            
        Returns:
            Guidance if it's time for an update, None otherwise
        """
        self.generated_tokens.append(token)
        self.token_count += 1
        
        # Check if it's time for guidance
        if self.token_count % self.guidance_frequency == 0:
            return self._generate_streaming_guidance()
        
        return None
    
    def _generate_streaming_guidance(self) -> GenerationGuidance:
        """Generate guidance for current generation state."""
        # Get current text
        current_text = " ".join(self.generated_tokens)
        
        # Evaluate coherence
        coherence_score = self._evaluate_coherence(self.context, current_text)
        self.coherence_history.append(coherence_score)
        
        # Determine trend
        trend = "stable"
        if len(self.coherence_history) >= 3:
            recent_avg = np.mean(list(self.coherence_history)[-3:])
            older_avg = np.mean(list(self.coherence_history)[-6:-3]) if len(self.coherence_history) >= 6 else recent_avg
            
            if recent_avg > older_avg + 0.05:
                trend = "improving"
            elif recent_avg < older_avg - 0.05:
                trend = "declining"
        
        # Generate streaming recommendations
        recommendations = self._generate_streaming_recommendations(coherence_score, trend)
        
        # Calculate confidence
        confidence = min(1.0, len(self.coherence_history) / 10)  # Builds with more data
        
        self.guidance_count += 1
        
        return GenerationGuidance(
            coherence_score=coherence_score,
            coherence_trend=trend,
            recommendations=recommendations,
            confidence=confidence,
            metadata={
                "token_count": self.token_count,
                "guidance_count": self.guidance_count,
                "window_size": len(self.generated_tokens)
            }
        )
    
    def _evaluate_coherence(self, context: str, text: str) -> float:
        """Evaluate coherence for streaming context."""
        if not text.strip():
            return 1.0
        
        prop_set = PropositionSet.from_qa_pair(context, text)
        if len(prop_set.propositions) <= 1:
            return 1.0
        
        result = self.coherence_measure.compute(prop_set)
        return result.score
    
    def _generate_streaming_recommendations(self, 
                                          coherence_score: float, 
                                          trend: str) -> List[str]:
        """Generate recommendations for streaming generation."""
        recommendations = []
        
        if trend == "declining":
            recommendations.append("Coherence declining - consider steering toward more relevant content")
            if coherence_score < 0.3:
                recommendations.append("Critical: Very low coherence - major revision needed")
        elif trend == "improving":
            recommendations.append("Good progress - coherence improving")
        
        if coherence_score > 0.7:
            recommendations.append("Excellent coherence - continue current direction")
        elif coherence_score < 0.3:
            recommendations.append("Low coherence - consider revising recent content")
        
        return recommendations
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current streaming session."""
        if not self.coherence_history:
            return {"no_data": True}
        
        coherence_list = list(self.coherence_history)
        
        return {
            "session_stats": {
                "total_tokens": self.token_count,
                "guidance_updates": self.guidance_count,
                "final_coherence": coherence_list[-1],
                "avg_coherence": np.mean(coherence_list),
                "coherence_variance": np.var(coherence_list)
            },
            "trend_analysis": {
                "overall_trend": "improving" if coherence_list[-1] > coherence_list[0] else "declining" if len(coherence_list) > 1 else "stable",
                "peak_coherence": max(coherence_list),
                "lowest_coherence": min(coherence_list)
            },
            "generated_text": " ".join(self.generated_tokens)
        }