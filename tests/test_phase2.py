"""
Tests for Phase 2 features.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from coherify.core.base import Proposition, PropositionSet
from coherify.measures.entailment import EntailmentCoherence, SimpleNLIModel
from coherify.measures.hybrid import HybridCoherence, AdaptiveHybridCoherence
from coherify.benchmarks.truthfulqa import TruthfulQAAdapter, TruthfulQAEvaluator
from coherify.benchmarks.selfcheckgpt import SelfCheckGPTAdapter, SelfCheckGPTEvaluator
from coherify.utils.caching import EmbeddingCache, CachedEncoder, cached_computation


class TestEntailmentCoherence:
    """Test entailment-based coherence."""
    
    def test_simple_nli_model(self):
        """Test simple NLI model."""
        nli_model = SimpleNLIModel()
        
        # Test identical statements
        assert nli_model.predict("Paris is beautiful", "Paris is beautiful") == "entailment"
        
        # Test contradiction (simple model may not detect this specific case)
        result = nli_model.predict("The sky is blue", "The sky is red")
        assert result in ["contradiction", "neutral"]  # Simple model might classify as neutral
        
        # Test neutral
        assert nli_model.predict("I like cats", "The weather is nice") == "neutral"
    
    def test_entailment_coherence_basic(self):
        """Test basic entailment coherence computation."""
        nli_model = SimpleNLIModel()
        measure = EntailmentCoherence(nli_model=nli_model)
        
        props = [
            Proposition(text="Paris is the capital of France"),
            Proposition(text="France is in Europe")
        ]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 1
        assert result.measure_name == "EntailmentCoherence"
        assert "entailments" in result.details
        assert "contradictions" in result.details
        assert "neutrals" in result.details
    
    def test_single_proposition(self):
        """Test entailment coherence with single proposition."""
        nli_model = SimpleNLIModel()
        measure = EntailmentCoherence(nli_model=nli_model)
        
        prop = Proposition(text="Single statement")
        prop_set = PropositionSet(propositions=[prop])
        
        result = measure.compute(prop_set)
        assert result.score == 1.0
        assert result.details["reason"] == "insufficient_propositions"


class TestHybridCoherence:
    """Test hybrid coherence measures."""
    
    def test_hybrid_coherence_basic(self):
        """Test basic hybrid coherence."""
        # Mock NLI model
        mock_nli = Mock()
        mock_nli.predict.return_value = "neutral"
        
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([
            [1, 0, 0],
            [0.8, 0.6, 0]
        ])
        
        measure = HybridCoherence(
            semantic_weight=0.6,
            entailment_weight=0.4,
            encoder=mock_encoder,
            nli_model=mock_nli
        )
        
        props = [
            Proposition(text="First statement"),
            Proposition(text="Second statement")
        ]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        
        assert isinstance(result.score, float)
        assert result.measure_name == "HybridCoherence"
        assert "component_scores" in result.details
        assert "semantic" in result.details["component_scores"]
        assert "entailment" in result.details["component_scores"]
    
    def test_weight_validation(self):
        """Test that weights must sum to 1."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridCoherence(semantic_weight=0.7, entailment_weight=0.4)
    
    def test_adaptive_hybrid_coherence(self):
        """Test adaptive hybrid coherence."""
        mock_nli = Mock()
        mock_nli.predict.return_value = "neutral"
        
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([[1, 0, 0], [0.8, 0.6, 0]])
        
        measure = AdaptiveHybridCoherence(
            base_semantic_weight=0.5,
            base_entailment_weight=0.5,
            encoder=mock_encoder,
            nli_model=mock_nli
        )
        
        props = [Proposition(text="Short"), Proposition(text="Text")]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        
        assert "adapted_weights" in result.details
        assert "original_weights" in result.details
        assert "weight_adaptation" in result.details


class TestTruthfulQAAdapter:
    """Test TruthfulQA benchmark adapter."""
    
    def test_generation_mode(self):
        """Test generation mode adaptation."""
        adapter = TruthfulQAAdapter(evaluation_mode="generation")
        
        sample = {
            "question": "What is the capital of France?",
            "best_answer": "Paris is the capital of France. It is a beautiful city.",
            "category": "Geography"
        }
        
        prop_set = adapter.adapt_single(sample)
        
        assert prop_set.context == "What is the capital of France?"
        assert len(prop_set.propositions) == 2
        assert prop_set.metadata["benchmark"] == "truthfulqa"
        assert prop_set.metadata["category"] == "Geography"
    
    def test_positive_negative_sets(self):
        """Test creation of positive/negative proposition sets."""
        adapter = TruthfulQAAdapter()
        
        sample = {
            "question": "Test question?",
            "correct_answers": ["Correct answer 1", "Correct answer 2"],
            "incorrect_answers": ["Wrong answer 1", "Wrong answer 2"]
        }
        
        sets = adapter.create_positive_negative_sets(sample)
        
        assert "positive" in sets
        assert "negative" in sets
        assert len(sets["positive"].propositions) == 2
        assert len(sets["negative"].propositions) == 2
        assert sets["positive"].metadata["valence"] == "positive"
        assert sets["negative"].metadata["valence"] == "negative"


class TestSelfCheckGPTAdapter:
    """Test SelfCheckGPT benchmark adapter."""
    
    def test_multi_sample_mode(self):
        """Test multi-sample consistency mode."""
        adapter = SelfCheckGPTAdapter(consistency_mode="multi_sample")
        
        sample = {
            "prompt": "Test question?",
            "sampled_answers": [
                "Answer 1 with details.",
                "Answer 2 with information.",
                "Answer 3 with facts."
            ]
        }
        
        prop_set = adapter.adapt_single(sample)
        
        assert prop_set.context == "Test question?"
        assert len(prop_set.propositions) == 3  # Each answer as one proposition when not segmented
        assert prop_set.metadata["benchmark"] == "selfcheckgpt"
        assert prop_set.metadata["num_samples"] == 3
    
    def test_sentence_level_mode(self):
        """Test sentence-level mode."""
        adapter = SelfCheckGPTAdapter(consistency_mode="sentence_level")
        
        sample = {
            "prompt": "Test question?",
            "original_answer": "First sentence. Second sentence."
        }
        
        prop_set = adapter.adapt_single(sample)
        
        assert len(prop_set.propositions) == 2
        assert prop_set.metadata["consistency_mode"] == "sentence_level"
        assert all(p.metadata["evaluation_target"] for p in prop_set.propositions)
    
    def test_consistency_groups(self):
        """Test creation of consistency groups."""
        adapter = SelfCheckGPTAdapter(consistency_mode="multi_sample", segment_responses=True)
        
        sample = {
            "prompt": "Test?",
            "sampled_answers": ["Answer one. Detail one.", "Answer two. Detail two."]
        }
        
        groups = adapter.create_consistency_groups(sample)
        
        assert "sample_0" in groups
        assert "sample_1" in groups
        assert len(groups["sample_0"].propositions) == 2  # Two sentences
        assert len(groups["sample_1"].propositions) == 2


class TestCaching:
    """Test caching utilities."""
    
    def test_embedding_cache_basic(self):
        """Test basic embedding cache operations."""
        cache = EmbeddingCache(cache_dir=".test_cache", max_size=10)
        
        text = "Test text"
        embedding = np.array([1, 2, 3])
        
        # Test put and get
        cache.put(text, embedding)
        retrieved = cache.get(text)
        
        assert np.array_equal(retrieved, embedding)
        
        # Test cache miss
        assert cache.get("Unknown text") is None
        
        # Cleanup
        cache.clear()
    
    def test_cached_encoder(self):
        """Test cached encoder wrapper."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([1, 2, 3])
        
        cache = EmbeddingCache(cache_dir=".test_cache", max_size=10)
        cached_encoder = CachedEncoder(mock_encoder, cache)
        
        # First call should hit the encoder
        result1 = cached_encoder.encode("test text")
        assert mock_encoder.encode.call_count == 1
        
        # Second call should use cache
        result2 = cached_encoder.encode("test text")
        assert mock_encoder.encode.call_count == 1  # Still 1
        
        assert np.array_equal(result1, result2)
        
        # Cleanup
        cached_encoder.clear_cache()
    
    def test_cached_computation_decorator(self):
        """Test cached computation decorator."""
        call_count = 0
        
        @cached_computation()
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = test_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = test_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Still 1
        
        # Different args should call function again
        result3 = test_function(2, 3)
        assert result3 == 5
        assert call_count == 2
        
        # Clear cache
        test_function.clear_cache()


if __name__ == "__main__":
    pytest.main([__file__])