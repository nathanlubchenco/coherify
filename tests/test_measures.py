"""
Tests for coherence measures to improve coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from coherify.core.base import Proposition, PropositionSet, CoherenceResult
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.entailment import EntailmentCoherence, SimpleNLIModel
from coherify.measures.hybrid import HybridCoherence, AdaptiveHybridCoherence


class TestSemanticCoherence:
    """Test semantic coherence with better coverage."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        measure = SemanticCoherence()
        assert measure.model_name == "all-MiniLM-L6-v2"
        assert measure.similarity_threshold == 0.3

    def test_initialization_custom(self):
        """Test custom initialization."""
        measure = SemanticCoherence(
            model_name="custom-model", similarity_threshold=0.5
        )
        assert measure.model_name == "custom-model"
        assert measure.similarity_threshold == 0.5

    @patch('coherify.measures.semantic.SentenceTransformer')
    def test_compute_with_mock_encoder(self, mock_st):
        """Test compute with mocked encoder."""
        # Mock encoder to return fixed embeddings
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # First proposition
            [0.8, 0.6, 0.0],  # Second proposition (high similarity)
        ])
        mock_st.return_value = mock_encoder

        measure = SemanticCoherence()
        props = [
            Proposition(text="Paris is beautiful"),
            Proposition(text="Paris is lovely"),
        ]
        prop_set = PropositionSet(propositions=props)

        result = measure.compute(prop_set)
        
        assert isinstance(result, CoherenceResult)
        assert result.score > 0.5  # Should be high due to similar embeddings
        assert len(result.pairwise_scores) == 1  # One pair

    def test_empty_proposition_set(self):
        """Test with empty proposition set."""
        measure = SemanticCoherence()
        prop_set = PropositionSet(propositions=[])
        
        result = measure.compute(prop_set)
        assert result.score == 0.0
        assert result.pairwise_scores == []

    def test_single_proposition(self):
        """Test with single proposition."""
        measure = SemanticCoherence()
        props = [Proposition(text="Single statement")]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        assert result.score == 1.0  # Perfect coherence for single statement
        assert result.pairwise_scores == []


class TestEntailmentCoherenceCoverage:
    """Additional tests for entailment coherence coverage."""

    def test_simple_nli_model_edge_cases(self):
        """Test SimpleNLIModel edge cases."""
        model = SimpleNLIModel()
        
        # Test empty strings
        result = model.predict("", "")
        assert result in ["entailment", "neutral", "contradiction"]
        
        # Test very long strings (should still work)
        long_text = "This is a very long text. " * 100
        result = model.predict(long_text, "Short text")
        assert result in ["entailment", "neutral", "contradiction"]

    def test_entailment_coherence_initialization(self):
        """Test EntailmentCoherence initialization."""
        nli_model = SimpleNLIModel()
        measure = EntailmentCoherence(nli_model=nli_model)
        assert measure.nli_model == nli_model
        
        # Test with default model
        measure_default = EntailmentCoherence()
        assert measure_default.nli_model is not None

    def test_entailment_coherence_empty_set(self):
        """Test with empty proposition set."""
        measure = EntailmentCoherence()
        prop_set = PropositionSet(propositions=[])
        
        result = measure.compute(prop_set)
        assert result.score == 0.0

    def test_entailment_coherence_single_prop(self):
        """Test with single proposition."""
        measure = EntailmentCoherence()
        props = [Proposition(text="Single statement")]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        assert result.score == 1.0

    @patch('coherify.measures.entailment.create_pipeline_with_suppressed_warnings')
    def test_entailment_with_mock_pipeline(self, mock_pipeline):
        """Test entailment with mocked pipeline."""
        # Mock pipeline to return specific predictions
        mock_pipe = Mock()
        mock_pipe.return_value = [
            {"label": "ENTAILMENT", "score": 0.9},
            {"label": "NEUTRAL", "score": 0.1},
        ]
        mock_pipeline.return_value = mock_pipe

        measure = EntailmentCoherence()
        props = [
            Proposition(text="Statement A"),
            Proposition(text="Statement B"),
        ]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        assert isinstance(result, CoherenceResult)
        assert result.score >= 0.0


class TestHybridCoherenceCoverage:
    """Additional tests for hybrid coherence coverage."""

    def test_hybrid_coherence_initialization(self):
        """Test HybridCoherence initialization."""
        semantic = SemanticCoherence()
        entailment = EntailmentCoherence()
        
        # Test with custom weights
        measure = HybridCoherence(
            semantic_measure=semantic,
            entailment_measure=entailment,
            semantic_weight=0.7,
            entailment_weight=0.3
        )
        
        assert measure.semantic_weight == 0.7
        assert measure.entailment_weight == 0.3

    def test_hybrid_coherence_normalization(self):
        """Test weight normalization."""
        semantic = SemanticCoherence()
        entailment = EntailmentCoherence()
        
        # Weights that don't sum to 1
        measure = HybridCoherence(
            semantic_measure=semantic,
            entailment_measure=entailment,
            semantic_weight=2.0,
            entailment_weight=1.0
        )
        
        # Should normalize to 2/3 and 1/3
        assert abs(measure.semantic_weight - 2/3) < 1e-6
        assert abs(measure.entailment_weight - 1/3) < 1e-6

    def test_adaptive_hybrid_initialization(self):
        """Test AdaptiveHybridCoherence initialization."""
        measure = AdaptiveHybridCoherence()
        assert hasattr(measure, 'semantic_measure')
        assert hasattr(measure, 'entailment_measure')
        assert hasattr(measure, 'learning_rate')

    def test_adaptive_hybrid_empty_set(self):
        """Test adaptive hybrid with empty set."""
        measure = AdaptiveHybridCoherence()
        prop_set = PropositionSet(propositions=[])
        
        result = measure.compute(prop_set)
        assert result.score == 0.0

    @patch('coherify.measures.semantic.SentenceTransformer')
    @patch('coherify.measures.entailment.create_pipeline_with_suppressed_warnings')
    def test_hybrid_compute_with_mocks(self, mock_nli, mock_st):
        """Test hybrid compute with mocked components."""
        # Mock semantic encoder
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ])
        mock_st.return_value = mock_encoder
        
        # Mock NLI pipeline
        mock_pipe = Mock()
        mock_pipe.return_value = [{"label": "NEUTRAL", "score": 0.6}]
        mock_nli.return_value = mock_pipe
        
        semantic = SemanticCoherence()
        entailment = EntailmentCoherence()
        measure = HybridCoherence(
            semantic_measure=semantic,
            entailment_measure=entailment
        )
        
        props = [
            Proposition(text="Statement A"),
            Proposition(text="Statement B"),
        ]
        prop_set = PropositionSet(propositions=props)
        
        result = measure.compute(prop_set)
        assert isinstance(result, CoherenceResult)
        assert 0.0 <= result.score <= 1.0