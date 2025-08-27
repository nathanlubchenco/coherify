"""
Tests for coherence measures to improve coverage.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from coherify.core.base import CoherenceResult, Proposition, PropositionSet
from coherify.measures.entailment import EntailmentCoherence, SimpleNLIModel
from coherify.measures.hybrid import AdaptiveHybridCoherence, HybridCoherence
from coherify.measures.semantic import SemanticCoherence


class TestSemanticCoherence:
    """Test semantic coherence with better coverage."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        measure = SemanticCoherence()
        assert measure.similarity_threshold == 0.0
        assert measure.aggregation == "mean"
        # Encoder should be created but we can't test the exact model name
        assert measure.encoder is not None

    def test_initialization_custom(self):
        """Test custom initialization."""
        mock_encoder = Mock()
        measure = SemanticCoherence(
            encoder=mock_encoder, similarity_threshold=0.5, aggregation="min"
        )
        assert measure.encoder == mock_encoder
        assert measure.similarity_threshold == 0.5
        assert measure.aggregation == "min"

    def test_compute_with_mock_encoder(self):
        """Test compute with mocked encoder."""
        # Mock encoder to return fixed embeddings
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],  # First proposition
                [0.8, 0.6, 0.0],  # Second proposition (high similarity)
            ]
        )

        measure = SemanticCoherence(encoder=mock_encoder)
        props = [
            Proposition(text="Paris is beautiful"),
            Proposition(text="Paris is lovely"),
        ]
        prop_set = PropositionSet(propositions=props)

        result = measure.compute(prop_set)

        assert isinstance(result, CoherenceResult)
        assert result.score > 0.5  # Should be high due to similar embeddings
        # Check details structure
        assert "pairwise_similarities" in result.details
        assert len(result.details["pairwise_similarities"]) == 1  # One pair

    def test_empty_proposition_set(self):
        """Test with empty proposition set."""
        mock_encoder = Mock()
        measure = SemanticCoherence(encoder=mock_encoder)
        prop_set = PropositionSet(propositions=[])

        result = measure.compute(prop_set)
        assert result.score == 1.0  # Insufficient propositions returns 1.0
        assert result.details["reason"] == "insufficient_propositions"
        assert "pairwise_similarities" not in result.details

    def test_single_proposition(self):
        """Test with single proposition."""
        mock_encoder = Mock()
        measure = SemanticCoherence(encoder=mock_encoder)
        props = [Proposition(text="Single statement")]
        prop_set = PropositionSet(propositions=props)

        result = measure.compute(prop_set)
        assert result.score == 1.0  # Perfect coherence for single statement
        assert result.details["reason"] == "insufficient_propositions"
        assert "pairwise_similarities" not in result.details


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
        mock_nli = Mock()
        measure = EntailmentCoherence(nli_model=mock_nli)
        prop_set = PropositionSet(propositions=[])

        result = measure.compute(prop_set)
        assert result.score == 1.0  # Insufficient propositions returns 1.0
        assert result.details["reason"] == "insufficient_propositions"

    def test_entailment_coherence_single_prop(self):
        """Test with single proposition."""
        mock_nli = Mock()
        measure = EntailmentCoherence(nli_model=mock_nli)
        props = [Proposition(text="Single statement")]
        prop_set = PropositionSet(propositions=props)

        result = measure.compute(prop_set)
        assert result.score == 1.0
        assert result.details["reason"] == "insufficient_propositions"

    def test_entailment_with_mock_pipeline(self):
        """Test entailment with mocked NLI model."""
        # Mock NLI model to return specific predictions
        mock_nli = Mock()
        # Mock predict method to return entailment for both directions
        mock_nli.predict.return_value = "entailment"

        measure = EntailmentCoherence(nli_model=mock_nli)
        props = [
            Proposition(text="Statement A"),
            Proposition(text="Statement B"),
        ]
        prop_set = PropositionSet(propositions=props)

        result = measure.compute(prop_set)
        assert isinstance(result, CoherenceResult)
        # With all entailments, score should be positive
        assert result.score > 0
        # Should have called predict for both directions (A->B and B->A)
        assert mock_nli.predict.call_count == 2


class TestHybridCoherenceCoverage:
    """Additional tests for hybrid coherence coverage."""

    def test_hybrid_coherence_initialization(self):
        """Test HybridCoherence initialization."""
        # Test with custom weights and mock models
        mock_encoder = Mock()
        mock_nli = Mock()

        measure = HybridCoherence(
            semantic_weight=0.7,
            entailment_weight=0.3,
            encoder=mock_encoder,
            nli_model=mock_nli,
        )

        assert measure.semantic_weight == 0.7
        assert measure.entailment_weight == 0.3
        assert measure.semantic_measure.encoder == mock_encoder
        assert measure.entailment_measure.nli_model == mock_nli

    def test_hybrid_coherence_normalization(self):
        """Test weight validation."""
        mock_encoder = Mock()
        mock_nli = Mock()

        # Weights that don't sum to 1 should raise ValueError
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridCoherence(
                semantic_weight=0.7,
                entailment_weight=0.4,  # 0.7 + 0.4 = 1.1 ≠ 1.0
                encoder=mock_encoder,
                nli_model=mock_nli,
            )

    def test_adaptive_hybrid_initialization(self):
        """Test AdaptiveHybridCoherence initialization."""
        mock_encoder = Mock()
        mock_nli = Mock()
        measure = AdaptiveHybridCoherence(
            base_semantic_weight=0.3,
            base_entailment_weight=0.7,
            encoder=mock_encoder,
            nli_model=mock_nli,
        )
        assert hasattr(measure, "semantic_measure")
        assert hasattr(measure, "entailment_measure")
        assert measure.base_semantic_weight == 0.3
        assert measure.base_entailment_weight == 0.7

    def test_adaptive_hybrid_empty_set(self):
        """Test adaptive hybrid with empty set."""
        mock_encoder = Mock()
        mock_nli = Mock()
        measure = AdaptiveHybridCoherence(encoder=mock_encoder, nli_model=mock_nli)
        prop_set = PropositionSet(propositions=[])

        result = measure.compute(prop_set)
        assert result.score == 1.0  # Insufficient propositions returns 1.0
        assert result.details["reason"] == "insufficient_propositions"

    def test_hybrid_compute_with_mocks(self):
        """Test hybrid compute with mocked components."""
        # Mock semantic encoder
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )

        # Mock NLI model
        mock_nli = Mock()
        mock_nli.predict.return_value = "neutral"

        # Create HybridCoherence with mock components
        measure = HybridCoherence(encoder=mock_encoder, nli_model=mock_nli)

        props = [
            Proposition(text="Statement A"),
            Proposition(text="Statement B"),
        ]
        prop_set = PropositionSet(propositions=props)

        result = measure.compute(prop_set)
        assert isinstance(result, CoherenceResult)
        assert 0.0 <= result.score <= 1.0
