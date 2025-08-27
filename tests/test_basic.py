"""
Basic tests for coherify functionality.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from coherify.benchmarks.adapters import QABenchmarkAdapter
from coherify.core.base import CoherenceResult, Proposition, PropositionSet
from coherify.measures.semantic import SemanticCoherence


class TestProposition:
    """Test Proposition class."""

    def test_create_proposition(self):
        """Test basic proposition creation."""
        prop = Proposition(text="Paris is the capital of France.")
        assert prop.text == "Paris is the capital of France."
        assert prop.probability is None
        assert isinstance(prop.metadata, dict)

    def test_proposition_with_metadata(self):
        """Test proposition with metadata."""
        metadata = {"source": "wikipedia", "confidence": 0.9}
        prop = Proposition(text="Test", metadata=metadata)
        assert prop.metadata == metadata


class TestPropositionSet:
    """Test PropositionSet class."""

    def test_from_qa_pair(self):
        """Test creating PropositionSet from QA pair."""
        question = "What is the capital of France?"
        answer = "Paris is the capital. It's a beautiful city."

        prop_set = PropositionSet.from_qa_pair(question, answer)

        assert prop_set.context == question
        assert len(prop_set.propositions) == 2
        assert prop_set.propositions[0].text == "Paris is the capital"
        assert prop_set.propositions[1].text == "It's a beautiful city"

    def test_from_multi_answer(self):
        """Test creating PropositionSet from multiple answers."""
        question = "What is 2+2?"
        answers = ["Four", "The answer is 4", "2+2=4"]

        prop_set = PropositionSet.from_multi_answer(question, answers)

        assert prop_set.context == question
        assert len(prop_set.propositions) == 3
        assert [p.text for p in prop_set.propositions] == answers

    def test_len_and_iter(self):
        """Test len and iter methods."""
        props = [Proposition(text=f"Statement {i}") for i in range(3)]
        prop_set = PropositionSet(propositions=props)

        assert len(prop_set) == 3
        assert list(prop_set) == props


class TestSemanticCoherence:
    """Test SemanticCoherence class."""

    def test_single_proposition(self):
        """Test coherence with single proposition."""
        # Mock encoder to avoid dependency on sentence-transformers in tests
        mock_encoder = Mock()
        coherence = SemanticCoherence(encoder=mock_encoder)

        prop = Proposition(text="Single statement")
        prop_set = PropositionSet(propositions=[prop])

        result = coherence.compute(prop_set)

        assert result.score == 1.0
        assert result.measure_name == "SemanticCoherence"
        assert result.details["reason"] == "insufficient_propositions"

    def test_coherence_computation(self):
        """Test coherence computation with mock encoder."""
        # Mock encoder that returns predictable embeddings
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.array(
            [[1, 0, 0], [0.8, 0.6, 0]]  # First embedding  # Second embedding (similar)
        )

        coherence = SemanticCoherence(encoder=mock_encoder)

        props = [
            Proposition(text="First statement"),
            Proposition(text="Second statement"),
        ]
        prop_set = PropositionSet(propositions=props)

        result = coherence.compute(prop_set)

        assert isinstance(result.score, float)
        assert result.measure_name == "SemanticCoherence"
        assert result.details["num_propositions"] == 2
        assert result.details["total_pairs"] == 1

        # Check that encoder was called with correct texts
        mock_encoder.encode.assert_called_once_with(
            ["First statement", "Second statement"]
        )


class TestQABenchmarkAdapter:
    """Test QA benchmark adapter."""

    def test_basic_qa_adaptation(self):
        """Test basic QA adaptation."""
        adapter = QABenchmarkAdapter("test_benchmark")

        sample = {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence. It involves machine learning.",
        }

        prop_set = adapter.adapt_single(sample)

        assert prop_set.context == "What is AI?"
        assert len(prop_set.propositions) == 2
        assert prop_set.propositions[0].text == "AI is artificial intelligence"
        assert prop_set.propositions[1].text == "It involves machine learning"

    def test_multiple_answers_adaptation(self):
        """Test adaptation with multiple answers."""
        adapter = QABenchmarkAdapter(
            "test_benchmark", multiple_answers_key="all_answers"
        )

        sample = {
            "question": "What is 2+2?",
            "answer": "Four",
            "all_answers": ["4", "Four", "2+2=4"],
        }

        prop_set = adapter.adapt_single(sample)

        assert prop_set.context == "What is 2+2?"
        assert len(prop_set.propositions) == 3
        assert [p.text for p in prop_set.propositions] == ["4", "Four", "2+2=4"]


class TestCoherenceResult:
    """Test CoherenceResult class."""

    def test_exceeds_threshold(self):
        """Test threshold checking."""
        result = CoherenceResult(
            score=0.75, measure_name="Test", details={}, computation_time=0.1
        )

        assert result.exceeds_threshold(0.5) == True
        assert result.exceeds_threshold(0.8) == False

    def test_string_representation(self):
        """Test string representation."""
        result = CoherenceResult(
            score=0.756, measure_name="TestMeasure", details={}, computation_time=0.123
        )

        str_repr = str(result)
        assert "TestMeasure" in str_repr
        assert "0.756" in str_repr
        assert "0.123" in str_repr


if __name__ == "__main__":
    pytest.main([__file__])
