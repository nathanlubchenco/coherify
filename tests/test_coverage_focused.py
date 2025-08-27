"""
Focused tests to improve coverage on key modules.
"""

import tempfile

from coherify.core.base import CoherenceResult, Proposition, PropositionSet
from coherify.measures.hybrid import HybridCoherence
from coherify.measures.semantic import SemanticCoherence
from coherify.utils.caching import ComputationCache, EmbeddingCache


class TestSemanticCoherenceFocused:
    """Simple tests for semantic coherence to improve coverage."""

    def test_empty_set(self):
        """Test with empty proposition set."""
        measure = SemanticCoherence()
        prop_set = PropositionSet(propositions=[])
        result = measure.compute(prop_set)
        # Empty/single proposition sets return 1.0 (perfectly coherent with self)
        assert result.score == 1.0
        assert result.details["reason"] == "insufficient_propositions"

    def test_single_proposition(self):
        """Test with single proposition."""
        measure = SemanticCoherence()
        props = [Proposition(text="Test")]
        prop_set = PropositionSet(propositions=props)
        result = measure.compute(prop_set)
        assert result.score == 1.0


class TestHybridCoherenceFocused:
    """Simple tests for hybrid coherence."""

    def test_empty_set(self):
        """Test with empty set."""
        measure = HybridCoherence()
        prop_set = PropositionSet(propositions=[])
        result = measure.compute(prop_set)
        # Empty/single proposition sets return 1.0 (perfectly coherent with self)
        assert result.score == 1.0
        assert result.details["reason"] == "insufficient_propositions"

    def test_weight_normalization(self):
        """Test that weights must sum to 1.0."""
        # HybridCoherence validates weights sum to 1.0, doesn't normalize them
        try:
            measure = HybridCoherence(semantic_weight=2.0, entailment_weight=1.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Weights must sum to 1.0" in str(e)

        # Valid weights should work
        measure = HybridCoherence(semantic_weight=0.7, entailment_weight=0.3)
        assert abs(measure.semantic_weight + measure.entailment_weight - 1.0) < 1e-6


class TestCachingFocused:
    """Simple caching tests."""

    def test_embedding_cache_basic(self):
        """Test basic embedding cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)

            # Test cache key generation
            key1 = cache._get_cache_key("text1", "model")
            key2 = cache._get_cache_key("text2", "model")
            assert key1 != key2

            # Test empty cache
            assert cache.get("nonexistent") is None

    def test_computation_cache_basic(self):
        """Test basic computation cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ComputationCache(cache_dir=temp_dir)

            # Test cache key generation
            key1 = cache._get_cache_key("func", (1,), {})
            key2 = cache._get_cache_key("func", (2,), {})
            assert key1 != key2

            # Test empty cache
            assert cache.get("func", (1,), {}) is None


class TestPropositionSetExtended:
    """Extended tests for PropositionSet to improve core coverage."""

    def test_from_qa_pair_edge_cases(self):
        """Test QA pair creation edge cases."""
        # Empty answer
        prop_set = PropositionSet.from_qa_pair("Question?", "")
        assert len(prop_set.propositions) == 0

        # Single word answer
        prop_set = PropositionSet.from_qa_pair("Question?", "Yes")
        assert len(prop_set.propositions) == 1

        # Answer with only punctuation
        prop_set = PropositionSet.from_qa_pair("Question?", "...")
        assert len(prop_set.propositions) == 0

    def test_from_multi_answer_edge_cases(self):
        """Test multi-answer creation edge cases."""
        # Empty answers list
        prop_set = PropositionSet.from_multi_answer("Question?", [])
        assert len(prop_set.propositions) == 0

        # List with empty strings (from_multi_answer creates propositions for all inputs)
        prop_set = PropositionSet.from_multi_answer("Question?", ["", "  ", "Valid"])
        assert len(prop_set.propositions) == 3
        assert prop_set.propositions[0].text == ""
        assert prop_set.propositions[1].text == "  "
        assert prop_set.propositions[2].text == "Valid"

    def test_metadata_handling(self):
        """Test metadata handling."""
        metadata = {"source": "test", "confidence": 0.8}
        prop_set = PropositionSet(
            propositions=[Proposition(text="Test")], metadata=metadata
        )
        assert prop_set.metadata == metadata

    def test_string_representation(self):
        """Test string representation."""
        props = [Proposition(text="Test 1"), Proposition(text="Test 2")]
        prop_set = PropositionSet(propositions=props, context="Context")
        str_repr = str(prop_set)
        assert "Context" in str_repr
        assert "Test 1" in str_repr


class TestCoherenceResultExtended:
    """Extended tests for CoherenceResult."""

    def test_coherence_result_creation(self):
        """Test coherence result creation."""
        result = CoherenceResult(
            score=0.75,
            measure_name="TestMeasure",
            details={"method": "test", "pairwise_scores": [0.8, 0.7]},
            computation_time=0.1,
        )
        assert result.score == 0.75
        assert len(result.details["pairwise_scores"]) == 2
        assert result.details["method"] == "test"

    def test_coherence_result_string(self):
        """Test string representation."""
        result = CoherenceResult(
            score=0.5, measure_name="TestMeasure", details={}, computation_time=0.05
        )
        str_repr = str(result)
        assert "0.5" in str_repr


class TestPropositionExtended:
    """Extended tests for Proposition."""

    def test_proposition_equality(self):
        """Test proposition equality."""
        prop1 = Proposition(text="Same text")
        prop2 = Proposition(text="Same text")
        prop3 = Proposition(text="Different text")

        # Propositions implement equality based on content (dataclass behavior)
        assert prop1 == prop2  # Same content
        assert prop1 != prop3  # Different content

        # Test with metadata
        prop4 = Proposition(text="Same text", metadata={"key": "value"})
        prop5 = Proposition(text="Same text", metadata={"key": "value"})
        prop6 = Proposition(text="Same text", metadata={"key": "different"})

        assert prop4 == prop5  # Same content and metadata
        assert prop4 != prop6  # Different metadata
        assert prop1 != prop4  # Different metadata (empty vs non-empty)

    def test_proposition_with_probability(self):
        """Test proposition with probability."""
        prop = Proposition(text="Test", probability=0.9)
        assert prop.probability == 0.9

    def test_proposition_metadata_modification(self):
        """Test metadata modification."""
        prop = Proposition(text="Test")
        prop.metadata["new_key"] = "new_value"
        assert prop.metadata["new_key"] == "new_value"
