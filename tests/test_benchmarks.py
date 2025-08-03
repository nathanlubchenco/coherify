"""
Tests for benchmark adapters to improve coverage.
"""

import pytest
from unittest.mock import Mock, patch

from coherify.core.base import Proposition, PropositionSet
from coherify.benchmarks.adapters import QABenchmarkAdapter, SummarizationBenchmarkAdapter
from coherify.benchmarks.selfcheckgpt import SelfCheckGPTAdapter
from coherify.benchmarks.truthfulqa import TruthfulQAAdapter


class TestQABenchmarkAdapter:
    """Test QA benchmark adapter with better coverage."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = QABenchmarkAdapter("test-qa")
        assert adapter is not None
        assert adapter.benchmark_name == "test-qa"

    def test_convert_qa_basic(self):
        """Test basic QA conversion."""
        adapter = QABenchmarkAdapter("test-qa")
        
        qa_item = {
            "question": "What is the capital of France?",
            "answer": "Paris is the capital. It's beautiful."
        }
        
        prop_set = adapter.adapt_single(qa_item)
        
        assert isinstance(prop_set, PropositionSet)
        assert prop_set.context == "What is the capital of France?"
        assert len(prop_set.propositions) == 2
        assert "Paris is the capital" in [p.text for p in prop_set.propositions]

    def test_convert_qa_multiple_answers(self):
        """Test QA conversion with multiple answers."""
        adapter = QABenchmarkAdapter("test-qa")
        
        qa_item = {
            "question": "What is 2+2?",
            "answers": ["Four", "The answer is 4", "2+2=4"]
        }
        
        prop_set = adapter.adapt_single(qa_item)
        
        assert isinstance(prop_set, PropositionSet)
        assert len(prop_set.propositions) == 3

    def test_convert_qa_empty_answer(self):
        """Test QA conversion with empty answer."""
        adapter = QABenchmarkAdapter("test-qa")
        
        qa_item = {
            "question": "What is the answer?",
            "answer": ""
        }
        
        prop_set = adapter.adapt_single(qa_item)
        
        assert isinstance(prop_set, PropositionSet)
        assert len(prop_set.propositions) == 0

    def test_convert_qa_missing_fields(self):
        """Test QA conversion with missing fields."""
        adapter = QABenchmarkAdapter("test-qa")
        
        # Missing answer field
        qa_item = {"question": "What is this?"}
        
        with pytest.raises(KeyError):
            adapter.convert(qa_item)

    def test_convert_qa_with_metadata(self):
        """Test QA conversion preserves metadata."""
        adapter = QABenchmarkAdapter("test-qa")
        
        qa_item = {
            "question": "Test question?",
            "answer": "Test answer.",
            "category": "science",
            "difficulty": "easy"
        }
        
        prop_set = adapter.adapt_single(qa_item)
        
        assert prop_set.metadata["category"] == "science"
        assert prop_set.metadata["difficulty"] == "easy"


class TestSummarizationBenchmarkAdapter:
    """Test document summary adapter coverage."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = SummarizationBenchmarkAdapter("test-benchmark")
        assert adapter is not None
        assert adapter.benchmark_name == "test-benchmark"

    def test_convert_document_summary(self):
        """Test document-summary conversion."""
        adapter = SummarizationBenchmarkAdapter("test-benchmark")
        
        item = {
            "document": "This is a long document with multiple facts.",
            "summary": "Document contains facts. It has information."
        }
        
        prop_set = adapter.adapt_single(item)
        
        assert isinstance(prop_set, PropositionSet)
        assert "This is a long document" in prop_set.context
        assert len(prop_set.propositions) == 2

    def test_convert_empty_summary(self):
        """Test with empty summary."""
        adapter = SummarizationBenchmarkAdapter("test-benchmark")
        
        item = {
            "document": "Test document",
            "summary": ""
        }
        
        prop_set = adapter.adapt_single(item)
        assert len(prop_set.propositions) == 0

    def test_convert_missing_document(self):
        """Test with missing document field."""
        adapter = SummarizationBenchmarkAdapter("test-benchmark")
        
        item = {"summary": "Test summary"}
        
        with pytest.raises(KeyError):
            adapter.convert(item)


class TestSelfCheckGPTAdapter:
    """Test SelfCheckGPT adapter coverage."""

    def test_initialization_default(self):
        """Test default initialization."""
        adapter = SelfCheckGPTAdapter()
        assert adapter.sample_size == 5
        assert adapter.temperature == 0.8

    def test_initialization_custom(self):
        """Test custom initialization."""
        adapter = SelfCheckGPTAdapter(sample_size=10, temperature=0.5)
        assert adapter.sample_size == 10
        assert adapter.temperature == 0.5

    def test_convert_basic(self):
        """Test basic conversion."""
        adapter = SelfCheckGPTAdapter()
        
        item = {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence. It involves machines learning."
        }
        
        prop_set = adapter.adapt_single(item)
        
        assert isinstance(prop_set, PropositionSet)
        assert len(prop_set.propositions) == 2

    @patch('coherify.benchmarks.selfcheckgpt.OpenAI')
    def test_generate_samples_mock(self, mock_openai):
        """Test sample generation with mocked OpenAI."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Sample response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        adapter = SelfCheckGPTAdapter()
        samples = adapter._generate_samples("Test question", "Original answer")
        
        assert len(samples) == adapter.sample_size
        assert all(sample == "Sample response" for sample in samples)

    def test_convert_with_samples(self):
        """Test conversion that includes sample generation."""
        adapter = SelfCheckGPTAdapter()
        
        # Mock the sample generation to avoid API calls
        adapter._generate_samples = Mock(return_value=[
            "Sample 1 response",
            "Sample 2 response",
        ])
        
        item = {
            "question": "Test question?",
            "answer": "Original answer"
        }
        
        prop_set = adapter.adapt_single(item)
        
        assert isinstance(prop_set, PropositionSet)
        # Should include original answer plus samples
        assert len(prop_set.propositions) >= 2


class TestTruthfulQAAdapter:
    """Test TruthfulQA adapter coverage."""

    def test_initialization_default(self):
        """Test default initialization."""
        adapter = TruthfulQAAdapter()
        assert adapter.include_incorrect_answers is True
        assert adapter.max_incorrect_answers == 3

    def test_initialization_custom(self):
        """Test custom initialization."""
        adapter = TruthfulQAAdapter(
            include_incorrect_answers=False,
            max_incorrect_answers=5
        )
        assert adapter.include_incorrect_answers is False
        assert adapter.max_incorrect_answers == 5

    def test_convert_basic(self):
        """Test basic TruthfulQA conversion."""
        adapter = TruthfulQAAdapter()
        
        item = {
            "question": "What is the capital of France?",
            "best_answer": "Paris",
            "correct_answers": ["Paris", "Paris is the capital"],
            "incorrect_answers": ["London", "Berlin", "Madrid"]
        }
        
        prop_set = adapter.adapt_single(item)
        
        assert isinstance(prop_set, PropositionSet)
        assert prop_set.context == "What is the capital of France?"
        # Should include correct and some incorrect answers
        assert len(prop_set.propositions) >= 2

    def test_convert_only_correct(self):
        """Test conversion with only correct answers."""
        adapter = TruthfulQAAdapter(include_incorrect_answers=False)
        
        item = {
            "question": "What is 2+2?",
            "best_answer": "4",
            "correct_answers": ["4", "Four"],
            "incorrect_answers": ["5", "3"]
        }
        
        prop_set = adapter.adapt_single(item)
        
        # Should only include correct answers
        assert len(prop_set.propositions) == 2
        assert all("incorrect" not in p.metadata for p in prop_set.propositions)

    def test_convert_with_categories(self):
        """Test conversion preserves categories."""
        adapter = TruthfulQAAdapter()
        
        item = {
            "question": "Test question?",
            "best_answer": "Test answer",
            "correct_answers": ["Test answer"],
            "incorrect_answers": ["Wrong answer"],
            "category": "Science",
            "source": "test"
        }
        
        prop_set = adapter.adapt_single(item)
        
        assert prop_set.metadata["category"] == "Science"
        assert prop_set.metadata["source"] == "test"

    def test_convert_missing_fields(self):
        """Test handling of missing fields."""
        adapter = TruthfulQAAdapter()
        
        # Missing correct_answers
        item = {
            "question": "Test?",
            "best_answer": "Answer"
        }
        
        prop_set = adapter.adapt_single(item)
        # Should still work with just best_answer
        assert len(prop_set.propositions) >= 1

    def test_convert_empty_answers(self):
        """Test with empty answer lists."""
        adapter = TruthfulQAAdapter()
        
        item = {
            "question": "Empty test?",
            "best_answer": "",
            "correct_answers": [],
            "incorrect_answers": []
        }
        
        prop_set = adapter.adapt_single(item)
        assert len(prop_set.propositions) == 0