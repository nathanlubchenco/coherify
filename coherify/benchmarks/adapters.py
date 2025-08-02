"""
Adapters for common benchmark formats to PropositionSet.
This is the key to easy benchmark integration.
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod

from coherify.core.base import PropositionSet, Proposition


class BenchmarkAdapter(ABC):
    """Base adapter for converting benchmark data to PropositionSets."""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
    
    @abstractmethod
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert a single benchmark sample to PropositionSet."""
        pass
    
    def adapt_dataset(self, dataset) -> List[PropositionSet]:
        """Convert entire dataset."""
        return [self.adapt_single(sample) for sample in dataset]


class QABenchmarkAdapter(BenchmarkAdapter):
    """Adapter for QA-style benchmarks (TruthfulQA, SimpleQA, etc.)"""
    
    def __init__(self, 
                 benchmark_name: str,
                 question_key: str = "question",
                 answer_key: str = "answer",
                 multiple_answers_key: Optional[str] = None,
                 segment_answers: bool = True):
        """
        Initialize QA benchmark adapter.
        
        Args:
            benchmark_name: Name of the benchmark
            question_key: Key for question in data dict
            answer_key: Key for answer in data dict
            multiple_answers_key: Key for multiple answers (optional)
            segment_answers: Whether to segment answers into sentences
        """
        super().__init__(benchmark_name)
        self.question_key = question_key
        self.answer_key = answer_key
        self.multiple_answers_key = multiple_answers_key
        self.segment_answers = segment_answers
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert a single QA sample to PropositionSet."""
        question = sample[self.question_key]
        
        if self.multiple_answers_key and self.multiple_answers_key in sample:
            # Multiple answers format (e.g., for self-consistency)
            answers = sample[self.multiple_answers_key]
            return PropositionSet.from_multi_answer(question, answers)
        else:
            # Single answer format
            answer = sample[self.answer_key]
            if self.segment_answers:
                return PropositionSet.from_qa_pair(question, answer)
            else:
                # Treat entire answer as single proposition
                props = [Proposition(text=answer)]
                return PropositionSet(propositions=props, context=question)


class SummarizationBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for summarization benchmarks."""
    
    def __init__(self, 
                 benchmark_name: str,
                 document_key: str = "document",
                 summary_key: str = "summary"):
        super().__init__(benchmark_name)
        self.document_key = document_key
        self.summary_key = summary_key
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert a single summarization sample to PropositionSet."""
        document = sample[self.document_key]
        summary = sample[self.summary_key]
        
        # Treat summary sentences as propositions with document as context
        summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
        props = [Proposition(text=sent) for sent in summary_sentences]
        
        return PropositionSet(propositions=props, context=document)


class MultiTurnDialogueAdapter(BenchmarkAdapter):
    """Adapter for multi-turn dialogue benchmarks."""
    
    def __init__(self, 
                 benchmark_name: str,
                 turns_key: str = "turns",
                 response_key: str = "response"):
        super().__init__(benchmark_name)
        self.turns_key = turns_key
        self.response_key = response_key
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert a dialogue to PropositionSet."""
        turns = sample[self.turns_key]
        
        # Extract all responses in the dialogue
        responses = []
        context_parts = []
        
        for turn in turns:
            if isinstance(turn, dict):
                if self.response_key in turn:
                    responses.append(turn[self.response_key])
                # Build context from previous turns
                context_parts.append(str(turn))
            else:
                responses.append(str(turn))
                context_parts.append(str(turn))
        
        # Create propositions from responses
        props = []
        for response in responses:
            # Segment each response
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            props.extend([Proposition(text=sent) for sent in sentences])
        
        context = " ".join(context_parts)
        return PropositionSet(propositions=props, context=context)


# Pre-configured adapters for common benchmarks
BENCHMARK_ADAPTERS = {
    "truthfulqa": QABenchmarkAdapter(
        "truthfulqa",
        question_key="question",
        answer_key="best_answer",
        multiple_answers_key="correct_answers"
    ),
    "selfcheckgpt": QABenchmarkAdapter(
        "selfcheckgpt",
        question_key="prompt",
        answer_key="original_answer",
        multiple_answers_key="sampled_answers"
    ),
    "xsum": SummarizationBenchmarkAdapter(
        "xsum",
        document_key="document",
        summary_key="summary"
    ),
    "simple_qa": QABenchmarkAdapter(
        "simple_qa",
        question_key="question",
        answer_key="answer"
    ),
}


def get_adapter(benchmark_name: str) -> BenchmarkAdapter:
    """Get pre-configured adapter for a benchmark."""
    if benchmark_name not in BENCHMARK_ADAPTERS:
        raise ValueError(
            f"No adapter configured for benchmark: {benchmark_name}. "
            f"Available: {list(BENCHMARK_ADAPTERS.keys())}"
        )
    return BENCHMARK_ADAPTERS[benchmark_name]


def register_adapter(benchmark_name: str, adapter: BenchmarkAdapter) -> None:
    """Register a new benchmark adapter."""
    BENCHMARK_ADAPTERS[benchmark_name] = adapter