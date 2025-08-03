"""Benchmark integration adapters."""

from .adapters import (
    BenchmarkAdapter,
    QABenchmarkAdapter,
    SummarizationBenchmarkAdapter,
    MultiTurnDialogueAdapter,
    get_adapter,
    register_adapter,
    BENCHMARK_ADAPTERS,
)
from .truthfulqa import TruthfulQAAdapter, TruthfulQAEvaluator
from .selfcheckgpt import SelfCheckGPTAdapter, SelfCheckGPTEvaluator

__all__ = [
    "BenchmarkAdapter",
    "QABenchmarkAdapter",
    "SummarizationBenchmarkAdapter",
    "MultiTurnDialogueAdapter",
    "get_adapter",
    "register_adapter",
    "BENCHMARK_ADAPTERS",
    "TruthfulQAAdapter",
    "TruthfulQAEvaluator",
    "SelfCheckGPTAdapter",
    "SelfCheckGPTEvaluator",
]
