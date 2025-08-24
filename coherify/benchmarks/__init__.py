"""
Benchmark integration adapters.

CRITICAL: All benchmarks MUST provide both:
1. Official evaluation (faithful reproduction) - FIRST
2. Coherence-enhanced evaluation (our improvement) - SECOND

Never skip official baselines. Always validate against published results.
"""

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

# Import complete benchmarks that have BOTH official and coherence
try:
    from .truthfulqa_complete import TruthfulQACompleteBenchmark
    HAS_COMPLETE_BENCHMARKS = True
except ImportError:
    HAS_COMPLETE_BENCHMARKS = False

# Import official evaluators for baseline establishment
try:
    from .official import (
        TruthfulQAOfficialEvaluator,
        # FEVEROfficialEvaluator,  # TODO: Implement
        # SelfCheckGPTOfficialEvaluator,  # TODO: Implement
    )
    HAS_OFFICIAL = True
except ImportError:
    HAS_OFFICIAL = False

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

if HAS_COMPLETE_BENCHMARKS:
    __all__.append("TruthfulQACompleteBenchmark")

if HAS_OFFICIAL:
    __all__.append("TruthfulQAOfficialEvaluator")
