"""
coherify: Formal Coherence Theory for AI Truth-Seeking

A library implementing philosophical coherence measures (Shogenji, Olsson, Fitelson)
for practical AI applications, particularly hallucination detection and reduction.
"""

__version__ = "0.1.0"

from coherify.approximation import (
    ClusterBasedApproximator,
    DiversitySampler,
    HierarchicalCoherenceApproximator,
    ImportanceSampler,
    IncrementalCoherenceTracker,
    RandomSampler,
    SamplingBasedApproximator,
    StratifiedSampler,
    StreamingCoherenceEstimator,
)
from coherify.benchmarks.adapters import QABenchmarkAdapter, get_adapter
from coherify.benchmarks.api_enhanced import (
    APIBenchmarkConfig,
    APIBenchmarkEvaluator,
    APIEnhancedQAAdapter,
)
from coherify.benchmarks.faithbench_adapter import (
    FaithBenchAdapter,
    FaithBenchConfig,
    FaithfulnessCoherence,
)
from coherify.benchmarks.fever_adapter import (
    EvidenceBasedCoherence,
    FEVERAdapter,
    FEVERConfig,
)
from coherify.benchmarks.multi_format_adapters import (
    GSM8KAdapter,
    HellaSwagAdapter,
    MMLUAdapter,
    MultiResponseBenchmarkConfig,
)
from coherify.benchmarks.selfcheckgpt import SelfCheckGPTAdapter, SelfCheckGPTEvaluator
from coherify.benchmarks.truthfulqa import (
    EnhancedTruthfulQAEvaluator,
    TruthfulQAAdapter,
    TruthfulQAEvaluator,
)
from coherify.core.base import CoherenceResult, Proposition, PropositionSet
from coherify.evaluators import KRunBenchmarkEvaluator, MajorityVotingEvaluator
from coherify.generation import (
    AdaptiveCoherenceFilter,
    CoherenceBeamSearchDecoder,
    CoherenceFilter,
    CoherenceGuidedBeamSearch,
    CoherenceGuidedGenerator,
    MultiStageFilter,
    StreamingCoherenceGuide,
)
from coherify.measures.api_enhanced import (
    APICoherenceConfig,
    APIEnhancedEntailmentCoherence,
    APIEnhancedHybridCoherence,
    APIEnhancedSemanticCoherence,
)
from coherify.measures.entailment import EntailmentCoherence
from coherify.measures.hybrid import AdaptiveHybridCoherence, HybridCoherence
from coherify.measures.multi_response import (
    MultiResponseCoherenceMeasure,
    MultiResponseConfig,
    SelfConsistencyCoherence,
    TemperatureVarianceCoherence,
)
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.shogenji import (
    ConfidenceBasedProbabilityEstimator,
    ShogunjiCoherence,
)
from coherify.providers import (
    AnthropicProvider,
    ModelProvider,
    ModelResponse,
    OpenAIProvider,
    ProviderManager,
    get_provider,
    get_provider_manager,
    setup_providers,
)
from coherify.rag import CoherenceGuidedRetriever, CoherenceRAG, CoherenceReranker
from coherify.reporting import (
    BenchmarkContext,
    BenchmarkReport,
    BenchmarkReporter,
    ErrorInfo,
    ExampleResult,
    ModelInfo,
)
from coherify.ui import ResultViewer, start_result_server
from coherify.utils.caching import CachedEncoder, EmbeddingCache
from coherify.utils.clean_output import (
    clean_output,
    enable_clean_mode,
    enable_clean_output,
)
from coherify.utils.visualization import CoherenceAnalyzer, CoherenceVisualizer

__all__ = [
    "CoherenceResult",
    "Proposition",
    "PropositionSet",
    "SemanticCoherence",
    "EntailmentCoherence",
    "HybridCoherence",
    "AdaptiveHybridCoherence",
    "ShogunjiCoherence",
    "ConfidenceBasedProbabilityEstimator",
    "QABenchmarkAdapter",
    "get_adapter",
    "TruthfulQAAdapter",
    "TruthfulQAEvaluator",
    "EnhancedTruthfulQAEvaluator",
    "SelfCheckGPTAdapter",
    "SelfCheckGPTEvaluator",
    "CachedEncoder",
    "EmbeddingCache",
    "CoherenceVisualizer",
    "CoherenceAnalyzer",
    "CoherenceReranker",
    "CoherenceRAG",
    "CoherenceGuidedRetriever",
    "RandomSampler",
    "StratifiedSampler",
    "DiversitySampler",
    "ImportanceSampler",
    "SamplingBasedApproximator",
    "ClusterBasedApproximator",
    "HierarchicalCoherenceApproximator",
    "IncrementalCoherenceTracker",
    "StreamingCoherenceEstimator",
    "CoherenceGuidedBeamSearch",
    "CoherenceBeamSearchDecoder",
    "CoherenceFilter",
    "AdaptiveCoherenceFilter",
    "MultiStageFilter",
    "CoherenceGuidedGenerator",
    "StreamingCoherenceGuide",
    "ModelProvider",
    "ModelResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "ProviderManager",
    "get_provider_manager",
    "get_provider",
    "setup_providers",
    "APICoherenceConfig",
    "APIEnhancedSemanticCoherence",
    "APIEnhancedEntailmentCoherence",
    "APIEnhancedHybridCoherence",
    "APIBenchmarkConfig",
    "APIEnhancedQAAdapter",
    "APIBenchmarkEvaluator",
    "enable_clean_output",
    "clean_output",
    "enable_clean_mode",
    "MultiResponseConfig",
    "MultiResponseCoherenceMeasure",
    "TemperatureVarianceCoherence",
    "SelfConsistencyCoherence",
    "MultiResponseBenchmarkConfig",
    "GSM8KAdapter",
    "HellaSwagAdapter",
    "MMLUAdapter",
    "FEVERAdapter",
    "FEVERConfig",
    "EvidenceBasedCoherence",
    "FaithBenchAdapter",
    "FaithBenchConfig",
    "FaithfulnessCoherence",
    "MajorityVotingEvaluator",
    "KRunBenchmarkEvaluator",
    "BenchmarkReport",
    "BenchmarkReporter",
    "BenchmarkContext",
    "ModelInfo",
    "ExampleResult",
    "ErrorInfo",
    "ResultViewer",
    "start_result_server",
]
