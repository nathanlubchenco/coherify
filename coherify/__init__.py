"""
coherify: Formal Coherence Theory for AI Truth-Seeking

A library implementing philosophical coherence measures (Shogenji, Olsson, Fitelson)
for practical AI applications, particularly hallucination detection and reduction.
"""

__version__ = "0.1.0"

from coherify.core.base import CoherenceResult, Proposition, PropositionSet
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.entailment import EntailmentCoherence
from coherify.measures.hybrid import HybridCoherence, AdaptiveHybridCoherence
from coherify.measures.shogenji import ShogunjiCoherence, ConfidenceBasedProbabilityEstimator
from coherify.benchmarks.adapters import QABenchmarkAdapter, get_adapter
from coherify.benchmarks.truthfulqa import TruthfulQAAdapter, TruthfulQAEvaluator
from coherify.benchmarks.selfcheckgpt import SelfCheckGPTAdapter, SelfCheckGPTEvaluator
from coherify.utils.caching import CachedEncoder, EmbeddingCache
from coherify.utils.visualization import CoherenceVisualizer, CoherenceAnalyzer
from coherify.rag import CoherenceReranker, CoherenceRAG, CoherenceGuidedRetriever
from coherify.approximation import (
    RandomSampler, StratifiedSampler, DiversitySampler, ImportanceSampler,
    SamplingBasedApproximator, ClusterBasedApproximator, HierarchicalCoherenceApproximator,
    IncrementalCoherenceTracker, StreamingCoherenceEstimator
)
from coherify.generation import (
    CoherenceGuidedBeamSearch, CoherenceBeamSearchDecoder,
    CoherenceFilter, AdaptiveCoherenceFilter, MultiStageFilter,
    CoherenceGuidedGenerator, StreamingCoherenceGuide
)
from coherify.providers import (
    ModelProvider, ModelResponse, OpenAIProvider, AnthropicProvider,
    ProviderManager, get_provider_manager, get_provider, setup_providers
)
from coherify.measures.api_enhanced import (
    APICoherenceConfig, APIEnhancedSemanticCoherence, 
    APIEnhancedEntailmentCoherence, APIEnhancedHybridCoherence
)
from coherify.benchmarks.api_enhanced import (
    APIBenchmarkConfig, APIEnhancedQAAdapter, APIBenchmarkEvaluator
)
from coherify.utils.clean_output import enable_clean_output, clean_output, enable_clean_mode
from coherify.measures.multi_response import (
    MultiResponseConfig, MultiResponseCoherenceMeasure, 
    TemperatureVarianceCoherence, SelfConsistencyCoherence
)
from coherify.benchmarks.multi_format_adapters import (
    MultiResponseBenchmarkConfig, GSM8KAdapter, HellaSwagAdapter, MMLUAdapter
)

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
]