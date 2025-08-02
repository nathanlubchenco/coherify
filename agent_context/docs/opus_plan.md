Project Overview
coherify is a Python library that implements formal theories of coherence from philosophy (Shogenji, Olsson, Fitelson) as practical tools for evaluating and reducing AI hallucinations. The library is designed to seamlessly integrate with popular NLP benchmarks and evaluation frameworks.
Core Design Principles

Benchmark-First Design: API designed around common benchmark patterns (question-answer pairs, document-summary pairs, multi-turn dialogues)
Framework Agnostic: Works with any model that can provide text and optionally logits/probabilities
Efficient Computation: Approximation algorithms for NP-hard coherence calculations
Composable Metrics: Mix and match different coherence measures
Easy Integration: Drop-in compatibility with HuggingFace datasets, EleutherAI eval harness, and custom evaluation loops

Repository Structure
coherify/
├── README.md
├── setup.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE (MIT)
├── docs/
│   ├── philosophy_background.md
│   ├── api_reference.md
│   └── benchmark_integration.md
├── coherify/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base classes
│   │   ├── probability.py       # Probability estimation utilities
│   │   └── types.py            # Type definitions
│   ├── measures/
│   │   ├── __init__.py
│   │   ├── shogenji.py         # Shogenji's coherence measure
│   │   ├── olsson.py           # Olsson's overlap measure
│   │   ├── fitelson.py         # Fitelson's confirmation measure
│   │   └── hybrid.py           # Hybrid/composite measures
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── model_based.py      # LLM-based probability estimation
│   │   ├── frequency.py        # Corpus frequency estimation
│   │   └── ensemble.py         # Ensemble methods
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── adapters.py         # Adapters for common benchmarks
│   │   ├── truthfulqa.py       # TruthfulQA-specific integration
│   │   ├── selfcheckgpt.py     # SelfCheckGPT integration
│   │   └── metrics.py          # Benchmark-specific metrics
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── rag.py              # RAG reranking with coherence
│   │   ├── generation.py       # Generation-time filtering
│   │   └── beam_search.py      # Coherence-guided beam search
│   └── utils/
│       ├── __init__.py
│       ├── caching.py          # Computation caching
│       ├── approximations.py   # Fast approximation algorithms
│       └── visualization.py    # Coherence visualization tools
├── examples/
│   ├── basic_usage.ipynb
│   ├── benchmark_evaluation.ipynb
│   ├── rag_integration.ipynb
│   └── truthfulqa_example.py
├── tests/
│   ├── __init__.py
│   ├── test_measures.py
│   ├── test_estimators.py
│   ├── test_benchmarks.py
│   └── fixtures/
│       └── sample_data.json
└── scripts/
    ├── run_benchmark.py
    ├── evaluate_all.py
    └── download_benchmarks.py
Initial Code Context
coherify/__init__.py
python"""
coherify: Formal Coherence Theory for AI Truth-Seeking

A library implementing philosophical coherence measures (Shogenji, Olsson, Fitelson)
for practical AI applications, particularly hallucination detection and reduction.
"""

__version__ = "0.1.0"

from coherify.core.base import CoherenceResult, PropositionSet
from coherify.measures import ShogunjiCoherence, OlssonCoherence, FitelsonCoherence, HybridCoherence
from coherify.benchmarks import TruthfulQAAdapter, SelfCheckGPTAdapter
from coherify.filters import CoherenceRAGReranker, CoherenceBeamSearch

__all__ = [
    "CoherenceResult",
    "PropositionSet", 
    "ShogunjiCoherence",
    "OlssonCoherence",
    "FitelsonCoherence",
    "HybridCoherence",
    "TruthfulQAAdapter",
    "SelfCheckGPTAdapter",
    "CoherenceRAGReranker",
    "CoherenceBeamSearch",
]
coherify/core/base.py
python"""
Base classes and interfaces for coherence measurement.
Designed to work seamlessly with common NLP benchmark patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

@dataclass
class Proposition:
    """A single proposition with optional probability estimate."""
    text: str
    probability: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PropositionSet:
    """A set of propositions to evaluate for coherence."""
    propositions: List[Proposition]
    context: Optional[str] = None  # For conditional coherence
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def from_qa_pair(cls, question: str, answer: str, 
                     answer_segments: Optional[List[str]] = None) -> "PropositionSet":
        """Create from QA benchmark format."""
        if answer_segments is None:
            # Simple sentence segmentation
            answer_segments = [s.strip() for s in answer.split('.') if s.strip()]
        
        props = [Proposition(text=seg) for seg in answer_segments]
        return cls(propositions=props, context=question)
    
    @classmethod
    def from_multi_answer(cls, question: str, answers: List[str]) -> "PropositionSet":
        """Create from multiple answers to same question (e.g., SelfCheckGPT)."""
        props = [Proposition(text=ans) for ans in answers]
        return cls(propositions=props, context=question)

@dataclass
class CoherenceResult:
    """Result of coherence measurement."""
    score: float
    measure_name: str
    details: Dict[str, Any]
    computation_time: float
    
    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if coherence exceeds a threshold."""
        return self.score > threshold

class CoherenceMeasure(ABC):
    """Abstract base for coherence measures."""
    
    @abstractmethod
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute coherence for a set of propositions."""
        pass
    
    @abstractmethod
    def compute_pairwise(self, prop1: Proposition, prop2: Proposition) -> float:
        """Compute pairwise coherence between two propositions."""
        pass
    
    def batch_compute(self, prop_sets: List[PropositionSet]) -> List[CoherenceResult]:
        """Compute coherence for multiple sets efficiently."""
        return [self.compute(ps) for ps in prop_sets]

class ProbabilityEstimator(ABC):
    """Abstract base for probability estimation."""
    
    @abstractmethod
    def estimate_probability(self, proposition: Proposition, 
                           context: Optional[str] = None) -> float:
        """Estimate probability of a single proposition."""
        pass
    
    @abstractmethod
    def estimate_joint_probability(self, propositions: List[Proposition],
                                 context: Optional[str] = None) -> float:
        """Estimate joint probability of multiple propositions."""
        pass
coherify/benchmarks/adapters.py
python"""
Adapters for common benchmark formats to PropositionSet.
This is the key to easy benchmark integration.
"""

from typing import Dict, List, Any, Optional, Callable
from datasets import Dataset
import json

from coherify.core.base import PropositionSet, Proposition

class BenchmarkAdapter:
    """Base adapter for converting benchmark data to PropositionSets."""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        """Convert a single benchmark sample to PropositionSet."""
        raise NotImplementedError
    
    def adapt_dataset(self, dataset: Dataset) -> List[PropositionSet]:
        """Convert entire dataset."""
        return [self.adapt_single(sample) for sample in dataset]

class QABenchmarkAdapter(BenchmarkAdapter):
    """Adapter for QA-style benchmarks (TruthfulQA, SimpleQA, etc.)"""
    
    def __init__(self, benchmark_name: str,
                 question_key: str = "question",
                 answer_key: str = "answer",
                 multiple_answers_key: Optional[str] = None):
        super().__init__(benchmark_name)
        self.question_key = question_key
        self.answer_key = answer_key
        self.multiple_answers_key = multiple_answers_key
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        question = sample[self.question_key]
        
        if self.multiple_answers_key and self.multiple_answers_key in sample:
            # Multiple answers format (e.g., for self-consistency)
            answers = sample[self.multiple_answers_key]
            return PropositionSet.from_multi_answer(question, answers)
        else:
            # Single answer format
            answer = sample[self.answer_key]
            return PropositionSet.from_qa_pair(question, answer)

class SummarizationBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for summarization benchmarks."""
    
    def __init__(self, benchmark_name: str,
                 document_key: str = "document",
                 summary_key: str = "summary"):
        super().__init__(benchmark_name)
        self.document_key = document_key
        self.summary_key = summary_key
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        document = sample[self.document_key]
        summary = sample[self.summary_key]
        
        # Treat summary sentences as propositions with document as context
        summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
        props = [Proposition(text=sent) for sent in summary_sentences]
        
        return PropositionSet(propositions=props, context=document)

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
    # Add more as needed
}

def get_adapter(benchmark_name: str) -> BenchmarkAdapter:
    """Get pre-configured adapter for a benchmark."""
    if benchmark_name not in BENCHMARK_ADAPTERS:
        raise ValueError(f"No adapter configured for benchmark: {benchmark_name}")
    return BENCHMARK_ADAPTERS[benchmark_name]
coherify/measures/shogenji.py
python"""
Implementation of Shogenji's coherence measure.
"""

import time
from typing import List, Optional
import numpy as np

from coherify.core.base import (
    CoherenceMeasure, CoherenceResult, PropositionSet, 
    Proposition, ProbabilityEstimator
)

class ShogunjiCoherence(CoherenceMeasure):
    """
    Shogenji's coherence measure: C_S(S) = P(H1 ∧ H2 ∧ ... ∧ Hn) / ∏P(Hi)
    
    Measures mutual support via joint vs. independent probability.
    Values > 1 indicate positive coherence.
    """
    
    def __init__(self, probability_estimator: ProbabilityEstimator, 
                 smoothing: float = 1e-6):
        self.estimator = probability_estimator
        self.smoothing = smoothing
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        start_time = time.time()
        
        # Estimate individual probabilities
        individual_probs = []
        for prop in prop_set.propositions:
            if prop.probability is None:
                prop.probability = self.estimator.estimate_probability(
                    prop, context=prop_set.context
                )
            individual_probs.append(prop.probability + self.smoothing)
        
        # Estimate joint probability
        joint_prob = self.estimator.estimate_joint_probability(
            prop_set.propositions, context=prop_set.context
        ) + self.smoothing
        
        # Compute Shogenji score
        product_individual = np.prod(individual_probs)
        score = joint_prob / product_individual
        
        computation_time = time.time() - start_time
        
        return CoherenceResult(
            score=score,
            measure_name="Shogenji",
            details={
                "joint_probability": joint_prob,
                "product_individual": product_individual,
                "individual_probabilities": individual_probs,
                "num_propositions": len(prop_set.propositions)
            },
            computation_time=computation_time
        )
    
    def compute_pairwise(self, prop1: Proposition, prop2: Proposition) -> float:
        """Compute Shogenji coherence for a pair."""
        prop_set = PropositionSet(propositions=[prop1, prop2])
        result = self.compute(prop_set)
        return result.score
setup.py
pythonfrom setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coherify",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Formal coherence theory for AI truth-seeking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coherify",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
        ],
        "benchmarks": [
            "lm-eval>=0.3.0",  # EleutherAI eval harness
            "evaluate>=0.3.0",  # HuggingFace evaluate
        ],
    },
)
README.md
markdown# coherify

Formal coherence theory for AI truth-seeking. This library implements philosophical coherence measures (Shogenji, Olsson, Fitelson) as practical tools for evaluating and reducing AI hallucinations.

## Installation

```bash
pip install coherify
For development:
bashgit clone https://github.com/yourusername/coherify.git
cd coherify
pip install -e ".[dev,benchmarks]"
Quick Start
pythonfrom coherify import ShogunjiCoherence, PropositionSet
from coherify.estimators import ModelBasedEstimator

# Initialize with your preferred model
estimator = ModelBasedEstimator(model_name="gpt2")
coherence = ShogunjiCoherence(estimator)

# Evaluate coherence of an answer
prop_set = PropositionSet.from_qa_pair(
    question="What is the capital of France?",
    answer="The capital of France is Paris. Paris is located in northern France."
)

result = coherence.compute(prop_set)
print(f"Coherence score: {result.score:.3f}")
Benchmark Integration
pythonfrom datasets import load_dataset
from coherify.benchmarks import get_adapter

# Load TruthfulQA
dataset = load_dataset("truthfulqa", "generation")
adapter = get_adapter("truthfulqa")

# Convert to PropositionSets
prop_sets = adapter.adapt_dataset(dataset["validation"])

# Evaluate coherence
results = coherence.batch_compute(prop_sets)
Key Features

Multiple coherence measures: Shogenji, Olsson, Fitelson, and hybrid approaches
Benchmark integration: Pre-built adapters for TruthfulQA, SelfCheckGPT, etc.
Efficient computation: Approximation algorithms for large proposition sets
Flexible probability estimation: Model-based, frequency-based, or custom
RAG integration: Coherence-aware reranking and generation filtering

Documentation
See the docs folder for:

Philosophy Background
API Reference
Benchmark Integration Guide

Citation
If you use this library in your research, please cite:
bibtex@software{coherify2024,
  title = {coherify: Formal Coherence Theory for AI Truth-Seeking},
  year = {2024},
  url = {https://github.com/yourusername/coherify}
}
License
MIT License. See LICENSE for details.

### Key Design Decisions for Easy Benchmark Integration

1. **PropositionSet as Universal Container**: All benchmarks convert to this format, making coherence measures benchmark-agnostic.

2. **Adapter Pattern**: Each benchmark gets an adapter that knows how to convert its specific format to PropositionSets.

3. **Flexible Probability Estimation**: Separated from coherence calculation, allowing different strategies per benchmark.

4. **Batch Processing**: Built-in support for efficient evaluation on entire datasets.

5. **Result Objects**: Rich result objects that can be easily analyzed and compared across benchmarks.

This design allows you to:
- Easily add new benchmarks by writing a simple adapter
- Mix and match coherence measures with different benchmarks
- Integrate with existing evaluation frameworks
- Cache computations for efficiency
- Visualize and analyze results consistently

The coding agent should start by implementing the core base classes and one complete coherence measure (Shogenji) with a simple probability estimator, then expand to other measures and more sophisticated features.RetryNLhow do you think we should get probability estimates for the propositions or is there an approach we can take that doesn't rely on having probability estimates for propositions?EditGreat question! This is actually one of the most critical design decisions for making coherence measures practical. Let me outline several approaches, from most theoretically sound to most practical:
Approaches for Probability Estimation
1. Model-Based Confidence Estimation
Use the language model's own confidence/logits as probability proxies:
pythonclass ModelConfidenceEstimator(ProbabilityEstimator):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def estimate_probability(self, proposition: Proposition, context: Optional[str] = None) -> float:
        # Use the model's likelihood of generating the proposition
        prompt = f"{context}\nIs the following true? {proposition.text}\nAnswer:"
        
        # Get logits for "Yes" vs "No" tokens
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            yes_logit = logits[self.tokenizer.encode("Yes")[0]]
            no_logit = logits[self.tokenizer.encode("No")[0]]
            
            # Convert to probability via softmax
            prob = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)[1].item()
        
        return prob
Pros: Direct, uses model's "beliefs"
Cons: LLMs are poorly calibrated; overconfident on false statements
2. Ensemble Consistency Estimation
Use agreement across multiple samples as probability proxy:
pythonclass ConsistencyEstimator(ProbabilityEstimator):
    def __init__(self, model, num_samples=5):
        self.model = model
        self.num_samples = num_samples
    
    def estimate_probability(self, proposition: Proposition, context: Optional[str] = None) -> float:
        # Sample multiple times and check consistency
        agreements = 0
        
        for _ in range(self.num_samples):
            # Generate with temperature to get variety
            response = self.model.generate(
                f"Given: {context}\nEvaluate: {proposition.text}\nTrue or False?",
                temperature=0.7
            )
            if "true" in response.lower():
                agreements += 1
        
        return agreements / self.num_samples
Pros: More robust than single-shot confidence
Cons: Computationally expensive
3. Retrieval-Based Estimation
Use external knowledge sources:
pythonclass RetrievalEstimator(ProbabilityEstimator):
    def __init__(self, retriever, fact_checker):
        self.retriever = retriever
        self.fact_checker = fact_checker
    
    def estimate_probability(self, proposition: Proposition, context: Optional[str] = None) -> float:
        # Retrieve relevant documents
        docs = self.retriever.search(proposition.text, k=5)
        
        # Check support in retrieved documents
        support_scores = []
        for doc in docs:
            score = self.fact_checker.check_entailment(doc.text, proposition.text)
            support_scores.append(score)
        
        # Weight by retrieval confidence
        weighted_support = sum(s * d.score for s, d in zip(support_scores, docs))
        return weighted_support / sum(d.score for d in docs)
Pros: Grounded in external evidence
Cons: Requires good retriever and fact-checker
Alternative: Coherence Without Explicit Probabilities
4. Semantic Similarity-Based Coherence
Instead of probabilities, use semantic relationships:
pythonclass SemanticCoherence(CoherenceMeasure):
    def __init__(self, encoder_model):
        self.encoder = encoder_model
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        # Encode all propositions
        embeddings = [self.encoder.encode(p.text) for p in prop_set.propositions]
        
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Coherence = average pairwise similarity (excluding diagonal)
        n = len(embeddings)
        total_sim = (similarity_matrix.sum() - n) / (n * (n - 1))
        
        # Alternative: minimum similarity (weakest link)
        # min_sim = similarity_matrix[np.triu_indices(n, k=1)].min()
        
        return CoherenceResult(
            score=total_sim,
            measure_name="SemanticCoherence",
            details={"similarity_matrix": similarity_matrix}
        )
5. Entailment-Based Coherence
Use NLI models to check logical relationships:
pythonclass EntailmentCoherence(CoherenceMeasure):
    def __init__(self, nli_model):
        self.nli = nli_model
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        props = prop_set.propositions
        n = len(props)
        
        # Check all directed pairs for entailment/contradiction
        entailments = 0
        contradictions = 0
        neutrals = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    label = self.nli.predict(props[i].text, props[j].text)
                    if label == "entailment":
                        entailments += 1
                    elif label == "contradiction":
                        contradictions += 1
                    else:
                        neutrals += 1
        
        # Coherence score based on entailment/contradiction ratio
        if contradictions > 0:
            score = -contradictions / (n * (n - 1))  # Negative for contradictions
        else:
            score = entailments / (n * (n - 1))  # Positive for entailments
        
        return CoherenceResult(
            score=score,
            measure_name="EntailmentCoherence",
            details={
                "entailments": entailments,
                "contradictions": contradictions,
                "neutrals": neutrals
            }
        )
6. Graph-Based Coherence (Inspired by Thagard)
Build a constraint satisfaction network:
pythonclass GraphCoherence(CoherenceMeasure):
    def __init__(self, similarity_threshold=0.5):
        self.threshold = similarity_threshold
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        # Build coherence graph
        G = nx.Graph()
        props = prop_set.propositions
        
        # Add nodes
        for i, prop in enumerate(props):
            G.add_node(i, text=prop.text)
        
        # Add edges based on semantic similarity
        embeddings = [get_embedding(p.text) for p in props]
        
        for i in range(len(props)):
            for j in range(i+1, len(props)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > self.threshold:
                    G.add_edge(i, j, weight=sim)
        
        # Coherence metrics from graph theory
        if len(G.edges) == 0:
            clustering = 0
            density = 0
        else:
            clustering = nx.average_clustering(G, weight='weight')
            density = nx.density(G)
        
        # Combined score
        score = (clustering + density) / 2
        
        return CoherenceResult(
            score=score,
            measure_name="GraphCoherence",
            details={
                "clustering_coefficient": clustering,
                "density": density,
                "num_edges": len(G.edges)
            }
        )
My Recommendation: Hybrid Approach
For your implementation, I suggest a hybrid approach that doesn't rely solely on probability estimates:
pythonclass HybridCoherence(CoherenceMeasure):
    def __init__(self, 
                 semantic_weight: float = 0.3,
                 entailment_weight: float = 0.3,
                 consistency_weight: float = 0.4):
        self.semantic = SemanticCoherence()
        self.entailment = EntailmentCoherence()
        self.consistency = ConsistencyEstimator()
        
        self.weights = {
            'semantic': semantic_weight,
            'entailment': entailment_weight,
            'consistency': consistency_weight
        }
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        results = {
            'semantic': self.semantic.compute(prop_set),
            'entailment': self.entailment.compute(prop_set),
            'consistency': self.consistency.compute(prop_set)
        }
        
        # Weighted combination
        combined_score = sum(
            self.weights[name] * result.score 
            for name, result in results.items()
        )
        
        return CoherenceResult(
            score=combined_score,
            measure_name="HybridCoherence",
            details={
                'component_scores': {k: v.score for k, v in results.items()},
                'weights': self.weights
            }
        )
Why This Approach Works Better

No explicit probability calibration needed - Semantic similarity and entailment are more reliable than LLM probability estimates
Captures different aspects of coherence - Semantic (topical), logical (entailment), and behavioral (consistency)
Computationally tractable - Can cache embeddings and reuse
Benchmark-friendly - These metrics work well with existing benchmark formats
Interpretable - Can inspect which component drives the coherence score

For benchmarks like SelfCheckGPT that already generate multiple answers, you can use the consistency directly. For others, you can fall back on semantic/entailment measures.
Would you like me to elaborate on any of these approaches or help design the specific implementation for your use case?RetryClaude can make mistakes. Please double-check responses.
