# NOTE: Very much a work in progress and unverified

# Coherify: Formal Coherence Theory for AI Truth-Seeking

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)

Coherify is a comprehensive Python library that implements philosophical coherence theories (Shogenji, Olsson, Fitelson) as practical tools for AI truth-seeking, hallucination detection, and content quality assessment.

## 🎯 Key Features

### Core Coherence Measures
- **Semantic Coherence**: Using sentence transformers and cosine similarity
- **Entailment Coherence**: Based on Natural Language Inference models
- **Hybrid Coherence**: Combining multiple coherence dimensions
- **Traditional Shogenji**: Classical probability-based coherence measure

### Advanced Capabilities
- **RAG Integration**: Coherence-guided retrieval and reranking
- **Generation Guidance**: Beam search with coherence optimization
- **Scalability**: Approximation algorithms for large proposition sets
- **Real-time Processing**: Incremental and streaming coherence tracking
- **Comprehensive Analysis**: Visualization and reporting tools

### Production Ready
- **Caching System**: 10,000x+ speedups for repeated computations
- **Flexible Architecture**: Modular design for easy integration
- **Quality Assurance**: Comprehensive testing and validation
- **Performance Optimized**: Efficient algorithms for production use

## 🚀 Quick Start

### Installation

```bash
pip install coherify

# Or install with all optional dependencies
pip install coherify[viz,benchmarks]
```

For development:
```bash
git clone https://github.com/nathanlubchenco/coherify.git
cd coherify
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,viz,benchmarks]"
```

### Basic Usage

```python
from coherify import PropositionSet, HybridCoherence

# Create a proposition set
prop_set = PropositionSet.from_qa_pair(
    question="What is machine learning?",
    answer="Machine learning is AI that learns from data. Neural networks process information like brains."
)

# Evaluate coherence
coherence_measure = HybridCoherence()
result = coherence_measure.compute(prop_set)

print(f"Coherence score: {result.score:.3f}")
print(f"Semantic: {result.details['component_scores']['semantic']:.3f}")
print(f"Entailment: {result.details['component_scores']['entailment']:.3f}")
```

## 📚 Core Concepts

### Proposition Sets
The fundamental unit of analysis in Coherify is a `PropositionSet` - a collection of related statements that can be evaluated for coherence.

```python
from coherify import PropositionSet, Proposition

# Manual creation
propositions = [
    Proposition("Machine learning algorithms learn from data."),
    Proposition("Neural networks are inspired by biological brains."),
    Proposition("Deep learning uses multiple layers of neural networks.")
]
prop_set = PropositionSet(propositions)

# From Q&A format (automatic proposition extraction)
prop_set = PropositionSet.from_qa_pair(
    question="How does AI work?",
    answer="AI systems process data using algorithms. Machine learning enables automatic pattern recognition."
)
```

### Coherence Measures

#### Semantic Coherence
Measures coherence based on semantic similarity between propositions.

```python
from coherify import SemanticCoherence

measure = SemanticCoherence()
result = measure.compute(prop_set)
print(f"Semantic coherence: {result.score:.3f}")
```

#### Entailment Coherence
Uses Natural Language Inference to detect logical relationships.

```python
from coherify import EntailmentCoherence

measure = EntailmentCoherence()
result = measure.compute(prop_set)
print(f"Entailment coherence: {result.score:.3f}")
```

#### Hybrid Coherence (Recommended)
Combines semantic and entailment measures for robust evaluation.

```python
from coherify import HybridCoherence

measure = HybridCoherence(
    semantic_weight=0.6,
    entailment_weight=0.4
)
result = measure.compute(prop_set)
print(f"Hybrid coherence: {result.score:.3f}")
```

## 🔍 Practical Applications

### 1. AI Content Quality Assessment

```python
from coherify import CoherenceFilter, HybridCoherence

# Filter AI-generated content by quality
content_filter = CoherenceFilter(
    coherence_measure=HybridCoherence(),
    min_coherence_threshold=0.5
)

ai_outputs = [
    "Machine learning enables computers to learn from data automatically.",
    "The sky is blue. Pizza has cheese. Algorithms are mathematical procedures.",  # Incoherent
    "Neural networks process information using interconnected computational nodes."
]

filter_result = content_filter.filter_candidates(
    context="Artificial Intelligence Explanation",
    candidates=ai_outputs
)

print(f"High-quality outputs: {len(filter_result.passed_candidates)}")
for i, output in enumerate(filter_result.passed_candidates):
    score = filter_result.coherence_scores[i]
    print(f"  {score:.3f}: {output}")
```

### 2. RAG System Optimization

```python
from coherify import CoherenceRAG, CoherenceReranker
from coherify.rag.reranking import PassageCandidate

# Setup coherence-guided RAG
reranker = CoherenceReranker(
    coherence_weight=0.6,
    retrieval_weight=0.4
)

rag_system = CoherenceRAG(reranker=reranker)

# Your retrieval function
def my_retrieval_function(query):
    # Return list of PassageCandidate objects
    return [
        PassageCandidate(text="Machine learning uses algorithms to learn patterns.", score=0.8),
        PassageCandidate(text="Neural networks are computational models.", score=0.7)
    ]

# Retrieve and rerank with coherence
result = rag_system.retrieve_and_rerank(
    query="How does machine learning work?",
    retrieval_function=my_retrieval_function,
    top_k=5
)

# Build optimized context
context = rag_system.build_context(result)
print(f"Optimized context: {context}")
```

### 3. Coherence-Guided Generation

```python
from coherify import CoherenceGuidedGenerator

# Setup generator with coherence guidance
generator = CoherenceGuidedGenerator(
    beam_search_config={
        "coherence_weight": 0.5,
        "lm_weight": 0.5,
        "beam_size": 5
    }
)

# Generate with coherence optimization
text, guidance = generator.generate(
    context="Machine learning and artificial intelligence",
    prompt="Recent advances in",
    return_guidance=True
)

print(f"Generated: {text}")
print(f"Coherence: {guidance[0].coherence_score:.3f}")
```

### 4. Large-Scale Processing

```python
from coherify import ClusterBasedApproximator, RandomSampler, SamplingBasedApproximator

# For large proposition sets (100s-1000s of propositions)
large_prop_set = PropositionSet(many_propositions)

# Clustering-based approximation
cluster_approximator = ClusterBasedApproximator()
result = cluster_approximator.approximate_coherence(large_prop_set, target_clusters=20)

# Sampling-based approximation
sampler = RandomSampler()
sampling_approximator = SamplingBasedApproximator(sampler)
result = sampling_approximator.approximate_coherence(large_prop_set, sample_size=50)

print(f"Approximate coherence: {result.approximate_score:.3f}")
print(f"Speedup: {result.metadata['reduction_ratio']:.1%} computation reduction")
```

### 5. Real-Time Processing

```python
from coherify import IncrementalCoherenceTracker, StreamingCoherenceGuide

# Incremental updates for dynamic content
tracker = IncrementalCoherenceTracker()

propositions = ["AI learns from data.", "Neural networks process information.", "Machine learning automates analysis."]
for prop_text in propositions:
    update = tracker.add_proposition(Proposition(prop_text))
    print(f"Added: {prop_text}")
    print(f"Coherence: {update.new_score:.3f}")

# Streaming guidance for real-time generation
guide = StreamingCoherenceGuide()
guide.start_session("Context about machine learning")

tokens = ["Deep", "learning", "algorithms", "process", "complex", "data"]
for token in tokens:
    guidance = guide.add_token(token)
    if guidance:
        print(f"Guidance: {guidance.coherence_score:.3f} ({guidance.coherence_trend})")
```

## 📊 Visualization and Analysis

```python
from coherify import CoherenceVisualizer, CoherenceAnalyzer

# Setup analysis tools
visualizer = CoherenceVisualizer()
analyzer = CoherenceAnalyzer()

# Compare different measures
measures = [SemanticCoherence(), EntailmentCoherence(), HybridCoherence()]
results = [measure.compute(prop_set) for measure in measures]

# Create comparison plot
fig = visualizer.plot_coherence_scores(
    results, 
    labels=["Semantic", "Entailment", "Hybrid"],
    title="Coherence Measures Comparison"
)

# Comprehensive analysis
comparison = analyzer.compare_measures(prop_set, measures)
print(f"Best measure: {comparison['score_statistics']['max']:.3f}")

# Create detailed report with visualizations
report = analyzer.create_comprehensive_report(prop_set, measures)
print(f"Generated {len(report['figures'])} visualizations")
```

## 🔧 Configuration and Customization

### Caching for Performance

```python
from coherify import CachedEncoder, EmbeddingCache

# Setup caching for embeddings
cache = EmbeddingCache(max_size=10000)
cached_encoder = CachedEncoder(cache=cache)

# Use with coherence measures
measure = SemanticCoherence(encoder=cached_encoder)
# Subsequent computations will be much faster
```

### Custom Coherence Measures

```python
from coherify.core.base import CoherenceMeasure, CoherenceResult

class CustomCoherenceMeasure(CoherenceMeasure):
    def compute(self, prop_set):
        # Your custom coherence logic here
        score = your_coherence_algorithm(prop_set.propositions)
        
        return CoherenceResult(
            score=score,
            measure_name="Custom",
            details={"method": "custom_algorithm"}
        )

# Use your custom measure
custom_measure = CustomCoherenceMeasure()
result = custom_measure.compute(prop_set)
```

## 📈 Performance Guidelines

### Exact Computation
- **Small sets (≤20 propositions)**: Use any coherence measure directly
- **Medium sets (20-50 propositions)**: Consider caching for repeated evaluations
- **Large sets (50+ propositions)**: Use approximation algorithms

### Approximation Strategies
- **Sampling**: Best for diverse, unstructured content
- **Clustering**: Optimal for topically organized content  
- **Incremental**: Ideal for real-time applications
- **Streaming**: Perfect for continuous data processing

### Memory Usage
- **Caching**: ~100MB per 10K cached embeddings
- **Approximation**: Scales with sample/cluster size, not full dataset
- **Streaming**: Bounded memory usage regardless of input size

## 🧪 Examples and Tutorials

The `examples/` directory contains comprehensive demonstrations:

- `basic_usage.py` - Getting started with core functionality
- `phase2_features.py` - Advanced measures and caching
- `phase3_features.py` - Visualization and traditional measures
- `rag_integration.py` - RAG system optimization
- `approximation_algorithms.py` - Large-scale processing
- `coherence_guided_generation.py` - Generation optimization
- `practical_applications.py` - Real-world use cases

## 🏗️ Architecture

```
coherify/
├── core/           # Base abstractions and interfaces
├── measures/       # Coherence measure implementations
├── benchmarks/     # Benchmark integration and adapters
├── utils/          # Caching, visualization, utilities
├── rag/            # RAG integration and optimization
├── approximation/  # Scalability and approximation algorithms
├── generation/     # Generation guidance and filtering
└── providers/      # External API integrations (OpenAI, Anthropic)
```

## 📊 Operational Characteristics

### Quick Performance Reference

| Operation | Time | Cost | Best For |
|-----------|------|------|----------|
| **Local Evaluation** (100 samples) | 44 seconds | FREE | Development, testing |
| **API Enhanced** (100 samples) | 17 minutes | $0.45 | Quality analysis |
| **Full TruthfulQA** (817 samples) | 6 minutes | FREE | Research evaluation |

### Cost-Effective Options

- **Development**: Local processing (FREE, 1-6 minutes)
- **API Testing**: Claude-3 Haiku ($0.03 for full dataset)
- **Research**: GPT-4 Turbo Enhanced ($3.64 for full dataset)

**📋 For detailed cost analysis and planning:** See [Operational Guide](docs/OPERATIONAL_GUIDE.md)

**🧮 Calculate costs for your scenario:**
```bash
python scripts/benchmark_calculator.py --sample-size 100 --comparison
```

### Development Setup

```bash
git clone https://github.com/nathanlubchenco/coherify.git
cd coherify

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,viz,benchmarks]"

# Run tests
pytest tests/

# Run examples
python examples/basic_usage.py
python examples/practical_applications.py

# Code formatting
black coherify/
flake8 coherify/

# Type checking
mypy coherify/
```

## 🐳 Docker Support

```bash
# Build and run development container
docker compose up coherify-dev

# Run tests in container
docker compose up coherify-test

# Run examples in container
docker compose up coherify-example
```

## 📝 Citation

If you use Coherify in your research, please cite:

```bibtex
@software{coherify2024,
  title={Coherify: Formal Coherence Theory for AI Truth-Seeking},
  author={Nathan Lubchenco},
  year={2024},
  url={https://github.com/nathanlubchenco/coherify}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Philosophical coherence theory foundations from Shogenji, Olsson, and Fitelson
- Sentence transformers and Hugging Face ecosystem
- The broader AI safety and alignment research community

## 🔮 Roadmap

### Upcoming Features
- [ ] Multi-modal coherence (text + images)
- [ ] Advanced logical reasoning integration
- [ ] Distributed computing support
- [ ] Domain-specific coherence measures
- [ ] Interactive web interface

### Research Directions
- [ ] Temporal coherence for sequential content
- [ ] Cross-lingual coherence evaluation
- [ ] Causal coherence reasoning
- [ ] Coherence-guided fine-tuning

---

**Ready to enhance your AI systems with formal coherence theory? Get started with Coherify today!** 🚀
