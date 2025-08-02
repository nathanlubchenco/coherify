# coherify

Formal coherence theory for AI truth-seeking. This library implements philosophical coherence measures (Shogenji, Olsson, Fitelson) as practical tools for evaluating and reducing AI hallucinations.

## Installation

```bash
pip install coherify
```

For development:
```bash
git clone https://github.com/nathanlubchenco/coherify.git
cd coherify
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,benchmarks]"
```

## Quick Start

```python
from coherify import SemanticCoherence, PropositionSet

# Initialize coherence measure
coherence = SemanticCoherence()

# Evaluate coherence of an answer
prop_set = PropositionSet.from_qa_pair(
    question="What is the capital of France?",
    answer="The capital of France is Paris. Paris is located in northern France."
)

result = coherence.compute(prop_set)
print(f"Coherence score: {result.score:.3f}")
```

## Benchmark Integration

```python
from coherify.benchmarks import QABenchmarkAdapter

# Create adapter for your benchmark format
adapter = QABenchmarkAdapter("my_benchmark")

# Convert benchmark sample to PropositionSet
sample = {
    "question": "Who wrote Romeo and Juliet?",
    "answer": "William Shakespeare wrote Romeo and Juliet. He was an English playwright."
}

prop_set = adapter.adapt_single(sample)
result = coherence.compute(prop_set)
print(f"Coherence score: {result.score:.3f}")
```

## Key Features

- **Multiple coherence measures**: Semantic similarity-based coherence (more coming soon)
- **Benchmark integration**: Pre-built adapters for common evaluation formats
- **Efficient computation**: Optimized for batch processing on datasets
- **Flexible probability estimation**: Multiple strategies for probability-free coherence
- **Framework agnostic**: Works with any model that can provide text

## Development

```bash
# Run tests
pytest

# Run example
python examples/basic_usage.py

# Code formatting
black coherify/
flake8 coherify/

# Type checking
mypy coherify/
```

## Docker Support

```bash
# Build and run development container
docker-compose up coherify-dev

# Run tests in container
docker-compose up coherify-test

# Run examples in container
docker-compose up coherify-example
```

## License

MIT License. See LICENSE for details.
