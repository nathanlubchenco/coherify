# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coherify is a Python library implementing formal coherence theories from philosophy (Shogenji, Olsson, Fitelson) as practical tools for AI hallucination detection and reduction. The project emphasizes benchmark-first design and practical applicability over theoretical purity.

## Architecture Overview

The codebase follows a layered architecture with clear separation of concerns:

### Core Layer (`coherify/core/`)
- Abstract base classes and fundamental data structures
- `Proposition`: Single statement with optional probability metadata
- `PropositionSet`: Universal container for collections of propositions with context
- `CoherenceMeasure`: Abstract base for all coherence calculation implementations
- `ProbabilityEstimator`: Abstract base for probability estimation strategies

### Measures Layer (`coherify/measures/`)
- Implementations of different coherence theories
- Traditional probability-based measures (Shogenji, Olsson, Fitelson)
- Probability-free alternatives (semantic, entailment, graph-based)
- Hybrid measures combining multiple approaches

### Integration Layer (`coherify/benchmarks/`)
- Adapter pattern for converting benchmark formats to universal `PropositionSet` containers
- Pre-configured adapters for common benchmarks (TruthfulQA, SelfCheckGPT)
- Enables seamless integration with existing evaluation frameworks

### Application Layer (`coherify/filters/`)
- Practical applications of coherence measures
- RAG reranking, generation-time filtering, coherence-guided beam search

## Key Design Principles

### Benchmark-First Design
All coherence measures must work seamlessly with common benchmark patterns (QA pairs, document-summary pairs, multi-turn dialogues). The `PropositionSet` serves as the universal container that all benchmarks convert to.

### Probability-Free Alternatives Preferred
Traditional probability-based coherence measures face calibration challenges with LLMs. The recommended approach prioritizes:
1. **Semantic Coherence**: Embedding-based similarity measures
2. **Entailment Coherence**: NLI models for logical relationships  
3. **Hybrid Approaches**: Weighted combinations of multiple methods

### Implementation Strategy

**Phase 1 (Foundation)**: Start with core base classes, semantic coherence, and basic QA benchmark adapter before attempting traditional probability-based measures.

**Phase 2 (Core Measures)**: Add entailment-based coherence and hybrid approaches.

**Phase 3 (Advanced)**: Traditional Shogenji implementation with simple probability estimators.

**Phase 4 (Extensions)**: Graph-based measures, visualization tools, additional benchmark integrations.

## Development Environment

### Python Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install in development mode (when setup.py/pyproject.toml exists)
pip install -e ".[dev,benchmarks]"

# Deactivate when done
deactivate
```

### Docker Support
```bash
# Build development container (when Dockerfile exists)
docker build -t coherify:dev .

# Run container with code mounted
docker run -it --rm -v $(pwd):/app coherify:dev

# Run tests in container
docker run --rm -v $(pwd):/app coherify:dev pytest

# Build production container
docker build -t coherify:prod -f Dockerfile.prod .
```

### Development Commands
```bash
# Run tests (when implemented)
pytest

# Run specific test
pytest tests/test_measures.py::test_shogenji_coherence

# Code formatting (when implemented)
black coherify/
flake8 coherify/

# Type checking (when implemented)  
mypy coherify/

# Install package in editable mode
pip install -e .
```

## Agent Context Directory Structure

The `agent_context/` directory contains resources for AI agents working on this project:

### `/agent_context/docs/`
- `opus_plan.md`: Comprehensive project plan with architecture, code examples, and implementation strategy
- Contains detailed specifications for all planned components and design decisions

### `/agent_context/memory/`
- `coherify_analysis.md`: Analysis of project structure, implementation phases, and key insights
- Persistent memory for understanding project context across sessions

### `/agent_context/tasks/`
- Directory for tracking specific implementation tasks and progress
- Can be used to store task breakdowns and implementation notes

**Usage for Claude Code**: Always check these directories first when working on the project to understand the current context, planned architecture, and any previous analysis or decisions.

## Critical Implementation Notes

- **Modular Design**: Separate coherence calculation from probability estimation to allow mixing different approaches
- **Computational Efficiency**: Implement caching and approximation algorithms from the start
- **Rich Result Objects**: Enable detailed analysis and comparison across different coherence measures
- **Batch Processing**: Support efficient evaluation on entire datasets for benchmark integration
