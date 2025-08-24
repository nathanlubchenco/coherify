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

## Documentation Directory Structure

The `.claude/docs/` directory contains comprehensive project documentation:

### `.claude/docs/` - Core Documentation
- **`benchmark_references.md`**: Complete reference for all supported benchmarks including TruthfulQA, SelfCheckGPT, FEVER, and FaithBench with paper citations, usage examples, and evaluation guidelines
- **`benchmark_setup_guide.md`**: Step-by-step setup instructions, dependencies, troubleshooting, and quick start commands for running benchmarks
- **`benchmark_implementation_fixes.md`**: Critical fixes needed for benchmark implementations based on paper research - includes TruthfulQA evaluation logic errors, missing SelfCheckGPT methodology, and FEVER evidence chain requirements
- **`ui_development_history.md`**: Complete UI development history, architecture overview, resolved issues, and lessons learned
- **`performance_analysis.md`**: Operational data, timing benchmarks, memory usage, and performance optimization guidelines

### `agent_context/` - Legacy Context (Preserved)
- **`docs/opus_plan.md`**: Original comprehensive project plan and architecture
- **`memory/`**: Persistent analysis and implementation lessons
- **`tasks/`**: Historical task tracking and progress notes, including `benchmark_methodology_fixes.md` with critical implementation tasks

### `docs/` - Public Documentation
- **`README.md`**: User-facing documentation
- **`OPERATIONAL_GUIDE.md`**: Operations and deployment guide
- **`MULTI_FORMAT_BENCHMARKS.md`**: Multi-format benchmark integration guide

**Usage for Claude Code**: Always check `.claude/docs/` first for current project information. Use `agent_context/` for historical context and original planning documents.

**⚠️ Documentation Maintenance Note**: When creating or updating any documentation, always update this directory listing in CLAUDE.md to maintain an accurate reference.

## Notification Hooks

The repository includes notification hooks for Claude Code to provide desktop notifications during development:

### macOS (default - `.claude_hooks.json`)
Uses `terminal-notifier` for reliable desktop notifications with sound:
```json
{
  "hooks": {
    "Notification": [{"matcher": "", "hooks": [{"type": "command", "command": "terminal-notifier -title 'Claude Code - Coherify' -message 'Awaiting your input' -sound Blow"}]}],
    "OnTaskComplete": [{"matcher": "completed", "hooks": [{"type": "command", "command": "terminal-notifier -title 'Claude Code - Coherify' -message 'Task completed successfully' -sound Glass"}]}],
    "OnError": [{"matcher": "error|failed|exception", "hooks": [{"type": "command", "command": "terminal-notifier -title 'Claude Code - Coherify' -message 'An error occurred' -sound Basso"}]}]
  }
}
```

**Prerequisites**: Install terminal-notifier with `brew install terminal-notifier`

### Linux (`.claude_hooks_linux.json`)
For Linux systems, copy `.claude_hooks_linux.json` to `.claude_hooks.json` to use `notify-send` instead of `terminal-notifier`.

### Alternative (`.claude_hooks_alt.json`)
Audio-only version using `echo` and `say` commands for systems where desktop notifications don't work.

## Critical Implementation Notes

- **Modular Design**: Separate coherence calculation from probability estimation to allow mixing different approaches
- **Computational Efficiency**: Implement caching and approximation algorithms from the start
- **Rich Result Objects**: Enable detailed analysis and comparison across different coherence measures
- **Batch Processing**: Support efficient evaluation on entire datasets for benchmark integration
