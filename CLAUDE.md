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

### CRITICAL: Benchmark Fidelity FIRST, Enhancement SECOND
**This is non-negotiable**: We MUST faithfully reproduce existing benchmarks with their official evaluation methods BEFORE attempting any coherence-based improvements.

Every benchmark implementation MUST have:
1. **Official Evaluation**: Exact reproduction using original metrics (GPT-judge, BLEURT, BERTScore, etc.)
2. **Baseline Establishment**: Verify our reproduction matches published results
3. **Coherence Enhancement**: ONLY after baselines are established, show improvements

Example structure:
```python
class TruthfulQABenchmark:
    def evaluate_official(self, predictions):
        """Use GPT-judge or BLEURT - MUST match original paper"""
        pass
    
    def evaluate_with_coherence(self, predictions):
        """Our improvement - ONLY after official works"""
        pass
```

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

### Code Organization and Testing Guidelines

#### Temporary Scripts and Experimental Code
For quick testing, debugging, prototyping, and experimental code during development:

```bash
# Use the .tmp/ directory for all temporary scripts
.tmp/test_new_feature.py          # Quick feature tests
.tmp/debug_benchmark_issue.py     # Debugging scripts  
.tmp/scratch_coherence_idea.py    # Experimental code
.tmp/prototype_xyz.py             # Feature prototypes
```

**Guidelines**:
- ✅ **Use `.tmp/` for**: One-off tests, debugging, prototypes, scratch code
- ✅ **Automatic cleanup**: Entire `.tmp/` directory is git-ignored
- ✅ **No formal structure**: Write quick and dirty code as needed
- ❌ **Don't put temporary files in**: Root directory, `tests/`, or any tracked location

**File Naming Conventions**:
- `test_*.py` - Quick test scripts
- `debug_*.py` - Debugging scripts  
- `scratch_*.py` - Experimental/scratch code
- `prototype_*.py` - Prototyping new features
- `*_temp.py`, `*_scratch.py` - Any temporary files

#### Formal Testing Structure
For permanent, structured tests that are part of the codebase:

```bash
tests/                           # Formal test directory
├── test_measures.py            # Core functionality tests
├── test_benchmarks.py          # Benchmark integration tests
├── test_utils.py              # Utility function tests
└── test_*_validation.py       # Benchmark validation tests
```

**Guidelines**:
- ✅ **Use `tests/` for**: Permanent tests, CI/CD integration, regression testing
- ✅ **Follow conventions**: Proper structure, docstrings, assertions
- ✅ **Git tracked**: These tests are part of the codebase
- ✅ **Run with pytest**: `pytest tests/`

This separation keeps the main codebase clean while providing space for necessary development experimentation.

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
