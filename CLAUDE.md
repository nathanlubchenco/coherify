# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Standards
* You MUST validate your work.  We have extensive error message and debug logging, pay attention to it.
* You MUST NOT mock data outside of tests.
* You MUST NOT provide quiet fallbacks, ERROR loudly instead.
* Do not keep around legacy implimentations "just in case".  Be confident and decisive.
* ALWAYS update docs, readmes, context files and tasks after a unit of work.
* Remember to update tests, data output format and the UIs when changing contracts.

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

### CRITICAL: Three-Stage Research Pipeline

**Research Goal**: Use coherence theory to improve response selection in multi-generation scenarios for better factual accuracy.

**Three mandatory stages**:

1. **Stage 1: Official Baselines** ✅
   - Faithfully reproduce benchmark evaluation methods (GPT-judge, BLEURT, BERTScore, etc.)
   - Validate against published results to ensure implementation correctness
   - This is NOT what we're trying to improve - it's validation that our setup works

2. **Stage 2: K-Pass Majority Voting Baseline**
   - Generate K responses per question/prompt
   - Use simple majority voting to determine final answer
   - This becomes our **fair comparison baseline** for coherence methods
   - Must beat single-generation baseline to be meaningful

3. **Stage 3: Coherence-Enhanced Response Selection** (Our Contribution)
   - Generate K responses at multiple temperatures
   - Use coherence measures for intelligent response selection instead of naive voting
   - **Key insight**: Truth appears more consistently across temperature variations
   - Compare against K-pass majority voting (fair comparison)

**Never skip stages. Never compare single-generation vs K-generation directly.**

Example implementation:
```python
# Stage 1: Validate official baseline
official_score = evaluate_official_benchmark(single_responses, samples)

# Stage 2: K-pass majority voting baseline  
k_responses = generate_k_responses(prompts, k=5, temp=0.7)
majority_voted = majority_vote(k_responses)
k_pass_score = evaluate_official_benchmark(majority_voted, samples)

# Stage 3: Our coherence contribution
coherence_selected = coherence_selection(k_responses, coherence_measure)
coherence_score = evaluate_official_benchmark(coherence_selected, samples)

# Valid comparison: coherence_score vs k_pass_score
improvement = coherence_score - k_pass_score
```

### Benchmark-First Design
All coherence measures must work seamlessly with common benchmark patterns (QA pairs, document-summary pairs, multi-turn dialogues). The `PropositionSet` serves as the universal container that all benchmarks convert to.

### Probability-Free Alternatives Preferred
Traditional probability-based coherence measures face calibration challenges with LLMs. The recommended approach prioritizes:
1. **Semantic Coherence**: Embedding-based similarity measures
2. **Entailment Coherence**: NLI models for logical relationships  
3. **Hybrid Approaches**: Weighted combinations of multiple methods

### Implementation Status (UPDATED 2024-01-24)

**✅ COMPLETE - Core Pipeline**:
- Model runner for actual API calls (`generation/model_runner.py`)
- Official benchmark evaluators with GPT-4 judge (`benchmarks/official/`)
- K-pass generation system (`KPassGenerator`)
- Majority voting selector (`MajorityVotingSelector`)
- Coherence-based selector (`CoherenceSelector`)
- Full 3-stage comparison framework (`run_full_pipeline_comparison.py`)

**🔄 IN PROGRESS - Validation**:
- Testing with real API keys (GPT-4, Claude)
- Validating against published baselines
- Statistical significance testing

**📋 TODO - Extensions**:
- Additional coherence measures (graph-based, temporal)
- More benchmarks (FEVER, SelfCheckGPT, HaluEval)
- Optimization (caching, batching, parallel processing)
- Cost tracking and analysis

## Testing Commands (CRITICAL)

### Quick Pipeline Test
```bash
# With mock data (no API needed)
make benchmark-full-pipeline MODEL=default SAMPLES=5 K_RUNS=3

# With real API (set OPENAI_API_KEY first)
export OPENAI_API_KEY=your-key-here
make benchmark-full-pipeline MODEL=gpt4-mini SAMPLES=20 K_RUNS=5
```

### Individual Stage Testing
```bash
make benchmark-stage1 MODEL=gpt4-mini SAMPLES=20        # Baseline
make benchmark-stage2 MODEL=gpt4-mini SAMPLES=20 K_RUNS=5  # Majority voting
make benchmark-stage3 MODEL=gpt4-mini SAMPLES=20 K_RUNS=5  # Coherence selection
```

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

Use this approach sparingly, most development should occur in the main framework.

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

### `docs/` - Project Documentation
- **`CRITICAL_IMPLEMENTATION_GAPS.md`**: What was wrong with the original implementation and how it was fixed
- **`LEGACY_ARCHITECTURE.md`**: Historical context, original vision, and lessons learned from development
- **`TROUBLESHOOTING.md`**: Common issues, solutions, and debugging tips
- **`MULTI_FORMAT_BENCHMARKS.md`**: Multi-format benchmark integration guide
- **`OPERATIONAL_GUIDE.md`**: Operations and deployment guide

### Project Status Files (Root Directory)
- **`TODO.md`**: Current tasks and project status
- **`CURRENT_STATE.md`**: Quick context on what's working
- **`PROJECT_SUMMARY.md`**: High-level overview of the project and its goals

**Usage for Claude Code**: Check the `docs/` directory for technical documentation and root directory status files for current project state.

**⚠️ Documentation Maintenance Note**: When creating or updating any documentation, always update this directory listing in CLAUDE.md to maintain an accurate reference.

## Critical Implementation Notes

- **Modular Design**: Separate coherence calculation from probability estimation to allow mixing different approaches
- **Computational Efficiency**: Implement caching and approximation algorithms from the start
- **Rich Result Objects**: Enable detailed analysis and comparison across different coherence measures
- **Batch Processing**: Support efficient evaluation on entire datasets for benchmark integration
