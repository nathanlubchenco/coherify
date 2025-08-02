# Coherify Project Analysis & Implementation Guide

## Project Overview
Coherify is a Python library implementing formal coherence theories from philosophy (Shogenji, Olsson, Fitelson) as practical tools for AI hallucination detection and reduction. Key focus on benchmark integration and practical applicability.

## Core Architecture

### Layered Design
- **Core**: Abstract classes, data structures (`base.py`, `types.py`, `probability.py`)
- **Measures**: Coherence implementations (`shogenji.py`, `olsson.py`, `fitelson.py`, `hybrid.py`)
- **Estimators**: Probability estimation methods (`model_based.py`, `frequency.py`, `ensemble.py`)
- **Benchmarks**: Integration adapters (`adapters.py`, `truthfulqa.py`, `selfcheckgpt.py`)
- **Filters**: Applications (`rag.py`, `generation.py`, `beam_search.py`)
- **Utils**: Supporting tools (`caching.py`, `approximations.py`, `visualization.py`)

### Key Classes
- `Proposition`: Single statement with optional probability
- `PropositionSet`: Collection of propositions with context (universal container)
- `CoherenceMeasure`: Abstract base for coherence calculations
- `ProbabilityEstimator`: Abstract base for probability estimation
- `BenchmarkAdapter`: Pattern for converting benchmark formats to PropositionSets

## Coherence Theory Concepts

### Traditional Measures
- **Shogenji**: `C_S(S) = P(H1 ∧ H2 ∧ ... ∧ Hn) / ∏P(Hi)` (joint vs independent probability)
- **Olsson**: Overlap measure focusing on mutual support
- **Fitelson**: Confirmation-based coherence

### Probability-Free Alternatives (RECOMMENDED)
1. **Semantic Coherence**: Cosine similarity between embeddings
2. **Entailment Coherence**: NLI models for logical relationships
3. **Graph Coherence**: Constraint satisfaction networks
4. **Hybrid Coherence**: Weighted combination (semantic 30% + entailment 30% + consistency 40%)

## Implementation Strategy

### Phase 1: Foundation (START HERE)
1. Core base classes (`Proposition`, `PropositionSet`, `CoherenceMeasure`)
2. Semantic coherence measure (embedding-based)
3. Basic QA benchmark adapter
4. Minimal working example

### Phase 2: Core Measures
1. Entailment-based coherence with NLI
2. Hybrid coherence combining approaches
3. TruthfulQA and SelfCheckGPT adapters

### Phase 3: Advanced
1. Traditional Shogenji with simple probability estimator
2. Caching and approximation utilities
3. RAG integration applications

### Phase 4: Extensions
1. Graph-based measures
2. Visualization tools
3. Additional benchmark integrations

## Key Design Decisions

### Benchmark Integration
- Universal `PropositionSet` container for all benchmarks
- Adapter pattern for format conversion
- Pre-configured adapters for common benchmarks
- Batch processing support

### Probability Estimation Challenges
- LLMs poorly calibrated for probability estimates
- Semantic similarity more reliable than model confidence
- Hybrid approaches capture multiple coherence aspects
- Caching essential for computational efficiency

## Critical Insights
1. **Avoid probability dependence**: Semantic + entailment approaches more practical than traditional probability-based measures
2. **Benchmark-first design**: All coherence measures must work seamlessly with common benchmark formats
3. **Modular architecture**: Separate concerns to allow mixing coherence measures with different benchmarks
4. **Rich result objects**: Enable analysis and comparison across different approaches
5. **Computational efficiency**: Approximation algorithms and caching crucial for practical use

## Next Steps
Start with Phase 1 implementation focusing on semantic coherence and basic benchmark integration before attempting traditional probability-based measures.