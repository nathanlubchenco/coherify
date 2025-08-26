# Legacy Architecture & Historical Context

This document preserves important historical decisions and lessons from the original Coherify development.

## Original Vision (from opus_plan.md)

Coherify was conceived as a Python library implementing formal theories of coherence from philosophy (Shogenji, Olsson, Fitelson) as practical tools for evaluating and reducing AI hallucinations. The core insight was that philosophical coherence theory could be transformed into practical AI evaluation tools.

### Original Design Principles
1. **Benchmark-First Design**: API designed around common benchmark patterns
2. **Framework Agnostic**: Works with any model providing text and optionally logits
3. **Efficient Computation**: Approximation algorithms for NP-hard coherence calculations
4. **Composable Metrics**: Mix and match different coherence measures
5. **Easy Integration**: Drop-in compatibility with evaluation frameworks

## Critical Methodology Fix (January 2024)

### The Fundamental Misunderstanding

**Original Approach (❌ Wrong)**:
- Treated coherence as a replacement metric for benchmarks
- Used ground truth answers instead of generating responses
- Evaluated with fuzzy string matching
- No comparison baseline (unfair single vs K comparison)

**Corrected Approach (✅ Right)**:
- Coherence as a selection mechanism among multiple responses
- Actually generates responses using real LLM APIs
- Uses GPT-4 judge for proper evaluation
- Fair 3-stage comparison (baseline → majority → coherence)

### The 3-Stage Pipeline

```
Stage 1: Single response baseline
Stage 2: K-pass with majority voting
Stage 3: K-pass with coherence selection
```

This correction was critical - coherence is a BOTH/AND not an either/or. We must BOTH reproduce benchmarks AND improve them with coherence.

## Key Architectural Decisions

### 1. Probability-Free Approach
**Decision**: Prioritize semantic similarity and entailment over traditional probability-based measures  
**Rationale**: LLMs are poorly calibrated for probability estimates  
**Impact**: More reliable, practical, and computationally efficient evaluation  
**Lesson**: Pragmatic adaptations often work better than pure theory

### 2. Universal PropositionSet Container
**Decision**: All benchmarks convert to standardized PropositionSet format  
**Rationale**: Enables coherence measures to be benchmark-agnostic  
**Impact**: Seamless integration with any evaluation framework  
**Lesson**: Good abstractions pay dividends in extensibility

### 3. Hybrid Coherence as Primary
**Decision**: Combine semantic + entailment rather than single measures  
**Rationale**: Different aspects of coherence require different evaluation  
**Impact**: More robust and nuanced coherence evaluation  
**Lesson**: Multi-dimensional approaches outperform single metrics

### 4. Caching-First Performance
**Decision**: Built comprehensive caching from early stages  
**Rationale**: Coherence computation involves expensive ML inference  
**Impact**: 10,000x+ speedup on repeated computations  
**Lesson**: Performance optimizations should be architectural

## Implementation Lessons Learned

### What Worked Exceptionally Well
1. **Modular architecture**: Easy to extend and experiment
2. **Comprehensive caching**: Massive performance gains
3. **Benchmark adapters**: Seamless integration
4. **Response selection**: Clear improvement mechanism
5. **Progress monitoring**: Essential for long-running evaluations

### What Required Major Fixes
1. **Research methodology**: Complete restructure to 3-stage pipeline
2. **Model generation**: Had to build real API integration
3. **Evaluation methods**: Switched from string matching to GPT-4 judge
4. **Fair comparison**: K-to-K comparison, not 1-to-K
5. **Temperature handling**: Parameter passing conflicts

## Technical Insights

### Coherence Evaluation
- **Semantic coherence** captures topical consistency effectively
- **Entailment coherence** detects logical contradictions
- **Hybrid approaches** provide more robust evaluation
- **Contradiction detection** is more valuable than entailment for hallucination prevention

### Benchmark Integration
- **Generate real responses**: Can't evaluate without actual generation
- **Use proper judges**: String matching gives false results
- **Fair comparison essential**: Must compare apples to apples
- **Progress monitoring critical**: Long evaluations need visibility

## Development Process Insights

1. **Phase-based development** keeps complexity manageable
2. **TodoWrite tool** excellent for tracking complex implementations
3. **Incremental testing** prevents integration issues
4. **Working examples** crucial for validating design
5. **Clear documentation** prevents methodology misunderstandings

## Future Directions

Based on our experience, future work should focus on:
1. **Statistical validation** with N=100+ samples
2. **Optimal K exploration** (3, 5, 10, 20 responses)
3. **Coherence measure tuning** for different content types
4. **Extension to more benchmarks** (FEVER, HaluEval, etc.)
5. **Production optimization** (parallel processing, better caching)

---

*This document preserves the key historical context from the agent_context directory. The original detailed plans and task tracking have been consolidated here for reference.*