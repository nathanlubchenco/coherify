# Benchmark Documentation

This directory contains official README files from the benchmark repositories integrated with Coherify.

## Available Benchmarks

### 📊 [TruthfulQA](TruthfulQA_README.md)
Measures whether language models generate truthful answers, testing tendency to reproduce human falsehoods.
- 817 questions across 38 categories
- Human performance: 94%, GPT-3: 58%

### 🔍 [SelfCheckGPT](SelfCheckGPT_README.md)
Zero-resource black-box hallucination detection through self-consistency checking.
- Multiple detection methods (NLI, BERTScore, QA, etc.)
- Best performance: 93.42% AUC-PR with GPT-3.5

### ✅ [FEVER](FEVER_README.md)
Fact Extraction and VERification against Wikipedia sources.
- 185,441 claims for verification
- Three-way classification with evidence retrieval

### 🎯 [HaluEval](HaluEval_README.md)
Large-scale hallucination evaluation benchmark.
- 35,000 examples (5K general + 30K task-specific)
- Covers QA, dialogue, and summarization

## Integration with Coherify

All benchmarks are integrated using our 3-stage pipeline:

1. **Stage 1**: Single response baseline
2. **Stage 2**: K-pass with majority voting
3. **Stage 3**: K-pass with coherence selection

Expected improvement: 5-10% over baseline through coherence-based selection.

## Adding New Benchmarks

See [BENCHMARK_REFERENCES.md](../BENCHMARK_REFERENCES.md) for integration guidelines.

## Papers and Citations

Each README includes the original paper citations. When using these benchmarks with Coherify, please cite both the original papers and the Coherify framework.