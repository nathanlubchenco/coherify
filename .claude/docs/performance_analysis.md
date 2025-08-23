# Performance Analysis and Operational Data

*Consolidated from OPERATIONAL_ANALYSIS.md and WARNINGS_FIXED.md*

## TruthfulQA Benchmark Performance

### Dataset Characteristics
- **Total Samples**: 817 (validation split)
- **Average Question Length**: 48 characters (~12 tokens)
- **Average Answer Length**: 55 characters (~14 tokens)
- **Categories**: Misconceptions, Biology, Law, Nutrition, etc.
- **Storage**: ~500KB (lightweight, text-only)

### Computational Performance

#### Local Processing (No API)
- **SemanticCoherence**: ~0.1s per sample
- **HybridCoherence**: ~0.2s per sample  
- **EntailmentCoherence**: ~1.5s per sample
- **Full dataset (817 samples)**: 2-20 minutes depending on measures

#### Memory Usage
- **Baseline**: ~200MB (transformers models loaded)
- **Peak**: ~800MB (with multiple measures cached)
- **GPU**: Optional but provides 3-5x speedup for large evaluations

### Warning Resolution

#### Transformer Model Warnings
**Issue**: Repeated FutureWarning messages from transformers library
```
FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0
```

**Solution**: 
- Created `coherify/utils/transformers_utils.py`
- Centralized warning suppression with context managers
- Clean output mode for production use

#### Components Fixed
- ✅ **EntailmentCoherence**: Pipeline creation and prediction calls
- ✅ **APIEnhancedMeasures**: Model loading and inference
- ✅ **Benchmark runners**: Clean console output

### Performance Recommendations

#### For Development
```bash
# Fast iteration with small samples
python examples/run_truthfulqa_benchmark.py --sample-size 10

# Clean output for focus
export TOKENIZERS_PARALLELISM=false
python -c "from coherify.utils.clean_output import enable_clean_output; enable_clean_output()"
```

#### For Production
```bash
# Full evaluation with comprehensive reporting
python examples/comprehensive_benchmark_demo.py --sample-size 100

# Web UI for result analysis
python examples/comprehensive_benchmark_demo.py --use-ui
```

#### Memory Optimization
- Use approximation algorithms for large datasets
- Enable caching for repeated evaluations
- Consider batch processing for memory-constrained environments

### Benchmark Timing Estimates

| Benchmark | Samples | SemanticCoherence | HybridCoherence | EntailmentCoherence |
|-----------|---------|-------------------|------------------|---------------------|
| Small (10) | 10 | 1s | 2s | 15s |
| Medium (100) | 100 | 10s | 20s | 2.5min |
| Full (817) | 817 | 1.4min | 2.7min | 20min |

*Note: Times are approximate and depend on hardware configuration*