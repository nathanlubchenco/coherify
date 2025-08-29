# Unified Benchmark Runner Migration Guide
*Created: 2025-01-28*
*Last Updated: 2025-01-28*
*Status: In Progress*
*Author: System*

## Purpose
Document the migration from individual benchmark scripts to the unified benchmark runner.

## What Changed

### Old Approach (❌ Deprecated)
- Separate scripts for each benchmark (`run_truthfulqa_benchmark.py`, `run_fever_benchmark.py`)
- Inconsistent interfaces and configurations
- Duplicate code across scripts
- Hard to maintain and extend

### New Approach (✅ Unified)
- Single entry point: `coherify.benchmark_runner`
- Consistent configuration for all benchmarks
- Automatic 3-stage pipeline execution
- Easy to add new benchmarks

## Usage Examples

### Command Line

```bash
# Old way (deprecated)
python examples/run_truthfulqa_benchmark.py --model gpt4-mini --sample-size 50

# New way (unified)
python -m coherify.benchmark_runner truthfulqa --model gpt4-mini --sample-size 50
```

### Makefile Commands

```bash
# Run full 3-stage pipeline
make benchmark-unified BENCHMARK=truthfulqa MODEL=gpt4-mini K_RUNS=5 SAMPLES=50

# Run only baseline
make benchmark-unified-baseline BENCHMARK=fever MODEL=gpt4-mini SAMPLES=100

# Run baseline + majority voting
make benchmark-unified-majority BENCHMARK=truthfulqa MODEL=gpt4-mini K_RUNS=5 SAMPLES=50
```

### Python API

```python
from coherify.benchmark_runner import UnifiedBenchmarkRunner, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    benchmark="truthfulqa",
    model="gpt4-mini",
    k_runs=5,
    sample_size=50,
    coherence_measure="semantic",
    stages=["baseline", "majority", "coherence"]
)

# Run benchmark
runner = UnifiedBenchmarkRunner(config)
results = runner.run()

# Access results
print(f"Baseline: {results['stage1']['score']:.1%}")
print(f"Majority: {results['stage2']['score']:.1%}")
print(f"Coherence: {results['stage3']['score']:.1%}")
```

## Configuration

### Model Configuration
Models are configured in `config/benchmark_config.json`:

```json
{
  "models": {
    "gpt4-mini": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
}
```

### Benchmark Configuration
Benchmarks have default settings in the same file:

```json
{
  "benchmarks": {
    "truthfulqa": {
      "sample_size": 50,
      "evaluation_mode": "generation"
    }
  }
}
```

## Adding New Benchmarks

1. Create evaluator in `coherify/benchmarks/official/`
2. Add to `_initialize_evaluator()` in `benchmark_runner.py`
3. Add sample loader in `_load_samples()`
4. Update configuration in `config/benchmark_config.json`

## Migration Status

### ✅ Completed
- [x] Core unified runner implementation
- [x] TruthfulQA support
- [x] FEVER support (partial)
- [x] Makefile commands
- [x] Basic testing

### 🔄 In Progress
- [ ] Migrate all example scripts
- [ ] Update documentation
- [ ] Add SelfCheckGPT support

### 📋 TODO
- [ ] Add more coherence measures
- [ ] Support for custom datasets
- [ ] Batch processing optimization
- [ ] Cost tracking
- [ ] Statistical significance testing

## Benefits of Migration

1. **Consistency**: Same interface for all benchmarks
2. **Maintainability**: Single codebase to maintain
3. **Extensibility**: Easy to add new benchmarks
4. **Reliability**: Centralized error handling
5. **Performance**: Shared caching and optimization

## Deprecation Timeline

- **2025-01-28**: Unified runner introduced
- **2025-02-15**: Mark old scripts as deprecated
- **2025-03-01**: Remove old scripts from examples

## Troubleshooting

### Common Issues

**"Unknown benchmark: X"**
- Check spelling
- Ensure benchmark is implemented in unified runner

**"Model not found"**
- Check `config/benchmark_config.json`
- Ensure model name matches configuration

**"API key not set"**
- Export appropriate environment variables
- `export OPENAI_API_KEY=...` for OpenAI models
