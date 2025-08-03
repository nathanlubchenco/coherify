# Getting Started with Coherify Benchmarks

This guide shows you how to actually run benchmarks with Coherify.

## Quick Start

### 1. Basic Setup

```bash
# Install Coherify in development mode
pip install -e .

# Install benchmark dependencies
pip install datasets requests

# Run setup script to verify everything works
python scripts/setup_benchmarks.py
```

### 2. Run Your First Benchmark

```bash
# Run TruthfulQA with a small sample (no API required)
python examples/run_truthfulqa_benchmark.py --sample-size 5
```

This will:
- Download TruthfulQA data (or use mock data if unavailable)
- Run coherence evaluation with local models
- Show results and analysis

### 3. API-Enhanced Benchmarks (Optional)

If you have API keys, you can run enhanced benchmarks:

```bash
# Set up API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run with API enhancement
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 3
```

## What You'll See

### Basic Benchmark Output

```
🚀 TruthfulQA Benchmark Runner
==================================================
📚 Loading TruthfulQA data...
  ✅ Loaded 5 samples from datasets library

🔧 Setting up coherence measures...
  ✅ Added 2 local measures

🏃 Running Basic TruthfulQA Benchmark...

  📊 Evaluating with SemanticCoherence...
    Samples: 5
    Mean coherence: 0.734
    Evaluation time: 1.23s
    Categories:
      Nutrition: 0.756
      Misconceptions: 0.712

  📊 Evaluating with HybridCoherence...
    Samples: 5
    Mean coherence: 0.782
    Evaluation time: 1.87s
```

### API-Enhanced Output

```
🚀 Running API-Enhanced TruthfulQA Benchmark...
  🌐 Using provider: openai
  📊 Evaluating 3 samples...
    Progress: 100.0% (3/3)
  ✅ Completed in 15.42s

  📈 API Enhancement Statistics:
    Enhanced samples: 3
    Total API generations: 6
    Providers used: openai

  📊 Coherence Scores:
    HybridCoherence:
      Mean: 0.734 ± 0.089
      Range: 0.623 - 0.845
    APIEnhancedHybridCoherence:
      Mean: 0.812 ± 0.067
      Range: 0.734 - 0.891
```

## Understanding the Results

### Coherence Scores
- **0.0 - 0.3**: Low coherence (contradictory or unrelated statements)
- **0.3 - 0.6**: Moderate coherence (some logical connection)
- **0.6 - 0.8**: High coherence (logically consistent)
- **0.8 - 1.0**: Very high coherence (strongly interconnected)

### Benchmark Categories
- **Nutrition**: Food and health misconceptions
- **Misconceptions**: Common false beliefs
- **Biology**: Scientific misconceptions
- **Law**: Legal misconceptions

### API Enhancement Benefits
- **Temperature Variance**: Multiple responses at different creativity levels
- **Reasoning Traces**: Detailed reasoning from advanced models (o3)
- **Confidence Scoring**: Model's confidence in its responses
- **Answer Expansion**: More detailed explanations

## Troubleshooting

### Common Issues

1. **"No module named 'datasets'"**
   ```bash
   pip install datasets
   ```

2. **"No API providers available"**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **"Failed to load TruthfulQA data"**
   - The script will use mock data as fallback
   - Check internet connection for full dataset

4. **Slow evaluation**
   - Use `--sample-size 5` for faster testing
   - API calls take longer but provide enhanced analysis

### Performance Tips

- **Local evaluation**: Fast, works offline, good for development
- **API evaluation**: Slower, requires internet, provides enhanced insights
- **Sample size**: Start with 5-10 samples for testing
- **Caching**: API responses are automatically cached

## Next Steps

### Explore More Examples

```bash
# Basic functionality
python examples/basic_usage.py

# Advanced features
python examples/practical_applications.py

# API provider features
python examples/api_providers_demo.py

# Visualization tools
python examples/phase3_features.py
```

### Customize Evaluations

1. **Edit coherence measures**: Modify `coherify/measures/`
2. **Create new adapters**: Add to `coherify/benchmarks/`
3. **Adjust API settings**: Modify temperature ranges, generation counts
4. **Add new benchmarks**: Follow the adapter pattern

### Production Usage

```python
from coherify import setup_providers, APIEnhancedHybridCoherence
from coherify.benchmarks.api_enhanced import APIBenchmarkEvaluator

# Setup
setup_providers()
measure = APIEnhancedHybridCoherence()
evaluator = APIBenchmarkEvaluator(adapter, [measure])

# Evaluate
results = evaluator.evaluate_dataset(your_data)
```

## Support

- **Examples**: Check `examples/` directory
- **Documentation**: See `README.md`
- **Issues**: Report problems and get help
- **Development**: All code is in `coherify/` directory

Happy benchmarking! 🚀