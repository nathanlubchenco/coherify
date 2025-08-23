# Benchmark Setup and Requirements Guide

*Consolidated from BENCHMARK_REQUIREMENTS.md and GETTING_STARTED.md*

## Overview

Complete guide for setting up and running Coherify benchmarks with all dependencies and configurations.

## System Requirements

### Basic Setup
- Python 3.8+
- Virtual environment (recommended)
- Internet connection for dataset downloads

### Core Dependencies
```bash
# Install in development mode
pip install -e .

# Essential benchmark dependencies
pip install datasets requests evaluate

# Optional for enhanced features
pip install rouge-score nltk scikit-learn
```

## Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install Coherify
pip install -e .

# Verify installation
python scripts/setup_benchmarks.py
```

### 2. Run Your First Benchmark
```bash
# Basic TruthfulQA evaluation
python examples/run_truthfulqa_benchmark.py --sample-size 10

# With comprehensive reporting
python examples/comprehensive_benchmark_demo.py --sample-size 5

# Launch web UI for results
python examples/comprehensive_benchmark_demo.py --use-ui
```

## API Configuration (Optional)

### OpenAI Setup
```bash
export OPENAI_API_KEY="your-key-here"
```

### Anthropic Setup
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### API-Enhanced Benchmarks
```bash
# Run with API enhancements
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 5
```

## Troubleshooting

### Common Issues
1. **Module not found**: Ensure `pip install -e .` was run
2. **Dataset download fails**: Check internet connection
3. **API errors**: Verify API keys are set correctly
4. **Memory issues**: Reduce sample size or use approximation algorithms

### Performance Tips
- Use `--sample-size` to limit evaluation scope during testing
- Enable clean output mode for cleaner console output
- Use web UI for better result visualization

## Available Benchmarks

- **TruthfulQA**: Truthfulness evaluation (817 samples)
- **SelfCheckGPT**: Hallucination detection
- **FEVER**: Fact verification with evidence
- **FaithBench**: Faithfulness in summarization

## Next Steps

1. Run basic benchmarks to verify setup
2. Experiment with different coherence measures
3. Use comprehensive reporting for detailed analysis
4. Explore web UI for interactive results viewing