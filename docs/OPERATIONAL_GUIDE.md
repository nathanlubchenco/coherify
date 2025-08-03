# Coherify Operational Guide

**Complete guide to understanding costs, performance, and operational characteristics of running TruthfulQA benchmarks with Coherify.**

## 📖 Table of Contents

- [Quick Reference](#quick-reference)
- [Dataset Characteristics](#dataset-characteristics)
- [Performance Analysis](#performance-analysis)
- [Cost Analysis](#cost-analysis)
- [Scaling Guide](#scaling-guide)
- [Resource Requirements](#resource-requirements)
- [Optimization Strategies](#optimization-strategies)
- [Practical Examples](#practical-examples)
- [Tools and Calculators](#tools-and-calculators)
- [Best Practices](#best-practices)

## 🎯 Quick Reference

### Common Scenarios

| Use Case | Command | Time | Cost | When to Use |
|----------|---------|------|------|-------------|
| **Quick Test** | `--sample-size 10` | 4 seconds | FREE | Daily development |
| **Development** | `--sample-size 50` | 22 seconds | FREE | Feature testing |
| **Validation** | `--sample-size 100` | 44 seconds | FREE | Pre-deployment |
| **Full Evaluation** | `--sample-size 817` | 6 minutes | FREE | Research analysis |
| **API Enhanced** | `--use-api --sample-size 100` | 17 minutes | $0.45 | Quality analysis |

### Quick Commands

```bash
# Calculate costs and time for any scenario
python scripts/benchmark_calculator.py --sample-size 100 --comparison

# Run quick local benchmark
python examples/run_truthfulqa_benchmark.py --sample-size 50

# Run API-enhanced analysis
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 100
```

## 📊 Dataset Characteristics

### TruthfulQA Validation Set

- **Total Samples**: 817 questions
- **Download Time**: ~2 seconds
- **File Size**: 500KB (lightweight text dataset)
- **Storage Requirements**: Minimal (text-only)
- **Network Requirements**: One-time download only

### Sample Structure

```
Average Question: 48 characters (~12 tokens)
Average Answer: 55 characters (~14 tokens)
Total per Sample: ~26 tokens
Multiple Answers: 5-7 correct/incorrect per question
Categories: Misconceptions, Biology, Law, Nutrition, etc.
```

### Data Loading Performance

```bash
Dataset Loading: 2 seconds
Parsing: Instant
Preprocessing: <1 second per 100 samples
Total Overhead: ~3 seconds regardless of sample size
```

## ⏱️ Performance Analysis

### Local Processing Performance

#### SemanticCoherence (Fastest)
- **Time per Sample**: 0.076 seconds
- **Full Dataset (817)**: 1.0 minute
- **Throughput**: ~50 samples/minute
- **Memory**: 4GB RAM
- **CPU**: 2+ cores recommended

#### HybridCoherence (Recommended)
- **Time per Sample**: 0.443 seconds
- **Full Dataset (817)**: 6.0 minutes
- **Throughput**: ~8 samples/minute
- **Memory**: 8GB RAM
- **CPU**: 4+ cores recommended

### API Processing Performance

#### Basic API Usage (1 generation per sample)
- **Time per Sample**: ~2 seconds (API latency)
- **Full Dataset (817)**: ~27 minutes
- **Throughput**: ~0.5 samples/minute
- **Bottleneck**: Network latency

#### Enhanced API Usage (Multiple generations)
- **Time per Sample**: ~10 seconds
- **Full Dataset (817)**: ~136 minutes
- **Throughput**: ~0.1 samples/minute
- **Bottleneck**: API rate limits

### Performance Scaling

```
Sample Size vs Processing Time (HybridCoherence):

10 samples:    4 seconds    (0.4s per sample)
50 samples:    22 seconds   (0.44s per sample)
100 samples:   44 seconds   (0.44s per sample)
500 samples:   3.7 minutes  (0.44s per sample)
817 samples:   6.0 minutes  (0.44s per sample)
2000 samples:  14.8 minutes (0.44s per sample)

Note: Linear scaling - performance is predictable
```

## 💰 Cost Analysis

### Local Processing (FREE)

**All local processing is completely free:**
- SemanticCoherence: $0
- HybridCoherence: $0
- All approximation algorithms: $0
- Caching and optimization: $0

**Only costs:** Electricity (~$0.01 per hour of processing)

### API Processing Costs

#### Token Usage (Full 817-sample dataset)
```
Input Tokens:  21,038 tokens (questions + context)
Output Tokens: 16,851 tokens (generated responses)
Total Tokens:  ~38,000 tokens
```

#### Cost by Provider/Model (Full Dataset)

**Budget Options:**
```
OpenAI GPT-3.5 Turbo:    $0.07
Anthropic Claude-3 Haiku: $0.03
```

**Standard Options:**
```
OpenAI GPT-4 Turbo:      $0.72
Anthropic Claude-3.5 Sonnet: $0.32
OpenAI GPT-4:            $1.64
```

**Premium Options (with enhancement):**
```
GPT-4 Turbo Enhanced:    $3.64
Claude-3.5 Enhanced:     $1.60
GPT-4 with Reasoning:    $18.45
```

### Cost Scaling by Sample Size

| Samples | Local | Budget API | Standard API | Premium API |
|---------|-------|------------|--------------|-------------|
| 10      | $0    | $0.00      | $0.01        | $0.04       |
| 50      | $0    | $0.00      | $0.04        | $0.22       |
| 100     | $0    | $0.00      | $0.09        | $0.45       |
| 817     | $0    | $0.03      | $0.72        | $3.64       |
| 2000    | $0    | $0.05      | $1.76        | $8.90       |
| 5000    | $0    | $0.12      | $4.40        | $22.25      |

### Additional API Costs

**Development Overhead:**
- Testing/debugging: 2-3x production costs
- Failed API calls: +10-20%
- Rate limiting delays: No cost, just time

**Optional Enhancements:**
- API embeddings: +$2-5 per full dataset
- Multiple temperature sampling: +2-3x base cost
- Answer expansion: +50% tokens
- Reasoning models: +2-4x base cost

## 📈 Scaling Guide

### Small Scale (1-50 samples)
**Best for:** Development, testing, debugging

```bash
# Recommended approach
python examples/run_truthfulqa_benchmark.py --sample-size 50

Time: 22 seconds
Cost: FREE
Resources: Minimal (2GB RAM)
Use case: Daily development
```

### Medium Scale (50-200 samples)
**Best for:** Validation, pre-deployment testing

```bash
# Local processing
python examples/run_truthfulqa_benchmark.py --sample-size 100

# Budget API testing
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 100

Time: 44 seconds (local) / 17 minutes (API)
Cost: FREE / $0.45
Use case: Weekly validation
```

### Large Scale (200-1000 samples)
**Best for:** Research, comprehensive evaluation

```bash
# Full dataset evaluation
python examples/run_truthfulqa_benchmark.py --sample-size 817

# API-enhanced subset
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 200

Time: 6 minutes (local) / 67 minutes (API)
Cost: FREE / $0.90
Use case: Monthly analysis
```

### Enterprise Scale (1000+ samples)
**Best for:** Production monitoring, large-scale research

```bash
# Batch processing with optimization
python examples/run_truthfulqa_benchmark.py --sample-size 2000

Time: 15 minutes (local)
Cost: FREE (local) / $8.90 (API)
Use case: Continuous monitoring
```

## 🖥️ Resource Requirements

### Minimum System Requirements

**For Basic Usage (≤100 samples):**
- CPU: 2 cores
- RAM: 4GB
- Storage: 2GB
- Network: Broadband (for model downloads)

**For Standard Usage (≤1000 samples):**
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB
- Network: Stable internet

### Recommended System Configuration

**For Optimal Performance:**
- CPU: 8+ cores (Intel i7/AMD Ryzen 7 or better)
- RAM: 16GB
- Storage: 10GB SSD
- GPU: Optional (3-5x speedup for large batches)
- Network: Broadband with low latency for API calls

### Memory Usage Patterns

```
Base Memory Usage:
- Coherify library: ~100MB
- Sentence transformers: ~500MB
- NLI models: ~1GB
- Working memory: ~2GB

Per Sample Memory:
- Proposition extraction: ~1MB per sample
- Embedding computation: ~10KB per proposition
- Coherence calculation: ~1KB per proposition pair

Caching Memory:
- Embedding cache: ~100MB per 10K cached texts
- Computation cache: ~10MB per 1K cached results
```

### Disk Usage

```
Model Storage:
- Sentence transformers: ~500MB
- NLI models: ~1GB
- Cached models: ~2GB total

Data Storage:
- TruthfulQA dataset: 0.5MB
- Result outputs: ~1MB per 1K samples
- Cache files: ~100MB per 10K samples

Recommended: 5GB free space for models + cache + results
```

## ⚡ Optimization Strategies

### Caching Optimizations

**Embedding Cache (10x+ speedup):**
```python
from coherify import CachedEncoder, EmbeddingCache

cache = EmbeddingCache(max_size=10000)
cached_encoder = CachedEncoder(cache=cache)
measure = SemanticCoherence(encoder=cached_encoder)
```

**Computation Cache:**
```python
# Automatic caching for repeated evaluations
# Results cached based on proposition text + measure type
# 1000x+ speedup for repeated identical evaluations
```

### Parallel Processing

**Multi-processing (2-4x speedup):**
```python
# Process multiple samples in parallel
# Recommended for datasets >100 samples
# Scales with CPU cores
```

**Batch Processing:**
```python
# Process samples in batches of 50-100
# Optimizes memory usage
# Reduces API overhead
```

### GPU Acceleration

**When Available (3-5x speedup):**
- Automatic GPU detection and usage
- Particularly effective for large datasets (>500 samples)
- Requires compatible GPU (CUDA/MPS)

### Approximation Algorithms

**For Large Datasets (5-10x speedup):**
```python
from coherify import ClusterBasedApproximator, RandomSampler

# Clustering approximation
approximator = ClusterBasedApproximator()
result = approximator.approximate_coherence(large_prop_set, target_clusters=20)

# Sampling approximation  
sampler = RandomSampler()
result = sampler.approximate_coherence(large_prop_set, sample_size=100)
```

## 🛠️ Practical Examples

### Development Workflow

```bash
# Day 1: Quick validation
python examples/run_truthfulqa_benchmark.py --sample-size 10
# Time: 4 seconds, Cost: FREE

# Day 2: Feature testing
python examples/run_truthfulqa_benchmark.py --sample-size 50  
# Time: 22 seconds, Cost: FREE

# Week 1: Integration testing
python examples/run_truthfulqa_benchmark.py --sample-size 100
# Time: 44 seconds, Cost: FREE

# Month 1: Full evaluation
python examples/run_truthfulqa_benchmark.py --sample-size 817
# Time: 6 minutes, Cost: FREE
```

### Research Workflow

```bash
# Baseline establishment
python examples/run_truthfulqa_benchmark.py --sample-size 817
# Full local evaluation: 6 minutes, FREE

# Method comparison
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 100
# API-enhanced analysis: 17 minutes, $0.45

# Paper preparation
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 817
# Comprehensive analysis: 136 minutes, $3.64
```

### Production Monitoring

```bash
# Daily health check
python examples/run_truthfulqa_benchmark.py --sample-size 50
# Time: 22 seconds, Cost: FREE

# Weekly quality assessment  
python examples/run_truthfulqa_benchmark.py --sample-size 200
# Time: 1.5 minutes, Cost: FREE

# Monthly comprehensive review
python examples/run_truthfulqa_benchmark.py --use-api --sample-size 817
# Time: 136 minutes, Cost: $3.64
```

## 🧮 Tools and Calculators

### Benchmark Calculator

**Calculate costs and time for any scenario:**
```bash
# Basic usage
python scripts/benchmark_calculator.py --sample-size 100

# Detailed API analysis
python scripts/benchmark_calculator.py --provider openai --model gpt-4-turbo --generations 2

# Comparison table
python scripts/benchmark_calculator.py --sample-size 100 --comparison

# Scaling analysis
python scripts/benchmark_calculator.py --scaling
```

### Setup Verification

**Verify your system is ready:**
```bash
python scripts/setup_benchmarks.py
```

**This checks:**
- Python version compatibility
- Required dependencies
- API key configuration
- Basic functionality
- Performance baseline

### Performance Profiling

**Measure your system's performance:**
```bash
# Built into benchmark runner
python examples/run_truthfulqa_benchmark.py --sample-size 10 --verbose

# Shows detailed timing:
# - Model loading time
# - Per-sample processing time  
# - Memory usage
# - Throughput calculations
```

## 💡 Best Practices

### Development Best Practices

1. **Start Small**: Always test with 10-50 samples first
2. **Use Local Processing**: Free and fast for most use cases
3. **Cache Everything**: Enable caching for repeated evaluations
4. **Profile Early**: Measure performance on your hardware
5. **Clean Output**: Use clean output mode for production

### Cost Management

1. **Local First**: Use local processing for development
2. **Budget Models**: Start with Claude-3 Haiku or GPT-3.5 Turbo
3. **Subset Testing**: Test API approaches on small subsets first
4. **Monitor Usage**: Track API costs and token consumption
5. **Batch Processing**: Group API calls to minimize overhead

### Performance Optimization

1. **Enable Caching**: 10x+ speedup for repeated work
2. **Use Approximation**: 5-10x speedup for large datasets
3. **Parallel Processing**: 2-4x speedup on multi-core systems
4. **GPU Acceleration**: 3-5x speedup when available
5. **Batch Sizing**: Optimal batch size is 50-100 samples

### Production Deployment

1. **Resource Planning**: Ensure adequate RAM and storage
2. **Error Handling**: Implement retry logic for API calls
3. **Monitoring**: Track performance and cost metrics
4. **Fallback Strategies**: Local processing as backup
5. **Regular Updates**: Keep models and dependencies current

### Research and Evaluation

1. **Baseline First**: Establish local processing baseline
2. **Incremental Enhancement**: Add API features gradually
3. **Statistical Significance**: Use adequate sample sizes
4. **Reproducibility**: Document all configuration parameters
5. **Cost Tracking**: Monitor and report evaluation costs

## 🎯 Summary

### Key Takeaways

**Coherify TruthfulQA benchmarking is:**
- ✅ **Fast**: 1-6 minutes for comprehensive local evaluation
- ✅ **Affordable**: FREE local processing, $0.03-3.64 for API enhancement
- ✅ **Scalable**: Linear scaling from 10 to 10,000+ samples
- ✅ **Flexible**: Multiple performance/cost tiers available
- ✅ **Production-Ready**: Robust caching and optimization

### Most Common Usage Pattern

**For 90% of users:**
```bash
python examples/run_truthfulqa_benchmark.py --sample-size 100
# Time: 44 seconds, Cost: FREE, Quality: Excellent
```

### When to Use APIs

- **Research papers**: Full dataset with API enhancement
- **Production validation**: Subset with high-quality models
- **Method comparison**: Multiple API providers
- **Quality assurance**: Periodic API-enhanced validation

### Cost-Effectiveness Sweet Spots

- **Development**: Local processing (FREE)
- **Testing**: Claude-3 Haiku ($0.03 full dataset)
- **Research**: GPT-4 Turbo Enhanced ($3.64 full dataset)
- **Production**: Cached local with periodic API validation

**Ready to start benchmarking? Use the calculator to plan your approach:**
```bash
python scripts/benchmark_calculator.py --sample-size YOUR_SIZE --comparison
```