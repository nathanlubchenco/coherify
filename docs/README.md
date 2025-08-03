# Coherify Documentation

## 📚 Documentation Index

Welcome to the Coherify documentation! This directory contains comprehensive guides for using Coherify effectively.

### 📖 Core Documentation

- **[Operational Guide](OPERATIONAL_GUIDE.md)** - Complete guide to costs, performance, and scaling
- **[Multi-Format Benchmarks](MULTI_FORMAT_BENCHMARKS.md)** - Guide to GSM8K, HellaSwag, MMLU, and multi-response evaluation
- **[Getting Started](../GETTING_STARTED.md)** - Quick start guide for running benchmarks
- **[Main README](../README.md)** - Library overview and basic usage

### 🔧 Setup and Configuration

- **[Benchmark Requirements](../BENCHMARK_REQUIREMENTS.md)** - What you need to run benchmarks
- **[Warnings Fixed](../WARNINGS_FIXED.md)** - Clean output implementation details

### 🛠️ Tools and Utilities

- **[Benchmark Calculator](../scripts/benchmark_calculator.py)** - Cost and time estimation tool
- **[Setup Script](../scripts/setup_benchmarks.py)** - Automated setup and verification

### 📊 Examples and Tutorials

- **[Basic Usage](../examples/basic_usage.py)** - Getting started with core functionality
- **[TruthfulQA Runner](../examples/run_truthfulqa_benchmark.py)** - Complete benchmark runner
- **[Multi-Format Benchmarks](../examples/run_multi_format_benchmarks.py)** - GSM8K, HellaSwag, MMLU evaluation
- **[Multi-Format Basic Test](../examples/test_multi_format_basic.py)** - Test multi-format functionality
- **[API Providers Demo](../examples/api_providers_demo.py)** - External API integration
- **[Practical Applications](../examples/practical_applications.py)** - Real-world use cases

## 🎯 Quick Navigation

### New Users Start Here
1. [Getting Started Guide](../GETTING_STARTED.md)
2. [Benchmark Requirements](../BENCHMARK_REQUIREMENTS.md)  
3. [Basic Usage Example](../examples/basic_usage.py)
4. [Multi-Format Basic Test](../examples/test_multi_format_basic.py)

### Planning Your Benchmarks
1. [Operational Guide](OPERATIONAL_GUIDE.md) - Comprehensive cost/performance analysis
2. [Benchmark Calculator](../scripts/benchmark_calculator.py) - Interactive planning tool

### Advanced Features
1. [Multi-Format Benchmarks Guide](MULTI_FORMAT_BENCHMARKS.md) - GSM8K, HellaSwag, MMLU, multi-response evaluation
2. [API Providers Demo](../examples/api_providers_demo.py) - External model integration
3. [Practical Applications](../examples/practical_applications.py) - Production examples

## 📋 Documentation Standards

### File Organization
```
docs/
├── README.md                 # This index
├── OPERATIONAL_GUIDE.md      # Complete operational analysis
../
├── GETTING_STARTED.md        # Quick start guide
├── README.md                 # Main library documentation
├── examples/                 # Practical code examples
├── scripts/                  # Utility scripts and tools
└── *.md                      # Specialized guides
```

### Documentation Types

**Guides** - Step-by-step instructions for specific tasks
- Getting Started Guide
- Operational Guide
- Setup guides

**References** - Comprehensive information for planning and analysis
- Benchmark Requirements
- API cost analysis
- Performance characteristics

**Examples** - Working code demonstrating features
- Basic usage patterns
- Advanced integrations
- Real-world applications

**Tools** - Interactive utilities for planning and setup
- Benchmark calculator
- Setup verification scripts
- Performance measurement tools

## 🚀 Contributing to Documentation

### Adding New Documentation

1. **Create in appropriate location**:
   - Guides: `docs/GUIDE_NAME.md`
   - Examples: `examples/example_name.py`
   - Tools: `scripts/tool_name.py`

2. **Follow naming conventions**:
   - ALL_CAPS for major guides
   - snake_case for scripts and examples
   - Descriptive names indicating purpose

3. **Update this index** when adding new documentation

4. **Include practical examples** and real command-line usage

### Documentation Style Guide

**Structure each document with:**
- Clear title and purpose
- Table of contents for long documents
- Quick reference section
- Detailed explanations with examples
- Practical next steps

**Use consistent formatting:**
- `code blocks` for commands and code
- **bold** for emphasis
- *italics* for notes
- Tables for structured data
- Emojis for visual navigation (📊 📚 🚀 etc.)

**Include actionable content:**
- Specific commands users can run
- Expected outputs and results
- Troubleshooting information
- Cost and time estimates

## 🎉 Get Started

**Ready to use Coherify?** Start with:

```bash
# Quick setup verification
python scripts/setup_benchmarks.py

# Test multi-format benchmarks (no API required)
python examples/test_multi_format_basic.py

# Plan your benchmark
python scripts/benchmark_calculator.py --sample-size 100 --comparison

# Run your first benchmark
python examples/run_truthfulqa_benchmark.py --sample-size 10

# Try multi-format evaluation
python examples/run_multi_format_benchmarks.py --sample-size 10
```

**Need help?** Check the [Operational Guide](OPERATIONAL_GUIDE.md) for comprehensive information about costs, performance, and best practices!