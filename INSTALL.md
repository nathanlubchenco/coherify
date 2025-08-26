# Installation Guide for Coherify

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
bash setup_venv.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Install Coherify in development mode
4. Verify the installation

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install Coherify in development mode
pip install -e .

# Verify installation
python scripts/verify_environment.py
```

## Dependency Options

### Minimal Installation
For core functionality only:
```bash
pip install -r requirements-minimal.txt
```

### Development Installation
For development with testing and code quality tools:
```bash
pip install -r requirements-dev.txt
```

### Full Installation with All Options
```bash
pip install -e ".[dev,benchmarks,viz,api,ui]"
```

## Environment Variables

Set up your API keys:

```bash
# Required for benchmark evaluation
export OPENAI_API_KEY="your-openai-key-here"

# Optional for Anthropic models
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

## Verification

### Check Dependencies
```bash
python scripts/check_dependencies.py
```

### Verify Environment
```bash
python scripts/verify_environment.py
```

### Test the Pipeline
```bash
python examples/run_full_pipeline_comparison.py --model gpt4-mini --samples 3
```

## Troubleshooting

### Virtual Environment Not Active

If you see "Virtual Environment: ❌ Not active", make sure to activate it:
```bash
source venv/bin/activate
```

### Missing Dependencies

If specific packages are missing:
```bash
# Install a specific package
pip install openai

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

### Import Errors

If you get import errors for Coherify modules:
```bash
# Make sure Coherify is installed in development mode
pip install -e .
```

### API Key Issues

If API keys are not found:
```bash
# Check if they're set
echo $OPENAI_API_KEY

# Set them for current session
export OPENAI_API_KEY="sk-..."

# Add to shell profile for persistence
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

## System Requirements

- Python 3.8 or higher (3.10+ recommended)
- 8GB RAM minimum (16GB recommended for large benchmarks)
- CUDA-capable GPU (optional, for faster embedding computation)

## Common Issues

1. **PyTorch Installation Issues**: If PyTorch fails to install, visit https://pytorch.org for platform-specific instructions

2. **Transformers Cache**: First run will download models (~2GB). Set cache directory:
   ```bash
   export TRANSFORMERS_CACHE=/path/to/cache
   ```

3. **Memory Issues**: For large benchmarks, you may need to reduce batch sizes or use a machine with more RAM

## Next Steps

After installation:
1. Run the test pipeline: `python scripts/test_pipeline_with_progress.py`
2. Check the documentation: `docs/`
3. Review examples: `examples/`
4. Run full benchmarks: `python scripts/run_full_truthfulqa.py`