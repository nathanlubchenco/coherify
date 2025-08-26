# Troubleshooting Guide

## Common Issues and Solutions

### 1. "Performance 0.0% is unrealistically low"

**Symptoms**: 
- All evaluations showing 0% performance
- Warning messages about unrealistic performance

**Causes**:
1. **No API key configured** - Model returns empty/error responses
2. **Wrong API key** - Authentication failures
3. **Mock model in use** - Default responses not suitable for evaluation

**Solutions**:

#### Set up OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...your-key-here...
```

#### Verify API key works:
```python
from coherify.providers import OpenAIProvider
provider = OpenAIProvider(model_name="gpt-4o-mini")
response = provider.generate_text("What is 2+2?")
print(response.text)  # Should print a real answer
```

#### Check model configuration:
```bash
# Should NOT be "default" or "mock" for real testing
cat config/benchmark_config.json | grep -A3 '"default"'
```

#### Run with verbose mode to see actual responses:
```bash
python examples/run_full_pipeline_comparison.py --model gpt4-mini --samples 2 --verbose
```

### 2. "OpenAIProvider object has no attribute 'generate'"

**Cause**: Method name mismatch (should be `generate_text` not `generate`)

**Solution**: Already fixed in latest code. Pull latest changes or update:
```python
# Wrong
response = provider.generate(prompt)
# Correct  
response = provider.generate_text(prompt)
```

### 3. "Cannot import name 'Proposition' from 'coherify.core.types'"

**Cause**: Import path error (classes are in `base.py` not `types.py`)

**Solution**: Update imports:
```python
# Wrong
from coherify.core.types import Proposition, PropositionSet
# Correct
from coherify.core.base import Proposition, PropositionSet
```

### 4. UI Shows "Failed to fetch"

**Causes**:
1. No reports generated yet
2. Server not running
3. Port conflict

**Solutions**:

#### Generate some reports first:
```bash
make benchmark-stage1 MODEL=default SAMPLES=5
```

#### Start UI server:
```bash
python examples/comprehensive_benchmark_demo.py --ui-only
# Or
python -c "from coherify.ui import start_result_server; start_result_server()"
```

#### Check for port conflicts:
```bash
lsof -i:8080  # See what's using port 8080
# Kill if needed
lsof -ti:8080 | xargs kill -9
```

### 5. "BLEURT not available" or "GPT-judge not available"

**Cause**: Optional dependencies not installed

**Solution**: The system will automatically fall back to GPT-4 judge (if API key set) or embedding similarity. For best results:

```bash
# Install BLEURT (optional)
pip install bleurt-pytorch

# For GPT-judge, you need fine-tuned models (not publicly available)
# GPT-4 judge is used as fallback automatically
```

### 6. Slow Performance

**Causes**:
1. Sequential API calls
2. No caching
3. Large K values

**Solutions**:

#### Enable caching:
```python
from coherify.utils.caching import get_default_embedding_cache
cache = get_default_embedding_cache()
# Cache persists across runs
```

#### Reduce samples for testing:
```bash
# Start small
make benchmark-full-pipeline MODEL=gpt4-mini SAMPLES=5 K_RUNS=3
```

#### Use faster models:
```bash
# gpt-4o-mini is faster than gpt-4
MODEL=gpt4-mini  # Fast
MODEL=gpt4       # Slower but more accurate
```

## Validation Checklist

Before reporting an issue, verify:

1. ✅ API key is set: `echo $OPENAI_API_KEY`
2. ✅ Can generate responses: Test with simple prompt
3. ✅ Using real model not mock: Check MODEL parameter
4. ✅ Latest code: `git pull`
5. ✅ Dependencies installed: `pip install -e ".[dev,benchmarks]"`

## Debug Mode

For maximum debugging information:

```bash
# Set debug environment
export COHERIFY_DEBUG=1
export PYTHONUNBUFFERED=1

# Run with all verbosity
python examples/run_full_pipeline_comparison.py \
    --model gpt4-mini \
    --samples 2 \
    --k-responses 2 \
    --verbose 2>&1 | tee debug.log
```

## Getting Help

If issues persist:

1. Check `TODO.md` for known issues
2. Review `CURRENT_STATE.md` for project status
3. Look at recent commits for fixes
4. Create detailed issue with:
   - Exact command run
   - Full error message
   - Environment details (OS, Python version)
   - API key status (set/not set, don't share the key!)

---
**Last Updated**: 2024-01-24