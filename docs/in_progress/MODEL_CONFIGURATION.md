# Model Configuration Guide

## Quick Start

### Recommended Models (Jan 2025)

```python
# Best Quality (default)
model_config = {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Cost Efficient
model_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000
}
```

## Model Selection Guide

### OpenAI Models

| Model | Use Case | Cost | Temperature |
|-------|----------|------|-------------|
| `gpt-4o` | **Default choice** - Best quality | Medium | ✅ 0-2 |
| `gpt-4o-mini` | High volume, cost-sensitive | Low | ✅ 0-2 |
| ~~`gpt-4`~~ | **Deprecated** - Use gpt-4o | High | ✅ 0-2 |
| ~~`gpt-3.5-turbo`~~ | **Deprecated** - Use gpt-4o-mini | Medium | ✅ 0-2 |
| `o1-preview` | **Avoid** - No temperature support | High | ❌ N/A |
| `o1-mini` | **Avoid** - No temperature support | Medium | ❌ N/A |

### Anthropic Models

| Model | Use Case | Temperature |
|-------|----------|-------------|
| `claude-3-5-sonnet-20241022` | Latest, most capable | ✅ 0-1 |
| `claude-3-opus-20240229` | Previous best, expensive | ✅ 0-1 |
| `claude-3-haiku-20240307` | Fast, lightweight | ✅ 0-1 |

## Configuration Files

### 1. Command Line Arguments

```bash
# Use specific model
python examples/run_full_pipeline_comparison.py --model gpt4o

# Available model shortcuts
--model gpt4o       # GPT-4o (recommended)
--model gpt4-mini   # GPT-4o-mini (cost efficient)
--model claude      # Claude 3.5 Sonnet
```

### 2. Config File (`config/benchmark_config.json`)

```json
{
  "models": {
    "default": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.7,
      "max_tokens": 1000
    },
    "gpt4o": {
      "provider": "openai",
      "model": "gpt-4o",
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
}
```

### 3. Environment Variables

```bash
# Set defaults via environment
export COHERIFY_DEFAULT_MODEL="gpt-4o"
export COHERIFY_DEFAULT_TEMPERATURE="0.7"
```

## Temperature Configuration

### For Coherence Testing

Temperature variation is crucial for coherence-based selection:

```python
# K-pass generation with temperature variation
def generate_k_responses(prompt, k=5):
    base_temp = 0.7
    temperatures = [
        0.6,   # More focused
        0.7,   # Balanced
        0.8,   # Slightly creative
        0.9,   # More creative
        1.0    # Most varied
    ]

    responses = []
    for temp in temperatures[:k]:
        response = model.generate(
            prompt=prompt,
            temperature=temp
        )
        responses.append(response)
    return responses
```

### Temperature Guidelines

| Task | Recommended Temperature | Range |
|------|------------------------|-------|
| Single Response Baseline | 0.7 | 0.5-0.9 |
| K-Pass Generation | 0.6-1.0 (varied) | 0.5-1.2 |
| GPT-4 Judge Evaluation | 0.3 | 0.1-0.5 |
| Coherence Scoring | 0.5 | 0.3-0.7 |

## Migration from Old Models

### Automatic Replacements

The following replacements are made automatically:

```python
# Old → New
"gpt-4" → "gpt-4o"
"gpt-3.5-turbo" → "gpt-4o-mini"
"text-davinci-003" → "gpt-4o-mini"
```

### Manual Updates Required

Check and update these files if you have custom configurations:

1. **Scripts**: `/scripts/*.py`
2. **Examples**: `/examples/*.py`
3. **Tests**: `/tests/*.py`
4. **Custom configs**: Any JSON/YAML config files

### Verification

```python
# Verify model configuration
from coherify.generation.model_runner import ModelRunner

config = {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
}

runner = ModelRunner(config)
result = runner.generate_response("Test prompt")
print(f"Model: {result.model}")
print(f"Temperature: {result.temperature}")
```

## Cost Optimization

### Model Selection by Budget

| Budget | Primary Model | Fallback Model | Strategy |
|--------|--------------|----------------|----------|
| High | `gpt-4o` | `gpt-4o` | Quality first |
| Medium | `gpt-4o` | `gpt-4o-mini` | Balance quality/cost |
| Low | `gpt-4o-mini` | `gpt-4o-mini` | Minimize cost |

### Cost per 1M Tokens (Jan 2025)

| Model | Input | Output | 1K Responses* |
|-------|-------|--------|---------------|
| gpt-4o | $2.50 | $10.00 | ~$12.50 |
| gpt-4o-mini | $0.15 | $0.60 | ~$0.75 |

*Estimated for typical 1000-token responses

## Troubleshooting

### Model Not Found

```python
# Error: Model 'gpt-4' not found
# Solution: Update to gpt-4o
config["model"] = "gpt-4o"
```

### No Temperature Support

```python
# Error: Model doesn't support temperature
# Issue: Using o1-preview or o1-mini
# Solution: Switch to gpt-4o or gpt-4o-mini
```

### Rate Limits

```python
# Implement fallback strategy
try:
    response = generate_with_model("gpt-4o", prompt)
except RateLimitError:
    response = generate_with_model("gpt-4o-mini", prompt)
```

## Best Practices

1. **Always use gpt-4o** instead of gpt-4
2. **Always use gpt-4o-mini** instead of gpt-3.5-turbo
3. **Verify temperature support** before using a model
4. **Use varied temperatures** for K-pass generation
5. **Monitor costs** with token tracking
6. **Implement fallbacks** for rate limits

## API Keys

```bash
# Required for OpenAI models
export OPENAI_API_KEY="sk-..."

# Optional for Anthropic models
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify setup
python scripts/verify_environment.py
```
