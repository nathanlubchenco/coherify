# OpenAI Models Documentation for Coherify

*Last Updated: January 2025*

## Current Recommended Models

### 🚀 GPT-4o (Recommended Default)
**Model ID**: `gpt-4o`
- **Description**: Most capable multimodal model with vision, function calling, and JSON mode
- **Context Window**: 128K tokens
- **Max Output**: 16,384 tokens
- **Temperature**: 0-2 (default: 1)
- **Knowledge Cutoff**: October 2023
- **Use Cases**: Complex reasoning, analysis, code generation, multimodal tasks

### 💰 GPT-4o-mini (Cost-Efficient)
**Model ID**: `gpt-4o-mini`
- **Description**: Fast, affordable small model for focused tasks
- **Context Window**: 128K tokens
- **Max Output**: 16,384 tokens
- **Temperature**: 0-2 (default: 1)
- **Knowledge Cutoff**: October 2023
- **Pricing**: 15¢/1M input, 60¢/1M output (60% cheaper than GPT-3.5 Turbo)
- **Use Cases**: High-volume tasks, evaluation, quick responses

## Temperature Parameter

### Range and Effects
- **Range**: 0.0 to 2.0
- **Default**: 1.0
- **Low (0.0-0.5)**: Focused, deterministic, consistent
- **Medium (0.5-1.0)**: Balanced creativity and consistency
- **High (1.0-2.0)**: Creative, varied, potentially less coherent

### For Coherence Testing
Temperature is crucial for coherence-based selection:
```python
# Recommended temperature variation for K-pass generation
temperatures = [0.7, 0.8, 0.9, 1.0, 1.1]  # Slight variations
```

## Models to Avoid/Deprecate

### ❌ Legacy Models (DO NOT USE)
- `gpt-4`: Superseded by `gpt-4o` (better and cheaper)
- `gpt-3.5-turbo`: Superseded by `gpt-4o-mini` (better and cheaper)
- `text-davinci-003`: Deprecated
- `davinci-002`: Deprecated

### ⚠️ Special Purpose Models
- `o1-preview`, `o1-mini`: Reasoning models WITHOUT temperature parameter
  - These are not suitable for coherence testing as they don't support temperature variation

## API Usage Examples

### Basic Configuration
```python
# Recommended default configuration
model_config = {
    "provider": "openai",
    "model": "gpt-4o",        # or "gpt-4o-mini" for cost efficiency
    "temperature": 0.7,
    "max_tokens": 1000,
}
```

### For Coherence Testing (K-Pass Generation)
```python
# Generate diverse responses with temperature variation
def generate_k_responses(prompt, k=5):
    base_temp = 0.7
    temperatures = [base_temp + 0.1 * i for i in range(k)]

    responses = []
    for temp in temperatures:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=1000
        )
        responses.append(response)
    return responses
```

### For Evaluation (GPT-4 Judge)
```python
# Use lower temperature for consistent evaluation
judge_config = {
    "model": "gpt-4o",
    "temperature": 0.3,  # Low for consistency
    "response_format": {"type": "json_object"}
}
```

## Cost Optimization

### Model Selection by Use Case

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| Benchmark Evaluation | `gpt-4o` | Highest accuracy for judging |
| Mass Generation | `gpt-4o-mini` | 60% cheaper, sufficient quality |
| Development/Testing | `gpt-4o-mini` | Fast iteration, low cost |
| Production Pipeline | `gpt-4o` | Best quality-cost balance |

### Cost Comparison (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo (deprecated) | $0.50 | $1.50 |

## Migration Guide

### Update Configuration Files

Replace all instances:
- `"gpt-4"` → `"gpt-4o"`
- `"gpt-3.5-turbo"` → `"gpt-4o-mini"`
- `"text-davinci-003"` → `"gpt-4o-mini"`

### Files to Update
1. `/config/benchmark_config.json`
2. `/coherify/providers/openai_provider.py`
3. `/examples/*/`
4. `/scripts/*/`

## Best Practices for Coherify

### 1. Temperature Strategy
- **Single Response**: Use 0.7 for balanced output
- **K-Pass Generation**: Vary from 0.6-1.2 for diversity
- **Evaluation/Judging**: Use 0.3 for consistency

### 2. Model Selection
- **Default**: `gpt-4o` for quality
- **High Volume**: `gpt-4o-mini` for cost efficiency
- **Never Use**: Models without temperature parameter (o1 series)

### 3. Error Handling
```python
try:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )
except openai.RateLimitError:
    # Fallback to gpt-4o-mini
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )
```

## API Reference

### Required Parameters
- `model`: Model identifier (e.g., "gpt-4o")
- `messages`: List of message objects

### Optional Parameters for Coherence Testing
- `temperature`: 0-2 (REQUIRED for coherence testing)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Nucleus sampling (alternative to temperature)
- `frequency_penalty`: Reduce repetition (-2 to 2)
- `presence_penalty`: Encourage new topics (-2 to 2)

## Environment Setup

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Install latest OpenAI library
pip install openai>=1.0.0

# Verify installation
python -c "import openai; print(openai.__version__)"
```

## Notes

1. **Always use models with temperature support** for coherence testing
2. **GPT-4o is the recommended default** - better and cheaper than GPT-4
3. **GPT-4o-mini is ideal for high-volume** - 60% cheaper than GPT-3.5
4. **Avoid o1 models** - they don't support temperature variation
5. **Monitor costs** - use token counting and cost estimation
