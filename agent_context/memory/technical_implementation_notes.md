# Technical Implementation Notes & Patterns

## Proven Implementation Patterns

### Coherence Measure Implementation
```python
# Standard pattern for new coherence measures:
class NewCoherenceMeasure(CoherenceMeasure):
    def __init__(self, specific_params):
        # Initialize measure-specific components
        
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        start_time = time.time()
        
        # Handle single proposition case
        if len(prop_set) < 2:
            return CoherenceResult(score=1.0, ...)
        
        # Main computation logic
        score = self._compute_core_logic(prop_set)
        
        # Return rich result with details
        return CoherenceResult(
            score=score,
            measure_name="NewCoherence", 
            details={...},
            computation_time=time.time() - start_time
        )
```

### Benchmark Adapter Pattern
```python
# Universal pattern for benchmark adapters:
class NewBenchmarkAdapter(BenchmarkAdapter):
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        # Extract key components
        question = sample.get("question_key")
        answer = sample.get("answer_key") 
        
        # Create propositions (segment if needed)
        props = [Proposition(text=segment) for segment in segments]
        
        # Add benchmark-specific metadata
        metadata = {"benchmark": "new_benchmark", ...}
        
        return PropositionSet(
            propositions=props,
            context=question,
            metadata=metadata
        )
```

## Key Technical Details

### Caching Integration Points
- **Embedding caching**: Use `CachedEncoder` wrapper for any text encoder
- **Computation caching**: Use `@cached_computation` decorator for expensive functions
- **Manual caching**: Use `EmbeddingCache` or `ComputationCache` classes directly
- **Cache directories**: Default to `.coherify_cache/` with subdirectories by type

### Performance Optimization Strategies
1. **Batch processing**: Always implement `batch_compute()` methods
2. **Lazy evaluation**: Don't compute unless needed
3. **Caching at multiple levels**: Embeddings, computations, and results
4. **Memory management**: Set TTL and max_size for caches
5. **Efficient data structures**: Use numpy arrays where possible

### Error Handling Patterns
```python
# Robust error handling for model failures:
try:
    result = expensive_model_call(input)
except Exception as e:
    # Log warning but continue with fallback
    logger.warning(f"Model call failed: {e}")
    result = fallback_method(input)
```

### Testing Strategy
- **Mock expensive operations**: Use `Mock()` for models in tests
- **Test edge cases**: Single propositions, empty inputs, malformed data
- **Integration tests**: Test full pipelines with real data
- **Performance tests**: Verify caching provides expected speedups

## Dependency Management
- **Core dependencies**: numpy, torch, transformers, datasets, scikit-learn
- **Optional dependencies**: sentence-transformers (for default semantic), specific NLI models
- **Development dependencies**: pytest, black, flake8, mypy for code quality
- **Graceful degradation**: Provide fallbacks when optional dependencies missing

## Notification Hook Patterns
- **macOS**: Use `terminal-notifier` for reliable desktop notifications
- **Linux**: Use `notify-send` for system notifications
- **Fallback**: Use `echo` + `say` for audio-only notifications
- **Configuration**: Store in `.claude_hooks.json` (gitignored for per-developer customization)

## Future Extension Points

### New Coherence Measures
1. Inherit from `CoherenceMeasure`
2. Implement `compute()` and `compute_pairwise()` methods
3. Add to `coherify.measures.__init__.py`
4. Create comprehensive tests
5. Add to examples and documentation

### New Benchmark Adapters  
1. Inherit from `BenchmarkAdapter`
2. Implement `adapt_single()` method
3. Add specialized evaluator if needed
4. Register in `BENCHMARK_ADAPTERS` dict
5. Create examples and tests

### Performance Improvements
- **Approximation algorithms**: For very large proposition sets
- **Parallel processing**: Batch computation across multiple cores
- **Model optimization**: Quantization, distillation for faster inference
- **Smart batching**: Group similar computations for efficiency

## Common Pitfalls & Solutions

### Import Dependency Issues
- **Problem**: Optional dependencies not available
- **Solution**: Use try/except with informative error messages
- **Pattern**: Provide simple fallbacks for core functionality

### Memory Issues with Large Datasets
- **Problem**: Caching everything can consume excessive memory
- **Solution**: Implement LRU eviction and configurable cache sizes
- **Pattern**: Monitor memory usage and adjust cache limits

### Model Calibration Problems
- **Problem**: Models overconfident or poorly calibrated
- **Solution**: Use relative scoring and normalization
- **Pattern**: Focus on ranking/comparison rather than absolute scores

### Performance Bottlenecks
- **Problem**: Repeated expensive computations
- **Solution**: Multi-level caching strategy
- **Pattern**: Cache at embeddings, computations, and results levels

## Validation Strategies

### Coherence Measure Validation
1. **Sanity checks**: Single propositions should score 1.0
2. **Extremes testing**: Highly coherent vs incoherent content
3. **Consistency testing**: Similar inputs should produce similar scores
4. **Component analysis**: Verify individual components behave correctly

### Benchmark Integration Validation
1. **Format compliance**: Verify PropositionSet structure
2. **Metadata preservation**: Important benchmark info retained
3. **Edge case handling**: Empty, malformed, or unusual inputs
4. **Round-trip testing**: Adapter → coherence → analysis pipeline

This technical foundation provides reliable patterns for extending the Coherify library while maintaining quality and performance standards.