# Task: Temperature-Based Coherence Evaluation

**Priority:** HIGH  
**Status:** Partially Implemented  
**Objective 3:** Using our coherence approach across different temperatures to apply to that benchmark  

## Current State
✅ `TemperatureVarianceCoherence` exists in `multi_response.py`  
✅ API-enhanced measures support temperature ranges  
✅ `APICoherenceConfig` has temperature configuration  
❌ Not integrated into main benchmark evaluation workflow  
❌ No systematic temperature sweep analysis  

## Design Approach
Integrate temperature variance into benchmark evaluation to study coherence across temperature ranges.

## Implementation Steps

### Phase 1: Temperature Sweep Infrastructure
- [ ] Create `TemperatureSweepEvaluator` class
- [ ] Define standard temperature ranges (e.g., [0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
- [ ] Implement systematic temperature grid evaluation
- [ ] Add temperature-specific result tracking

### Phase 2: Coherence Analysis Across Temperatures
- [ ] Measure coherence degradation as temperature increases
- [ ] Track response diversity vs coherence trade-offs
- [ ] Implement temperature-coherence correlation analysis
- [ ] Add optimal temperature detection based on coherence

### Phase 3: Benchmark Integration
- [ ] Modify `run_truthfulqa_benchmark.py` to support `--temperature-sweep`
- [ ] Add temperature analysis to `APIBenchmarkEvaluator`
- [ ] Create temperature-aware majority voting (combine with Task 03)
- [ ] Add temperature coherence to existing coherence measures

### Phase 4: Advanced Analysis
- [ ] Implement coherence-guided temperature selection
- [ ] Add ensemble coherence across temperature ranges
- [ ] Create temperature calibration based on question difficulty
- [ ] Compare different models' temperature-coherence relationships

## Detailed Implementation

### TemperatureSweepEvaluator Class
```python
class TemperatureSweepEvaluator:
    def __init__(self, 
                 base_evaluator,
                 temperature_range: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
                 coherence_measures: List[CoherenceMeasure] = None):
        self.base_evaluator = base_evaluator
        self.temperatures = temperature_range
        self.coherence_measures = coherence_measures or [HybridCoherence()]
    
    def evaluate_temperature_sweep(self, dataset) -> TemperatureSweepResults:
        # For each temperature:
        #   - Generate responses at that temperature  
        #   - Compute coherence for each measure
        #   - Track response diversity metrics
        # Return comprehensive analysis
```

### Key Metrics to Track
- **Coherence by Temperature**: How coherence changes with temperature
- **Response Diversity**: Unique responses, vocabulary diversity
- **Temperature Stability**: Variance in coherence across runs
- **Optimal Temperature**: Best coherence/diversity trade-off

## Integration with Existing Components

### Leverage Existing Infrastructure
- `APIEnhancedHybridCoherence` with temperature variance already exists
- `MultiResponseCoherenceMeasure` framework for multi-temperature responses
- `APIBenchmarkConfig` has temperature configuration

### Enhancement Strategy  
- Extend existing temperature variance to full benchmark evaluation
- Add temperature sweep to all coherence measures, not just API-enhanced ones
- Integrate with majority voting for temperature-aware consensus

## Success Criteria
- Temperature sweep runs across 0.1-1.2 range successfully
- Clear coherence-temperature relationship documented  
- Optimal temperature identified for different question types/categories
- Integration with benchmark evaluation workflow complete
- Performance acceptable for research use (< 2x evaluation time)

## Expected Findings
- Lower temperatures (0.1-0.3) show higher coherence but less diversity
- Moderate temperatures (0.5-0.7) may show optimal coherence/diversity trade-off
- Higher temperatures (0.9+) show lower coherence but more creative responses
- Different question categories may have different optimal temperatures

## Files to Create/Modify
- `coherify/evaluators/temperature_sweep.py` - New core implementation
- `coherify/analysis/temperature_analysis.py` - Analysis tools
- `examples/run_truthfulqa_benchmark.py` - Add temperature sweep support
- `coherify/measures/temperature_coherence.py` - Enhanced temperature-aware measures
- `tests/test_temperature_sweep.py` - Comprehensive tests

## Dependencies  
- Working API providers (OpenAI/Anthropic)
- Fixed test suite (Task 01)
- Working benchmark replication (Task 02)
- API keys for temperature-controlled generation

## Estimated Effort
**2-3 days** - Building on existing temperature infrastructure