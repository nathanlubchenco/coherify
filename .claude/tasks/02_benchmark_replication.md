# Task: Run Existing Benchmark to Replicate Known Results

**Priority:** HIGH  
**Status:** Not Started  
**Objective 1:** Being able to run an existing benchmark as-is to replicate known results  

## Current State
✅ TruthfulQA benchmark implementation complete  
✅ FEVER benchmark implementation complete  
✅ Comprehensive benchmark runners exist  
❌ No validated baseline results for comparison  
❌ Test failures block reliable runs  

## Implementation Steps

### Phase 1: Establish Baseline (Post Test Fix)
- [ ] Run `examples/run_truthfulqa_benchmark.py` with sample data
- [ ] Document performance characteristics (timing, scores)
- [ ] Compare with any available literature baselines
- [ ] Run `examples/run_fever_benchmark.py` for comparison data

### Phase 2: Validate Implementation
- [ ] Test with known TruthfulQA samples and expected outcomes
- [ ] Verify coherence scores are reasonable (not NaN, reasonable ranges)
- [ ] Check category-wise performance consistency
- [ ] Validate contrastive evaluation (positive > negative coherence)

### Phase 3: Performance Benchmarking
- [ ] Measure timing across different coherence measures
- [ ] Test scalability with larger sample sizes (100, 500, 1000 samples)
- [ ] Document resource usage (memory, computation time)
- [ ] Compare semantic vs hybrid vs Shogenji performance

### Phase 4: Replication Documentation  
- [ ] Create baseline results documentation
- [ ] Document any deviations from literature results
- [ ] Create reproduction guide for researchers
- [ ] Add benchmark validation to CI/CD

## Success Criteria
- TruthfulQA benchmark runs successfully on full validation set
- Coherence scores within reasonable ranges (0.0-1.0, not NaN)
- Performance characteristics documented
- Baseline results established for future comparison
- At least one benchmark shows replicable patterns

## Expected Results to Validate
Based on `run_truthfulqa_benchmark.py` structure:
- Mean coherence scores by category
- Positive vs negative coherence contrast
- Performance timing per sample
- Category-wise coherence patterns

## Files to Use
- `examples/run_truthfulqa_benchmark.py` - Primary benchmark runner
- `examples/run_fever_benchmark.py` - Alternative benchmark  
- `coherify/benchmarks/truthfulqa.py` - Core implementation
- `coherify/measures/*.py` - Various coherence measures

## Dependencies
- datasets library for real TruthfulQA data
- API keys for enhanced benchmarks (optional)
- Fixed test suite (Task 01)

## Estimated Effort
**1-2 days** - After test suite fixes