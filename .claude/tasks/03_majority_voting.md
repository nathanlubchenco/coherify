# Task: Implement K-Run Majority Voting

**Priority:** HIGH  
**Status:** Not Started  
**Objective 2:** Being able to run that benchmark K times and apply naive majority voting  

## Current State
❌ No majority voting implementation exists  
❌ No K-run orchestration system  
✅ Multi-response framework exists (`multi_response.py`) that could be adapted  
✅ API providers support multiple calls

## Design Approach
Build on existing `MultiResponseCoherenceMeasure` infrastructure to create K-run majority voting.

## Implementation Steps

### Phase 1: Core Majority Voting Algorithm
- [ ] Create `MajorityVotingEvaluator` class in `coherify/evaluators/`
- [ ] Implement naive majority voting for discrete answers
- [ ] Add weighted voting based on coherence scores
- [ ] Support both categorical and continuous response aggregation

### Phase 2: K-Run Orchestration
- [ ] Create `KRunBenchmarkEvaluator` that wraps existing evaluators
- [ ] Add retry logic and error handling for failed runs
- [ ] Implement parallel execution for K runs (threading/async)
- [ ] Add progress tracking and intermediate result caching

### Phase 3: Integration with Benchmarks
- [ ] Modify `TruthfulQAEvaluator` to support K-run mode
- [ ] Add majority voting to `run_truthfulqa_benchmark.py`
- [ ] Support both generation and multiple-choice TruthfulQA formats
- [ ] Add configuration for K value and voting strategy

### Phase 4: Analysis and Reporting
- [ ] Track agreement/disagreement across K runs  
- [ ] Calculate confidence intervals for majority votes
- [ ] Report individual run coherence vs final majority coherence
- [ ] Add visualization of vote distributions

## Detailed Implementation

### MajorityVotingEvaluator Class
```python
class MajorityVotingEvaluator:
    def __init__(self, base_evaluator, k_runs: int = 5, voting_strategy: str = "simple"):
        self.base_evaluator = base_evaluator
        self.k_runs = k_runs
        self.voting_strategy = voting_strategy  # "simple", "weighted", "confidence"
    
    def evaluate_with_voting(self, dataset, coherence_threshold=0.5):
        # Run K evaluations
        # Collect all responses
        # Apply majority voting
        # Return aggregated results with individual run details
```

### Integration Points
- Modify `examples/run_truthfulqa_benchmark.py` to add `--k-runs N` flag
- Use existing `APIBenchmarkEvaluator` for response generation
- Leverage `TemperatureVarianceCoherence` for response diversity

## Success Criteria
- K-run evaluation completes successfully for TruthfulQA  
- Majority voting produces stable results across multiple runs
- Individual run coherence is tracked and reported
- Performance scales reasonably with K (linear time complexity)
- Clear improvement in answer quality through voting

## Test Cases
- K=1 should match single-run evaluation
- K=3, K=5, K=10 majority voting on known samples
- Disagreement analysis on ambiguous questions
- Performance timing for different K values

## Files to Create/Modify
- `coherify/evaluators/majority_voting.py` - New core implementation
- `coherify/evaluators/k_run.py` - K-run orchestration
- `examples/run_truthfulqa_benchmark.py` - Add K-run support
- `tests/test_majority_voting.py` - Comprehensive tests

## Dependencies
- Fixed test suite (Task 01)  
- Working benchmark replication (Task 02)
- Threading/async for parallel runs
- Statistical libraries for confidence intervals

## Estimated Effort
**3-4 days** - New functionality with integration