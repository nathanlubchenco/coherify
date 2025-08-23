# Task: Fix Test Suite

**Priority:** CRITICAL  
**Status:** Not Started  
**Objective:** Foundation for all other objectives  

## Problem
38 out of 95 tests are failing, blocking reliable development and benchmark validation.

## Root Causes
1. **Import/Mock Issues**: `transformers_utils` module attribute errors
2. **Test Infrastructure**: Missing test data setup for benchmark adapters  
3. **Dependency Mocking**: Incorrect mocks for ML pipeline components

## Implementation Steps

### Phase 1: Fix Core Test Infrastructure
- [ ] Fix `transformers_utils` import issues in test files
- [ ] Update mock setups for ML pipelines (sentence-transformers, NLI models)
- [ ] Resolve attribute errors in utility function mocks
- [ ] Fix empty set handling in coherence measures

### Phase 2: Fix Benchmark Adapter Tests  
- [ ] Create proper test data for TruthfulQA adapter tests
- [ ] Fix SelfCheckGPT adapter mock setup
- [ ] Resolve FEVER adapter test failures
- [ ] Add missing test data files/fixtures

### Phase 3: Fix Measure Tests
- [ ] Fix SemanticCoherence initialization tests
- [ ] Resolve HybridCoherence normalization tests  
- [ ] Fix EntailmentCoherence pipeline mock issues
- [ ] Update AdaptiveHybrid empty set handling

### Phase 4: Validation
- [ ] Run full test suite with 0 failures
- [ ] Add test coverage reporting
- [ ] Document test patterns for new contributions

## Success Criteria
- All tests pass (95/95)
- Test coverage > 80%
- Clear test patterns documented
- CI/CD ready test suite

## Files to Modify
- `tests/test_*.py` - All test files
- `coherify/utils/transformers_utils.py` - Fix attribute issues
- `conftest.py` - Add proper test fixtures (if needed)

## Dependencies
- pytest, pytest-cov
- Mock test data creation
- ML model mocking strategies

## Estimated Effort
**2-3 days** - High impact foundational work