# Benchmark Methodology Fixes - Implementation Tasks

*Derived from comprehensive research of benchmark papers*

## Task Overview

Research into the original benchmark papers revealed critical implementation issues in our current codebase. This document outlines specific tasks to fix these methodological problems.

## Critical Issues Identified

### 1. TruthfulQA Backwards Evaluation Logic
**Problem**: We're treating coherent responses as correct, but TruthfulQA is designed to elicit coherent but false responses.
**Impact**: False positive truthfulness scores, invalid baseline measurements
**Priority**: 🔴 Critical - Immediate Fix Required

### 2. SelfCheckGPT Missing Core Algorithm  
**Problem**: Missing the fundamental sampling-based consistency checking methodology
**Impact**: Not actually implementing SelfCheckGPT, just basic hallucination detection
**Priority**: 🟡 High - Core Feature Missing

### 3. FEVER Incomplete Evidence Handling
**Problem**: No support for multi-sentence/multi-page evidence chains (required for 30%+ of claims)
**Impact**: Incomplete benchmark coverage, missing complex reasoning cases
**Priority**: 🟡 High - Significant Coverage Gap

### 4. FaithBench Challenge Level Missing
**Problem**: Not focusing on challenging cases where SOTA models disagree
**Impact**: Testing on easier cases, not representative of benchmark difficulty
**Priority**: 🟢 Medium - Enhancement Opportunity

---

## Detailed Implementation Tasks

### Task 1: Fix TruthfulQA Evaluation Logic
**File**: `coherify/benchmarks/native_metrics.py`
**Function**: `TruthfulQAMetrics._check_truthfulness()`

**Current Logic (Wrong)**:
```python
# Treats best_answer as what model should generate
if TruthfulQAMetrics._fuzzy_match(best_answer, prediction):
    return True
```

**Required Fix**:
```python
# Check for known misinformation first
for incorrect in incorrect_answers:
    if fuzzy_match(incorrect, prediction):
        return False  # Contains misinformation
        
# Then check alignment with truthful answers
for correct in correct_answers:
    if fuzzy_match(correct, prediction):
        return True
        
return len(incorrect_answers) == 0  # Truthful if no misinformation
```

**Validation**: Results should show ~20% truthfulness (matching our current results) but with correct logic

### Task 2: Implement SelfCheckGPT Consistency Algorithms
**Files**: 
- `coherify/benchmarks/selfcheckgpt.py`
- New: `coherify/measures/consistency.py`

**Required Methods**:
1. **Multi-response generation**: `generate_multiple_responses(prompt, n=5)`
2. **BERTScore consistency**: `selfcheck_bertscore(responses)`  
3. **NLI consistency**: `selfcheck_nli(responses)`
4. **N-gram consistency**: `selfcheck_ngram(responses)`

**Integration**: Consistency scores should correlate with coherence measures

### Task 3: Add FEVER Evidence Chain Support
**File**: `coherify/benchmarks/fever_adapter.py`

**Required Features**:
1. **Multi-sentence evidence**: Handle 31.75% of claims requiring multiple sentences
2. **Cross-page evidence**: Handle 12.15% requiring multiple Wikipedia pages  
3. **Evidence composition**: Logical reasoning across evidence pieces
4. **Evidence type detection**: Classify evidence requirements

**Implementation**:
```python
def get_evidence_requirements(claim: str) -> Dict[str, Any]:
    return {
        "evidence_type": "single|multi_sentence|multi_page",
        "sentence_count": int,
        "page_count": int,
        "requires_composition": bool
    }
```

### Task 4: Enhance FaithBench Challenge Filtering
**File**: `coherify/benchmarks/faithbench_adapter.py`

**Required Features**:
1. **Challenge level filtering**: Focus on cases where SOTA disagrees
2. **Difficulty scoring**: Measure detection difficulty per sample
3. **Model disagreement tracking**: Cases where multiple detection methods conflict

**Expected Performance**: ~50% accuracy baseline (matching paper results)

### Task 5: Recalibrate Performance Expectations
**Files**: 
- `examples/comprehensive_benchmark_demo.py`
- `coherify/reporting/comprehensive_results.py`
- All benchmark adapters

**Required Changes**:
1. **Realistic baselines**: Align with published paper results
2. **Performance warnings**: Alert users to expected difficulty
3. **Improvement focus**: Frame coherence as improvement tool, not replacement
4. **Correlation metrics**: Show coherence-performance relationships

---

## Validation Checklist

### TruthfulQA Validation
- [ ] Baseline truthfulness ~58% (GPT-3 level) or lower
- [ ] Higher coherence does not automatically mean higher truthfulness  
- [ ] Results align with misconception patterns in paper
- [ ] Contrastive evaluation (correct vs incorrect answers) works properly

### SelfCheckGPT Validation
- [ ] Multiple response generation implemented
- [ ] Consistency scores decrease with hallucination rate
- [ ] BERTScore, NLI, and N-gram variants all functional
- [ ] AUC-PR scores in 70-75% range for WikiBio-style datasets

### FEVER Validation  
- [ ] Multi-sentence evidence handling (31.75% of claims)
- [ ] Cross-page evidence retrieval (12.15% of claims)
- [ ] Evidence composition for complex claims (16.82%)
- [ ] Baseline accuracy ~32% with correct evidence

### FaithBench Validation
- [ ] Challenge level filtering implemented
- [ ] Focus on cases where SOTA detection models disagree
- [ ] Baseline detection accuracy ~50% (matching paper)
- [ ] Coherence correlation with faithfulness measured

---

## Success Criteria

### Technical Success
1. **All benchmarks match published baseline results**
2. **Coherence measures show meaningful correlation with benchmark metrics**
3. **Implementation follows paper methodologies accurately**
4. **Performance expectations are realistic and documented**

### User Success  
1. **Clear understanding that coherence ≠ correctness**
2. **Realistic expectations about benchmark difficulty**
3. **Focus on relative improvement rather than absolute performance**
4. **Proper interpretation of coherence as a filtering/ranking tool**

---

## Timeline Estimate

### Week 1: Critical Fixes
- Fix TruthfulQA evaluation logic
- Update performance expectations in reporting
- Add realistic baseline documentation

### Week 2: Core Implementations
- Implement SelfCheckGPT consistency methods
- Add FEVER evidence chain support  
- Basic FaithBench challenge filtering

### Week 3: Validation & Integration
- Validate all fixes against paper results
- Update comprehensive demo with correct expectations
- Full documentation update

### Week 4: Testing & Polish
- Comprehensive testing of all benchmark fixes
- User feedback integration
- Final documentation and examples

---

## Notes for Implementation

### Key Principles
1. **Paper-first approach**: Always reference original methodology
2. **Realistic expectations**: Don't oversell coherence capabilities
3. **Correlation over causation**: Focus on coherence-performance relationships
4. **Validation required**: All changes must match published results

### Common Pitfalls to Avoid
1. **Coherence bias**: Assuming coherent = correct
2. **Overfitting**: Tuning coherence measures to benchmark performance
3. **Cherry-picking**: Showing only positive correlation examples
4. **Baseline inflation**: Making baseline performance unrealistically high

This systematic approach ensures our benchmark implementations are methodologically sound and provide valid evaluation frameworks for coherence measures.