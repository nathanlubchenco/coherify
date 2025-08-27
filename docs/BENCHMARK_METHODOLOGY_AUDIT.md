# Benchmark Methodology Audit Results

## Summary
Systematic audit of all benchmark implementations revealed that most benchmarks are measuring coherence scores instead of actual benchmark performance metrics. This fundamentally undermines the research goal of showing that coherence IMPROVES benchmark performance.

## Audit Results

### ❌ BROKEN: TruthfulQA (`run_truthfulqa_benchmark.py`)
**Problem:**
- Only measures coherence scores (0.429, 0.600, etc.)
- Shows 0% truthfulness and 0% accuracy
- Not actually evaluating if responses are truthful

**What it does:**
```python
# WRONG - Just measuring coherence
coherence_result = self.coherence_measure.compute(prop_set)
evaluation["coherence_score"] = coherence_result.score
```

**What it should do:**
```python
# RIGHT - Evaluate truthfulness, then use coherence to improve
truthfulness = evaluate_truthfulness(response)  # Is it true?
informativeness = evaluate_informativeness(response)  # Is it helpful?
score = truthfulness * informativeness  # Combined metric
```

### ❌ BROKEN: FEVER (`run_fever_benchmark.py`)
**Problem:**
- Only measures coherence scores
- Shows 0% label accuracy
- Not actually classifying claims as SUPPORTS/REFUTES/NOT ENOUGH INFO

**Evidence:**
```
FEVER-Specific Metrics:
  Label Accuracy: 0.0%  ← Not working!
  Evidence Consistency: 0.000
  FEVER Score: 0.000
```

**Status:** ✅ Fixed in `run_fever_benchmark.py`

### ✅ FIXED: FaithBench (`run_faithbench_benchmark.py`)
**Previously:**
- Only measured coherence scores
- Showed 0% prediction accuracy
- Not detecting hallucinations

**Now:**
- Properly detects hallucinations
- Reports F1 scores, precision, recall
- Shows realistic performance metrics

### ✅ CORRECT: HaluEval (`coherify/benchmarks/halueval.py`)
**Working Correctly:**
- Actually evaluates hallucination detection accuracy
- Compares selection methods properly
- Reports F1 scores, precision, recall

**Evidence:**
```python
# Correctly checks if hallucination was detected
is_hallucination_detected = self._check_hallucination(
    result.selected_response, sample
)
# Updates confusion matrix properly
if sample.is_hallucinated and pred:
    true_positives += 1
```

### ❌ BROKEN: Multi-Format Benchmarks (`run_multi_format_benchmarks.py`)
**Problem:**
- Only measures coherence scores across different formats
- No actual task performance metrics
- Missing accuracy for GSM8K, HellaSwag, MMLU

**Evidence:**
```python
# WRONG - Just collecting coherence scores
"coherence_scores": sample_coherence,
"subject_coherence_scores": {}  # Not task accuracy!
```

### ❌ BROKEN: Comprehensive Demo (`comprehensive_benchmark_demo.py`)
**Problem:**
- Has 3-stage structure but still measures coherence
- Shows 0% baseline accuracy
- Warning messages about unrealistic performance

**Evidence:**
```
Native TruthfulQA Metrics:
  Truthfulness: 0.000
  ⚠️ Performance Warning: Performance 0.0% is unrealistically low
  Baseline Accuracy: 0.000
  Coherence-Filtered: 0.000
```

## Root Cause Analysis

### The Fundamental Mistake
All broken benchmarks share the same pattern:
1. Convert samples to PropositionSets
2. Compute coherence scores
3. Report coherence as the metric
4. **MISSING: Never evaluate actual task performance**

### What's Missing
```python
# Every benchmark needs this pattern:

def evaluate_benchmark_performance(response, ground_truth):
    """Evaluate ACTUAL benchmark metric"""
    # For TruthfulQA: is it truthful?
    # For FEVER: is classification correct?
    # For GSM8K: is the math answer right?
    # For MMLU: is the multiple choice correct?
    return is_correct

def run_3_stage_pipeline(sample):
    # Stage 1: Baseline
    single_response = generate_one(sample)
    baseline_score = evaluate_benchmark_performance(single_response)
    
    # Stage 2: Majority Voting
    k_responses = generate_k(sample, k=5)
    majority_answer = majority_vote(k_responses)
    majority_score = evaluate_benchmark_performance(majority_answer)
    
    # Stage 3: Coherence Selection
    best_response = coherence_select(k_responses)
    coherence_score = evaluate_benchmark_performance(best_response)
    
    return {
        "baseline": baseline_score,      # e.g., 45% accurate
        "majority": majority_score,      # e.g., 52% accurate
        "coherence": coherence_score     # e.g., 58% accurate
    }
```

## Impact Assessment

### Research Validity
- **Current state**: Research claims cannot be validated
- **Impact**: Cannot prove coherence improves performance
- **Fix required**: Must evaluate actual metrics

### Benchmark Coverage
| Benchmark | Status | Actual Metric | Currently Measuring |
|-----------|--------|--------------|-------------------|
| TruthfulQA | ❌ Broken | Truthfulness×Informativeness | Coherence only |
| FEVER | ✅ Fixed | Classification Accuracy | Now measures accuracy |
| FaithBench | ✅ Fixed | Hallucination F1 Score | Now measures F1 |
| HaluEval | ✅ Working | Detection Accuracy | Correctly implemented |
| GSM8K | ❌ Broken | Math Answer Accuracy | Coherence only |
| MMLU | ❌ Broken | Multiple Choice Accuracy | Coherence only |
| HellaSwag | ❌ Broken | Completion Accuracy | Coherence only |

## Fix Priority

### Immediate (High Priority)
1. **TruthfulQA** - Core benchmark for the research
2. **Comprehensive Demo** - Main demonstration script

### Important (Medium Priority)
3. **Multi-Format Benchmarks** - Covers GSM8K, MMLU, HellaSwag
4. **Pipeline Comparison** - Needs proper metrics

### Already Fixed
- ✅ FEVER (run_fever_benchmark_fixed.py)
- ✅ FaithBench (run_faithbench_benchmark_fixed.py)
- ✅ HaluEval (correctly implemented from start)

## Implementation Template

Every benchmark needs this structure:

```python
class BenchmarkEvaluator:
    def evaluate_single_response(self, sample, response):
        """Evaluate actual benchmark metric for one response"""
        # Extract answer from response
        answer = self.extract_answer(response)
        
        # Check if correct
        is_correct = self.check_correctness(answer, sample.ground_truth)
        
        return is_correct  # Returns True/False or score
    
    def run_baseline(self, samples):
        """Stage 1: Single response"""
        correct = 0
        for sample in samples:
            response = model.generate(sample.question)
            if self.evaluate_single_response(sample, response):
                correct += 1
        return correct / len(samples)
    
    def run_majority_voting(self, samples, k=5):
        """Stage 2: K responses with voting"""
        correct = 0
        for sample in samples:
            responses = [model.generate(sample.question) for _ in range(k)]
            answers = [self.extract_answer(r) for r in responses]
            majority_answer = Counter(answers).most_common(1)[0][0]
            if self.check_correctness(majority_answer, sample.ground_truth):
                correct += 1
        return correct / len(samples)
    
    def run_coherence_selection(self, samples, k=5):
        """Stage 3: K responses with coherence selection"""
        correct = 0
        for sample in samples:
            responses = model.generate_k(sample.question, k)
            best = coherence_selector.select(responses)
            if self.evaluate_single_response(sample, best):
                correct += 1
        return correct / len(samples)
```

## Verification Tests

Run these to verify fixes:

```bash
# Should show actual accuracy percentages, not coherence scores

# FEVER - Should show SUPPORTS/REFUTES accuracy
python examples/run_fever_benchmark_fixed.py --compare

# FaithBench - Should show hallucination detection F1
python examples/run_faithbench_benchmark_fixed.py --compare

# TruthfulQA - Should show truthfulness scores (when fixed)
python examples/run_truthfulqa_benchmark_fixed.py --compare

# All should show pattern like:
# Baseline: 45%
# Majority: 52% (+7%)
# Coherence: 58% (+13%)
```

## Conclusion

The majority of benchmarks are fundamentally broken - they measure coherence instead of performance. This must be fixed to validate the research hypothesis that coherence-based selection improves benchmark performance. The fix pattern is clear and has been successfully applied to FEVER and FaithBench. The same pattern needs to be applied to all remaining benchmarks.