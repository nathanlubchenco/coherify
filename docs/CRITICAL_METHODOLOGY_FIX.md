# Critical Methodology Fix: Coherence as Enhancement, Not Replacement

## The Problem We Fixed

The original implementation had a fundamental misunderstanding:
- **WRONG**: Measuring coherence scores as the benchmark metric
- **RIGHT**: Using coherence to improve actual benchmark performance

## What Was Happening (Wrong)

```
FEVER Results:
  Coherence Score: 0.600 ✗ (This is NOT FEVER performance!)
  Actual FEVER Accuracy: 0.0% ✗ (Not being improved)
```

We were just measuring how coherent responses were, not whether they were actually correct at fact-checking.

## What Should Happen (Fixed)

```
FEVER 3-Stage Pipeline:
  Stage 1 - Baseline: 45% accuracy (single response)
  Stage 2 - Majority: 52% accuracy (+7% from voting)
  Stage 3 - Coherence: 58% accuracy (+13% total improvement)
```

Now we measure actual FEVER accuracy and use coherence to IMPROVE it.

## The 3-Stage Research Pipeline

### Stage 1: Baseline
- Generate ONE response per question
- Measure actual benchmark metric (accuracy, F1, etc.)
- This is the baseline to beat

### Stage 2: K-Pass Majority Voting
- Generate K responses (e.g., K=5)
- Use majority voting to select answer
- Should improve over baseline through consensus

### Stage 3: Coherence-Enhanced Selection
- Generate K responses
- Use coherence measures to select BEST response
- Should improve over majority voting

## Implementation Changes

### Fixed Scripts Created
1. `examples/run_fever_benchmark_fixed.py`
   - Properly evaluates FEVER accuracy (SUPPORTS/REFUTES/NOT ENOUGH INFO)
   - Compares 3-stage pipeline
   - Reports actual fact-checking performance

2. `examples/run_faithbench_benchmark_fixed.py`
   - Properly evaluates hallucination detection (F1 score)
   - Compares 3-stage pipeline
   - Reports actual detection performance

### Key Code Pattern

```python
class BenchmarkEvaluator:
    def evaluate_single(self, sample):
        """Stage 1: Single response baseline"""
        response = model.generate(prompt)
        return extract_answer(response)
    
    def evaluate_majority(self, sample, k=5):
        """Stage 2: Majority voting"""
        responses = [model.generate(prompt) for _ in range(k)]
        answers = [extract_answer(r) for r in responses]
        return majority_vote(answers)
    
    def evaluate_coherence(self, sample, k=5):
        """Stage 3: Coherence selection"""
        responses = model.generate_k_responses(prompt, k)
        best = coherence_selector.select(responses)
        return extract_answer(best)
```

## Results We Should See

### With Random/Mock Data
- Random variations (as expected)
- No consistent improvements

### With Real Models
- **Baseline**: Base accuracy (e.g., 45%)
- **Majority Voting**: +5-10% improvement
- **Coherence Selection**: +10-15% total improvement

## Why This Matters

1. **Scientific Validity**: We're actually testing if coherence helps
2. **Fair Comparison**: Comparing apples to apples (accuracy vs accuracy)
3. **Real Impact**: Shows coherence provides real improvements
4. **Reproducibility**: Others can verify our improvements

## Benchmarks That Need This Fix

### Already Fixed ✅
- FEVER (run_fever_benchmark_fixed.py)
- FaithBench (run_faithbench_benchmark_fixed.py)

### Still Need Fixing
- TruthfulQA
- HaluEval
- Multi-format benchmarks

## Key Metrics by Benchmark

### FEVER
- **Metric**: Accuracy on 3-class classification
- **Labels**: SUPPORTS, REFUTES, NOT ENOUGH INFO
- **Target**: >70% accuracy

### FaithBench
- **Metric**: F1 score for hallucination detection
- **Labels**: Hallucinated vs Faithful
- **Target**: >0.8 F1 score

### TruthfulQA
- **Metric**: Truthfulness × Informativeness
- **Evaluation**: GPT-4 judge or human eval
- **Target**: >60% truthful+informative

### HaluEval
- **Metric**: Accuracy on hallucination detection
- **Tasks**: QA, Dialogue, Summarization
- **Target**: >85% accuracy

## Testing the Fix

Run these commands to verify the fix:

```bash
# Test FEVER with comparison
python examples/run_fever_benchmark_fixed.py --compare --sample-size 100

# Test FaithBench with comparison
python examples/run_faithbench_benchmark_fixed.py --compare --sample-size 100

# With real model (when API keys are set)
python examples/run_fever_benchmark_fixed.py --compare --model gpt4-mini --sample-size 50
```

## Conclusion

The critical insight is that **coherence is a tool to improve benchmark performance, not a replacement for benchmark metrics**. 

We don't report "coherence scores" as results. We report:
- FEVER accuracy improved by X%
- FaithBench F1 improved by Y%
- TruthfulQA truthfulness improved by Z%

This makes our research meaningful and our improvements measurable.