# Research Pipeline Guide

## 🎯 Research Objective

**Use coherence theory to improve response selection in multi-generation scenarios for better factual accuracy.**

We are NOT trying to improve benchmark evaluation methods themselves. Instead, we use coherence to improve how we select the best response from multiple generations.

## Three-Stage Research Pipeline

### Stage 1: Official Baselines ✅
**Purpose**: Validation that our benchmark implementations are correct
- Faithfully reproduce official evaluation methods (GPT-judge, BLEURT, BERTScore)
- Validate against published results 
- This establishes that our experimental setup is sound

### Stage 2: K-Pass Majority Voting Baseline
**Purpose**: Fair comparison baseline for coherence methods
- Generate K responses per question/prompt
- Use simple majority voting to determine final answer
- Must beat single-generation to be meaningful
- This becomes our baseline that coherence must outperform

### Stage 3: Coherence-Enhanced Response Selection
**Purpose**: Our research contribution
- Generate K responses at multiple temperatures
- Apply coherence measures for intelligent selection (not naive voting)
- **Core insight**: Truth appears more consistently across temperature variations
- Compare against Stage 2 majority voting (fair comparison)

## Overview

Coherify implements adapters for several popular benchmarks (TruthfulQA, FEVER, SelfCheckGPT) to evaluate coherence-based approaches to hallucination detection. However, our evaluation methodology differs from the original benchmarks in important ways.

## TruthfulQA

### Original Implementation
- **Evaluation Method**: GPT-judge (fine-tuned GPT-3) or BLEURT scores
- **Accuracy**: GPT-judge achieves 90-95% agreement with human evaluation
- **Metrics**: Binary truthfulness and informativeness per question
- **Published Baselines**: GPT-3 (175B): 58% truthful, Human: 94% truthful

### Our Implementation
- **Default Method**: Fuzzy string matching (threshold=0.3)
- **Improved Method**: Sentence embedding similarity (cosine similarity)
- **Key Differences**:
  - We don't use GPT-judge (requires fine-tuned GPT-3)
  - We don't calculate BLEURT scores
  - Our scores will differ from published results

### Impact on Results
```python
# Original TruthfulQA might score:
"Nothing bad happens" -> TRUTHFUL (via GPT-judge)

# Our fuzzy matching might score:
"Nothing bad happens" -> UNTRUTHFUL (doesn't match exact reference)

# Our embedding similarity scores:
"Nothing bad happens" -> TRUTHFUL (high semantic similarity)
```

### Recommended Usage
```python
from coherify.benchmarks.truthfulqa_evaluator import ImprovedTruthfulQAEvaluator

# Use embedding-based evaluation for better approximation
evaluator = ImprovedTruthfulQAEvaluator(use_embeddings=True)
result = evaluator.evaluate_truthfulness(prediction, sample)

# Note: result.is_truthful is our approximation, not official TruthfulQA score
```

## FEVER (Fact Extraction and VERification)

### Original Implementation
- **Evidence Retrieval**: From 5.4M Wikipedia pages
- **Labels**: SUPPORTS, REFUTES, NOT ENOUGH INFO
- **Metric**: FEVER Score = Label Accuracy × Evidence Accuracy
- **Published Baseline**: ~68.5% FEVER score for best systems

### Our Implementation
- **Evidence**: Assumes evidence is provided with the sample
- **Focus**: Coherence between claim and evidence
- **Key Differences**:
  - We don't implement full Wikipedia retrieval
  - We focus on coherence rather than strict label matching
  - Our "FEVER score" is not the official metric

### Impact on Results
- Our scores measure coherence-based fact verification
- Not comparable to published FEVER leaderboard scores
- Useful for relative comparisons between models

## SelfCheckGPT

### Original Implementation
- **Method**: Generate multiple samples, measure consistency
- **Metrics**: BERTScore, Question Answering, N-gram overlap
- **Evaluation**: AUC-PR for hallucination detection
- **Published Baseline**: ~0.74 AUC-PR

### Our Implementation
- **Method**: PropositionSet coherence across multiple samples
- **Metrics**: Semantic and entailment coherence
- **Key Differences**:
  - We use coherence measures instead of BERTScore
  - Different consistency calculation
  - Our AUC-PR not directly comparable

## Comparison Table

| Benchmark | Official Method | Our Method | Comparable? |
|-----------|----------------|------------|-------------|
| TruthfulQA | GPT-judge/BLEURT | Embedding similarity | ❌ No |
| FEVER | Wikipedia retrieval + strict matching | Coherence scoring | ❌ No |
| SelfCheckGPT | BERTScore/QA/N-gram | Coherence measures | ❌ No |

## How to Validate

### 1. Small-Scale Comparison
```python
# Run on a small subset with known ground truth
from coherify.benchmarks import TruthfulQAEvaluator
from coherify.measures import SemanticCoherence

evaluator = TruthfulQAEvaluator(coherence_measure=SemanticCoherence())
results = evaluator.evaluate_dataset(test_samples)

# Compare trends, not absolute scores
```

### 2. Use for Relative Comparisons
```python
# Compare models A vs B using same evaluation
model_a_coherence = evaluate_with_coherify(model_a_outputs)
model_b_coherence = evaluate_with_coherify(model_b_outputs)

# Valid: Model A is more coherent than Model B
# Invalid: Model A achieves 72% on TruthfulQA (can't claim official score)
```

### 3. Document Your Methodology
When reporting results:
```
"Using Coherify's coherence-based evaluation (not official TruthfulQA scoring),
Model A showed 15% higher coherence than Model B on truthfulness tasks."
```

## Future Improvements

1. **Add Official Evaluators**: Integrate with official evaluation scripts when available
2. **Implement BLEURT**: Add BLEURT scoring as an option
3. **GPT-4 Judge**: Implement GPT-4 based evaluation as more accessible alternative
4. **Calibration Study**: Establish conversion factors between our scores and official scores

## Recommendations

### For Research
- Use Coherify for coherence-based analysis
- Don't claim official benchmark scores
- Always document evaluation methodology
- Consider running official evaluators in parallel

### For Development
- Use for rapid iteration and testing
- Focus on relative improvements
- Validate significant findings with official evaluators

### For Production
- Implement custom evaluation for your use case
- Use coherence as one signal among many
- Don't rely solely on benchmark approximations

## Code Example: Proper Usage

```python
from coherify.benchmarks.truthfulqa_evaluator import ImprovedTruthfulQAEvaluator
from coherify.measures import HybridCoherence

# Clear documentation of what we're measuring
print("Evaluating coherence-based truthfulness (not official TruthfulQA)")

# Use improved evaluator
evaluator = ImprovedTruthfulQAEvaluator(use_embeddings=True)

# Evaluate
results = evaluator.evaluate_dataset(predictions, samples)

# Report appropriately
print(f"Coherence-based truthfulness: {results['truthful_rate']:.2%}")
print(f"NOTE: Not comparable to official TruthfulQA scores")
print(f"Official scoring requires GPT-judge or BLEURT evaluation")

# Also measure coherence
coherence_measure = HybridCoherence()
coherence_scores = [coherence_measure.compute(p) for p in proposition_sets]
print(f"Mean coherence: {np.mean(coherence_scores):.3f}")
```

## Summary

Coherify's benchmark implementations are designed for:
- ✅ Coherence-based evaluation
- ✅ Relative model comparisons  
- ✅ Rapid development and testing
- ❌ Official benchmark scoring
- ❌ Leaderboard submissions
- ❌ Direct comparison with published results

Always clearly document that scores are from Coherify's coherence-based evaluation, not official benchmark scoring.