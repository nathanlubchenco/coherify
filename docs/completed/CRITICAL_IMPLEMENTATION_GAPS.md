# Critical Implementation Gaps Analysis

## Executive Summary
Our current implementation fundamentally misunderstands the research methodology. We're treating coherence as a replacement metric instead of an enhancement layer on top of official benchmarks.

## Critical Gaps Identified

### 1. **Benchmark Evaluation Methods are Wrong**

**Current Implementation:**
- Using fuzzy string matching against correct/incorrect answer lists
- Computing coherence scores and using them to determine "correctness"
- Mock data and simulated evaluations

**What Should Happen:**
- **TruthfulQA**: Use GPT-judge or BLEURT for evaluation (NOT string matching)
- **FEVER**: Use evidence retrieval + label verification
- **SelfCheckGPT**: Use BERTScore for consistency across multiple generations

### 2. **Not Actually Running Model Inference**

**Current Implementation:**
- Using mock responses like "N/A"
- No actual model API calls happening
- Predictions array filled with placeholder data

**What Should Happen:**
- Actually call OpenAI/Anthropic APIs to generate responses
- Store real model outputs for evaluation
- Use configured models (gpt-4o-mini, etc.) to generate answers

### 3. **Research Pipeline Structure is Backwards**

**Current Implementation:**
```
Enhanced Evaluator → Coherence Scores → Determine Correctness
```

**What Should Happen:**
```
Stage 1: Official Benchmark → Baseline Score
Stage 2: Generate K responses → Majority Voting → K-pass Score
Stage 3: Generate K responses → Coherence Selection → Enhanced Score
Compare: Enhanced Score > K-pass Score > Baseline Score
```

### 4. **Missing K-pass Generation System**

**Current Implementation:**
- No multi-generation capability
- No majority voting implementation
- No coherence-based selection from multiple responses

**What Should Happen:**
- Generate K responses per question (K=5, 10, etc.)
- Stage 2: Select answer via majority voting
- Stage 3: Select answer via highest coherence score
- Compare performance across all three stages

## Required Fixes

### Fix 1: Connect Real Model APIs
```python
# Current (WRONG)
predictions = ["N/A"] * len(samples)

# Should be
provider = OpenAIProvider(model="gpt-4o-mini")
predictions = []
for sample in samples:
    response = provider.generate(sample["question"])
    predictions.append(response.text)
```

### Fix 2: Use Official Evaluation Methods
```python
# Current (WRONG)
is_truthful = fuzzy_match(prediction, correct_answers)

# Should be
evaluator = TruthfulQAOfficialEvaluator(method="gpt-judge")
result = evaluator.evaluate(prediction, sample)
is_truthful = result.is_truthful
```

### Fix 3: Implement Proper Research Pipeline
```python
# Stage 1: Baseline
baseline_score = run_official_benchmark(model, samples)

# Stage 2: K-pass Majority Voting
k_responses = [model.generate(q) for _ in range(K)]
majority_answer = majority_vote(k_responses)
kpass_score = evaluate(majority_answer)

# Stage 3: Coherence Selection
k_responses = [model.generate(q) for _ in range(K)]
coherence_scores = [compute_coherence(r) for r in k_responses]
best_response = k_responses[argmax(coherence_scores)]
enhanced_score = evaluate(best_response)

# Results should show: enhanced_score > kpass_score > baseline_score
```

### Fix 4: Update Evaluation Flow

**Remove:**
- `EnhancedTruthfulQAEvaluator` (it conflates coherence with evaluation)
- Mock data generation
- Fuzzy matching for truthfulness

**Add:**
- `ModelRunner` class to handle actual API calls
- `KPassGenerator` for multi-response generation
- `MajorityVotingSelector` for Stage 2
- `CoherenceSelector` for Stage 3
- Proper comparison reporting across all three stages

## Implementation Priority

1. **Immediate (Stage 1):**
   - Connect real model APIs (OpenAI/Anthropic)
   - Fix official benchmark evaluators to use proper methods
   - Generate actual model responses, not mock data

2. **Next (Stage 2):**
   - Implement K-pass generation system
   - Add majority voting selection
   - Create comparison reporting

3. **Finally (Stage 3):**
   - Add coherence-based selection
   - Implement temperature variation
   - Full pipeline comparison

## Key Insight

**Coherence is NOT a replacement for benchmark metrics.**

It's an enhancement that helps select better responses from multiple candidates. The research question is: "Can coherence-based selection outperform simple majority voting when choosing among K generated responses?"

The answer should be demonstrated by showing:
```
Baseline (single response) < K-pass (majority voting) < Coherence-enhanced (coherence selection)
```

This requires actually generating multiple responses and having a selection mechanism, not just computing coherence scores on existing data.
