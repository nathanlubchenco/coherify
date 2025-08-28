# Implementation Improvement Plan

Based on analysis of the official benchmark repositories and our new documentation, here are the key improvements we should implement:

## 1. TruthfulQA Improvements

### Current Gaps
- Not using the new improved multiple-choice format (Jan 2025 update)
- Missing BLEURT scoring (important metric from official repo)
- Not implementing the full metric suite (BLEU, ROUGE, BLEURT, GPT-judge)

### Action Items
```python
# Priority 1: Add Multiple-Choice Support
- [ ] Implement MC1 (single correct answer) evaluation
- [ ] Implement MC2 (multiple correct answers) evaluation
- [ ] Add new improved MC format from Jan 2025 update

# Priority 2: Complete Metric Suite
- [ ] Add BLEURT scoring (install: pip install bleurt-pytorch)
- [ ] Add ROUGE scoring for generation evaluation
- [ ] Add BLEU scoring for generation evaluation
- [ ] Ensure GPT-4o judge is properly calibrated

# Priority 3: Category-Specific Analysis
- [ ] Add per-category performance tracking (38 categories)
- [ ] Identify which categories benefit most from coherence
```

### Implementation Code
```python
class TruthfulQACompleteEvaluator:
    def __init__(self):
        self.metrics = {
            'bleu': BLEUMetric(),
            'rouge': ROUGEMetric(),
            'bleurt': BLEURTMetric(),
            'gpt_judge': GPT4Judge(model='gpt-4o'),
            'mc1': MultipleChoiceMetric(mode='single'),
            'mc2': MultipleChoiceMetric(mode='multiple')
        }

    def evaluate(self, response, sample):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.score(response, sample)
        return results
```

## 2. SelfCheckGPT Integration

### Current Gaps
- Not leveraging SelfCheckGPT's consistency checking with our coherence measures
- Missing the NLI method (best performing at 92.50% AUC-PR)
- Not using the LLM-Prompt method with GPT-4o

### Action Items
```python
# Priority 1: Install and Integrate SelfCheckGPT
- [ ] Install selfcheckgpt package
- [ ] Implement NLI-based consistency checking
- [ ] Add to our K-pass generation pipeline

# Priority 2: Hybrid Coherence-Consistency
- [ ] Combine SelfCheckGPT scores with our coherence measures
- [ ] Weight consistency vs coherence for optimal selection
- [ ] Test on wiki_bio_gpt3_hallucination dataset

# Priority 3: Prompt-Based Checking
- [ ] Implement GPT-4o prompt-based consistency checking
- [ ] Compare with SelfCheck-NLI performance
```

### Implementation Code
```python
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckLLMPrompt

class CoherenceConsistencySelector:
    def __init__(self):
        self.selfcheck_nli = SelfCheckNLI(device='cuda')
        self.selfcheck_prompt = SelfCheckLLMPrompt('gpt-4o')
        self.coherence_measure = SemanticCoherence()

    def select_response(self, responses, alpha=0.5):
        # Compute consistency scores
        consistency_scores = self.selfcheck_nli.predict(
            sentences=[r.text for r in responses],
            sampled_passages=responses[1:]  # Use other responses as samples
        )

        # Compute coherence scores
        coherence_scores = [
            self.coherence_measure.compute(r)
            for r in responses
        ]

        # Hybrid score: low consistency = high hallucination
        hybrid_scores = [
            alpha * (1 - consistency) + (1 - alpha) * coherence
            for consistency, coherence in zip(consistency_scores, coherence_scores)
        ]

        return responses[np.argmax(hybrid_scores)]
```

## 3. Temperature Strategy Optimization

### Insights from OpenAI Docs
- Temperature range 0-2 (not 0-1 as commonly assumed)
- GPT-4o and GPT-4o-mini both support full range
- Different optimal ranges for different tasks

### Action Items
```python
# Priority 1: Expand Temperature Range
- [ ] Test with higher temperatures (1.0-2.0) for diversity
- [ ] Use very low temperatures (0.1-0.3) for evaluation
- [ ] Profile optimal temperature per benchmark

# Priority 2: Adaptive Temperature
- [ ] Implement adaptive temperature based on question difficulty
- [ ] Use entropy-based temperature adjustment
- [ ] Test temperature scheduling across K responses
```

### Implementation Code
```python
class AdaptiveTemperatureGenerator:
    def __init__(self, base_temp=0.7):
        self.base_temp = base_temp

    def generate_k_responses(self, prompt, k=5, difficulty=None):
        # Adaptive temperature based on difficulty
        if difficulty == 'hard':
            temps = np.linspace(0.5, 1.5, k)  # Wider range for hard questions
        elif difficulty == 'easy':
            temps = np.linspace(0.3, 0.7, k)  # Narrower range for easy
        else:
            # Default: exponential spacing for diversity
            temps = self.base_temp * np.exp(np.linspace(-0.3, 0.3, k))

        responses = []
        for temp in temps:
            response = self.model.generate(prompt, temperature=temp)
            responses.append(response)
        return responses
```

## 4. FEVER Evidence Chain Verification

### Current Gaps
- Not implementing evidence retrieval
- Missing multi-hop reasoning verification
- No evidence chain coherence checking

### Action Items
```python
# Priority 1: Evidence Retrieval
- [ ] Implement Wikipedia passage retrieval
- [ ] Add evidence ranking mechanism
- [ ] Score evidence-claim alignment

# Priority 2: Multi-Hop Coherence
- [ ] Track evidence chains for multi-hop claims
- [ ] Verify coherence across evidence hops
- [ ] Detect contradictions in evidence chain
```

## 5. Cost and Performance Optimizations

### From OpenAI Model Updates
- GPT-4o is 92% cheaper than old GPT-4
- GPT-4o-mini is 94% cheaper and sufficient for many tasks
- Batch API available for 50% discount

### Action Items
```python
# Priority 1: Intelligent Model Selection
- [ ] Use GPT-4o-mini for K-pass generation
- [ ] Reserve GPT-4o for final evaluation only
- [ ] Implement fallback from GPT-4o to GPT-4o-mini on rate limits

# Priority 2: Batch Processing
- [ ] Implement batch API calls for K-pass generation
- [ ] Cache embeddings and NLI scores
- [ ] Parallelize independent evaluations
```

## 6. New Benchmark: HaluEval

### Why Add It
- 35,000 examples specifically for hallucination detection
- Combines general queries with task-specific examples
- Perfect fit for coherence-based selection

### Action Items
```python
# Priority 1: Basic Integration
- [ ] Download HaluEval dataset
- [ ] Create HaluEvalAdapter class
- [ ] Implement baseline evaluation

# Priority 2: Task-Specific Coherence
- [ ] Custom coherence for QA vs Dialogue vs Summary
- [ ] Compare with SelfCheckGPT on same data
```

## 7. Unified Evaluation Framework

### Goal
Create a unified framework that combines all insights:

```python
class UnifiedCoherenceEvaluator:
    def __init__(self):
        self.benchmarks = {
            'truthfulqa': TruthfulQACompleteEvaluator(),
            'selfcheckgpt': SelfCheckGPTEvaluator(),
            'fever': FEVEREvaluator(),
            'halueval': HaluEvalEvaluator()
        }

        self.methods = {
            'baseline': SingleResponseBaseline(),
            'majority': MajorityVotingSelector(),
            'coherence': CoherenceSelector(),
            'consistency': SelfCheckConsistencySelector(),
            'hybrid': HybridCoherenceConsistencySelector()
        }

        self.model_strategy = {
            'generation': 'gpt-4o-mini',  # Cost efficient
            'evaluation': 'gpt-4o',        # High quality
            'embedding': 'text-embedding-3-small'
        }
```

## Implementation Priority

### Phase 1: Core Improvements (Week 1)
1. Install selfcheckgpt and integrate NLI method
2. Expand temperature range testing (0.1-2.0)
3. Add BLEURT scoring to TruthfulQA

### Phase 2: Advanced Features (Week 2)
1. Implement hybrid coherence-consistency selection
2. Add adaptive temperature strategies
3. Integrate HaluEval benchmark

### Phase 3: Optimization (Week 3)
1. Implement batch processing
2. Add caching layer
3. Profile and optimize performance

## Expected Improvements

Based on the documentation and research:

| Method | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| TruthfulQA (Coherence) | 55% | 65% | +10% |
| TruthfulQA (Hybrid) | - | 70% | +15% |
| SelfCheckGPT Integration | - | 85% | New |
| Cost per evaluation | $0.10 | $0.02 | -80% |

## Testing Strategy

```bash
# Test each improvement independently
python tests/test_selfcheck_integration.py
python tests/test_temperature_expansion.py
python tests/test_hybrid_selection.py

# Run full benchmark comparison
python scripts/run_unified_benchmark.py \
    --benchmarks truthfulqa,selfcheckgpt,halueval \
    --methods baseline,majority,coherence,hybrid \
    --samples 100 \
    --k-responses 5
```

## Success Metrics

1. **Performance**: 10-15% improvement over baseline
2. **Cost**: 80% reduction through model optimization
3. **Speed**: 2x faster through batching and caching
4. **Coverage**: Support all major hallucination benchmarks
5. **Robustness**: Consistent improvements across all benchmarks

---

This plan incorporates all insights from the new documentation and provides a clear roadmap for enhancing Coherify's capabilities.
