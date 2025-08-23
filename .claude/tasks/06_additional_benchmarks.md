# Task: Add Additional Hallucination-Related Benchmarks

**Priority:** MEDIUM  
**Status:** Infrastructure Ready  
**Objective 5:** Adding additional hallucination related benchmarks  

## Current State
✅ Generic multi-format adapter framework (`multi_format_adapters.py`)  
✅ FaithBench adapter already implemented  
✅ Clear patterns for benchmark integration established  
✅ TruthfulQA and FEVER as working examples  

## High-Priority Hallucination Benchmarks

### 1. HaluEval (Comprehensive Hallucination Evaluation)
**Focus**: Multi-task hallucination evaluation across domains
- [ ] Implement HaluEval adapter for QA, summarization, dialogue
- [ ] Support HaluEval's fine-grained hallucination categories  
- [ ] Add domain-specific coherence evaluation
- [ ] Create category-wise analysis tools

### 2. TRUE (Towards Richer Understanding and Evaluation)
**Focus**: Knowledge-grounded text generation evaluation
- [ ] Implement TRUE benchmark adapter
- [ ] Add knowledge-grounding coherence evaluation
- [ ] Support fact-checking against knowledge bases
- [ ] Integrate with retrieval-augmented coherence

### 3. FactualityPrompts (Factual Error Detection)  
**Focus**: Factual consistency in generated text
- [ ] Implement FactualityPrompts adapter
- [ ] Add fact-checking coherence measures
- [ ] Support claim-level factuality evaluation
- [ ] Integration with external knowledge verification

### 4. SelfCheckGPT Enhanced (Self-Consistency Evaluation)
**Focus**: Improved self-consistency checking
- [ ] Enhance existing SelfCheckGPT adapter  
- [ ] Add multiple sampling strategies
- [ ] Implement consistency-based coherence measures
- [ ] Support different self-check prompting strategies

### 5. PARENT (Hallucination in Table-to-Text)
**Focus**: Structured data hallucination detection  
- [ ] Implement PARENT benchmark adapter
- [ ] Add table-text coherence evaluation
- [ ] Support structured data grounding
- [ ] Create table-aware proposition extraction

## Implementation Strategy

### Phase 1: HaluEval Integration (Highest Impact)
HaluEval covers multiple tasks and provides comprehensive hallucination evaluation.

```python
class HaluEvalAdapter(MultiFormatBenchmarkAdapter):
    def __init__(self, task_type: str = "all"):  # qa, summarization, dialogue, all
        self.task_type = task_type
        self.task_adapters = {
            "qa": self._adapt_qa_sample,
            "summarization": self._adapt_summarization_sample, 
            "dialogue": self._adapt_dialogue_sample
        }
    
    def adapt_single(self, sample: Dict[str, Any]) -> PropositionSet:
        task = sample.get("task", "qa")
        return self.task_adapters[task](sample)
```

### Phase 2: TRUE/FactualityPrompts (Knowledge-Grounded)
Focus on benchmarks that require external knowledge validation.

### Phase 3: Enhanced SelfCheckGPT (Self-Consistency)
Improve existing SelfCheckGPT with better sampling and coherence measures.

## Benchmark Details and Implementation

### HaluEval Implementation
- **Dataset**: Available through Hugging Face datasets
- **Tasks**: QA, Summarization, Dialogue, Code generation
- **Hallucination Types**: Factual, logical, commonsense
- **Integration**: Use existing multi-format framework
- **Coherence Focus**: Task-specific coherence evaluation

### TRUE Benchmark Implementation  
- **Dataset**: Knowledge-grounded generation tasks
- **Knowledge Sources**: Wikipedia, structured KBs
- **Integration**: Extend existing retrieval framework
- **Coherence Focus**: Knowledge-grounding coherence

### FactualityPrompts Implementation
- **Dataset**: Factual claim datasets with labels
- **Validation**: External fact-checking APIs/databases
- **Integration**: Create fact-checking coherence measures  
- **Coherence Focus**: Factual consistency evaluation

### PARENT Implementation
- **Dataset**: Table-to-text generation with structured data
- **Structured Data**: Tables, lists, key-value pairs
- **Integration**: Create structured data proposition extraction
- **Coherence Focus**: Data-text alignment coherence

## New Coherence Measures for Benchmarks

### Knowledge-Grounding Coherence
```python  
class KnowledgeGroundingCoherence(CoherenceMeasure):
    def __init__(self, knowledge_base, retrieval_method="dense"):
        self.kb = knowledge_base
        self.retriever = retrieval_method
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        # 1. Retrieve relevant knowledge for each proposition
        # 2. Compute grounding score based on KB alignment  
        # 3. Return knowledge-grounded coherence
```

### Factual Consistency Coherence
```python
class FactualConsistencyCoherence(CoherenceMeasure):  
    def __init__(self, fact_checker="nli_based"):
        self.fact_checker = fact_checker
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        # 1. Extract factual claims from propositions
        # 2. Check each claim against reliable sources
        # 3. Return consistency-based coherence score
```

## Integration with Existing Infrastructure

### Leverage Current Components
- **Multi-format adapters** for diverse benchmark formats
- **API-enhanced evaluation** for generation tasks  
- **Temperature variance** for consistency evaluation
- **Caching system** for expensive knowledge lookups

### New Components Needed
- **Knowledge base integration** (Wikipedia API, structured KBs)
- **Fact-checking APIs** (Google Fact Check, claim verification)
- **Structured data processors** (table/JSON parsing)
- **Enhanced proposition extraction** for domain-specific content

## Success Criteria
- 3+ new hallucination benchmarks integrated and working
- Each benchmark shows different coherence patterns than existing ones
- Performance acceptable for research use (< 5 minutes per 100 samples)  
- Clear documentation and examples for each benchmark
- Integration with K-run majority voting and temperature analysis

## Implementation Priority Order

### Week 1: HaluEval (Multi-task Foundation)
- Provides broad coverage of hallucination types
- Tests multi-format adapter framework at scale
- High research impact

### Week 2: Enhanced SelfCheckGPT (Build on Existing)
- Improve existing implementation
- Add advanced self-consistency measures
- Validate self-check coherence approaches

### Week 3: TRUE/FactualityPrompts (Knowledge-Grounded)  
- Add external knowledge integration
- Create knowledge-grounding coherence measures
- Enable fact-checking workflows

### Week 4: PARENT (Structured Data)
- Add structured data coherence evaluation
- Create table-text alignment measures
- Complete diverse benchmark coverage

## Files to Create/Modify
- `coherify/benchmarks/halueval_adapter.py` - HaluEval integration
- `coherify/benchmarks/true_adapter.py` - TRUE benchmark
- `coherify/benchmarks/factuality_adapter.py` - FactualityPrompts
- `coherify/benchmarks/parent_adapter.py` - Table-to-text benchmark
- `coherify/measures/knowledge_grounding.py` - Knowledge-based coherence
- `coherify/measures/factual_consistency.py` - Fact-checking coherence
- `examples/run_halueval_benchmark.py` - HaluEval runner
- `examples/compare_hallucination_benchmarks.py` - Cross-benchmark analysis
- `tests/test_*_adapter.py` - Tests for each adapter

## Dependencies
- datasets library for benchmark access
- Knowledge base APIs (Wikipedia, Google Knowledge Graph)
- Fact-checking services (optional)
- Enhanced proposition extraction (Task 07)
- Fixed test suite (Task 01)

## Estimated Effort
**5-6 days** - 3-4 benchmarks with full integration