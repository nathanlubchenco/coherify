# Task: Enhance Proposition Extraction Foundations

**Priority:** CRITICAL  
**Status:** Major Gap  
**Objective 6:** Iterating on the foundations of proposition extraction  

## Current State
❌ Only basic sentence splitting in `PropositionSet.from_qa_pair()` (line 43: `answer.split(".")`)  
❌ No sophisticated claim extraction algorithms  
❌ No evaluation of extraction quality  
❌ Fundamental dependency for all other objectives  

## Problem Analysis
Proposition extraction is the foundation of all coherence evaluation. Poor extraction leads to:
- Inaccurate coherence measurements
- Missing logical relationships
- Poor benchmark performance
- Inability to handle complex texts

## Implementation Strategy

### Phase 1: Enhanced Sentence Segmentation
Replace naive `.split(".")` with proper linguistic segmentation.

```python
class PropositionExtractor:
    def __init__(self, method="spacy", language="en"):
        self.method = method
        self.nlp = self._init_nlp_pipeline(method, language)
    
    def extract_propositions(self, text: str, context: str = None) -> List[Proposition]:
        # 1. Sentence segmentation  
        # 2. Claim identification
        # 3. Proposition structuring
        # 4. Quality filtering
```

### Phase 2: Claim-Level Extraction
Move beyond sentences to extract meaningful claims and assertions.

### Phase 3: Quality Assessment
Evaluate extraction quality and filter low-quality propositions.

### Phase 4: Integration and Validation
Integrate with all existing benchmarks and measure improvement.

## Detailed Implementation

### 1. Linguistic Sentence Segmentation
- [ ] Add spaCy dependency for proper sentence boundaries
- [ ] Handle abbreviations, dates, numbers correctly
- [ ] Support multiple languages (start with English)
- [ ] Add sentence complexity filtering

```python  
def extract_sentences(self, text: str) -> List[str]:
    doc = self.nlp(text)
    sentences = []
    for sent in doc.sents:
        # Filter out fragments, questions, etc.
        if self._is_valid_sentence(sent):
            sentences.append(sent.text.strip())
    return sentences
```

### 2. Claim Identification and Extraction  
- [ ] Identify assertive statements vs questions/commands
- [ ] Extract subject-predicate-object triples
- [ ] Handle complex claims with multiple assertions
- [ ] Support conditional and causal claims

```python
def extract_claims(self, text: str) -> List[str]:  
    # Use dependency parsing to identify claims
    # Extract SVO triples and convert to natural language
    # Handle negations, conditionals, and complex structures
```

### 3. Proposition Structuring
- [ ] Convert raw claims to structured propositions  
- [ ] Add semantic roles (agent, action, object, etc.)
- [ ] Preserve logical relationships between propositions
- [ ] Add proposition metadata (confidence, complexity, etc.)

```python
@dataclass
class StructuredProposition(Proposition):
    subject: Optional[str] = None
    predicate: Optional[str] = None  
    object: Optional[str] = None
    semantic_roles: Dict[str, str] = field(default_factory=dict)
    logical_type: str = "assertion"  # assertion, question, conditional, etc.
    extraction_confidence: float = 1.0
```

### 4. Quality Assessment and Filtering
- [ ] Score proposition quality (completeness, clarity, informativeness)
- [ ] Filter fragments, duplicates, and low-quality extractions
- [ ] Add extraction confidence scores
- [ ] Implement proposition merging for similar claims

```python
def assess_proposition_quality(self, proposition: str) -> float:
    # Check completeness (has subject and predicate)
    # Check informativeness (not too generic)
    # Check clarity (not ambiguous pronouns)
    # Return confidence score 0-1
```

### 5. Advanced Extraction Methods

#### Claim Detection Models
- [ ] Fine-tune BERT/RoBERTa for claim detection
- [ ] Use pre-trained claim detection models if available
- [ ] Add claim boundary detection (multi-sentence claims)

#### Information Extraction Integration
- [ ] Use named entity recognition for better claim structure
- [ ] Add relation extraction for claim relationships  
- [ ] Support fact extraction from structured text

#### Template-Based Extraction
- [ ] Create extraction templates for common claim patterns
- [ ] Add domain-specific extraction rules (medical, legal, etc.)
- [ ] Support extraction from different text genres

## Integration Points

### PropositionSet Enhancement
```python
class PropositionSet:
    @classmethod  
    def from_text(cls, text: str, context: str = None, 
                  extractor: PropositionExtractor = None) -> "PropositionSet":
        """Extract propositions from raw text using sophisticated methods."""
        if extractor is None:
            extractor = PropositionExtractor()
        
        propositions = extractor.extract_propositions(text, context)
        return cls(propositions=propositions, context=context)
    
    @classmethod
    def from_qa_pair(cls, question: str, answer: str, 
                     extractor: PropositionExtractor = None) -> "PropositionSet":  
        """Enhanced QA proposition extraction."""
        # Use sophisticated extraction instead of naive splitting
        return cls.from_text(answer, context=question, extractor=extractor)
```

### Benchmark Integration
- Update all benchmark adapters to use enhanced extraction
- Add extraction quality metrics to benchmark results
- Compare coherence scores before/after extraction improvement

## Evaluation and Validation

### Extraction Quality Metrics
- [ ] **Completeness**: Are all important claims extracted?
- [ ] **Accuracy**: Are extracted claims factually correct?
- [ ] **Coherence Impact**: Do better extractions improve coherence scores?
- [ ] **Human Agreement**: Do humans agree with extracted propositions?

### Validation Datasets
- [ ] Create gold-standard proposition extractions for sample texts
- [ ] Use existing claim detection datasets for validation
- [ ] Test on diverse text types (academic, news, social media, etc.)

### Performance Benchmarking
- [ ] Measure extraction speed vs quality trade-offs
- [ ] Compare different extraction methods
- [ ] Test scalability on large documents

## Success Criteria
- Sophisticated proposition extraction replaces naive sentence splitting
- Extraction quality metrics show significant improvement
- Coherence evaluation improves across all benchmarks  
- System handles diverse text types and domains
- Performance remains acceptable (< 1s per paragraph)
- Clear documentation and examples for different extraction methods

## Files to Create/Modify
- `coherify/extraction/` - New package for proposition extraction
  - `__init__.py` - Package exports
  - `base_extractor.py` - Abstract base classes
  - `linguistic_extractor.py` - spaCy-based extraction
  - `claim_extractor.py` - Claim-focused extraction  
  - `quality_assessor.py` - Extraction quality evaluation
- `coherify/core/base.py` - Add StructuredProposition class
- `coherify/core/base.py` - Enhance PropositionSet with new methods
- `coherify/benchmarks/*.py` - Update all adapters to use enhanced extraction
- `tests/test_extraction.py` - Comprehensive extraction tests
- `examples/extraction_demo.py` - Demonstrate extraction capabilities

## Dependencies
- **spaCy** with English model (`python -m spacy download en_core_web_sm`)
- **Optional**: Transformers models for claim detection
- **Optional**: Named entity recognition models
- **Optional**: Relation extraction models

## Implementation Priority

### Week 1: Linguistic Foundation (Critical)
- spaCy integration for proper sentence segmentation
- Basic claim identification using dependency parsing
- Replace naive splitting across all adapters

### Week 2: Quality and Structure (High)  
- Proposition quality assessment
- Structured proposition representation
- Filtering and confidence scoring

### Week 3: Advanced Methods (Medium)
- Template-based extraction  
- Claim detection models
- Domain-specific extraction rules

### Week 4: Integration and Validation (High)
- Update all benchmarks with enhanced extraction
- Validation against gold standards
- Performance optimization

## Estimated Effort
**6-8 days** - Foundational component affecting all other work

## Dependencies on Other Tasks
- **Blocks**: All benchmark evaluation tasks (02-06)
- **Enables**: Accurate coherence measurement across all objectives
- **Requires**: Fixed test suite (Task 01) for validation