# Task: Add Additional Coherence Measures

**Priority:** MEDIUM  
**Status:** Well Positioned  
**Objective 4:** Adding additional coherence measures  

## Current State
✅ Excellent modular architecture for adding measures  
✅ Multiple measures implemented (Semantic, Entailment, Hybrid, Shogenji)  
✅ API-enhanced and multi-response variants  
✅ Clear extension patterns established  

## Opportunities for New Measures
Based on literature review and practical needs, prioritized list:

### High Priority Measures

#### 1. Graph-Based Coherence  
**Concept**: Model propositions as nodes in a graph, coherence as connectivity
- [ ] Implement proposition graph construction
- [ ] Add edge weights based on semantic/logical relationships  
- [ ] Calculate graph metrics (clustering coefficient, centrality, etc.)
- [ ] Create `GraphCoherence` class inheriting from `CoherenceMeasure`

#### 2. Probabilistic Coherence (Enhanced)
**Concept**: Improve on Shogenji with better probability estimation
- [ ] Add calibrated confidence-based probability estimator
- [ ] Implement proper conditional probability estimation
- [ ] Add Bayesian coherence measures (information theory based)
- [ ] Create `BayesianCoherence` class with multiple probability models

#### 3. Temporal Coherence
**Concept**: Coherence over time/sequence in multi-turn conversations
- [ ] Track coherence evolution across conversation turns
- [ ] Implement memory-based coherence (consistency with prior statements)
- [ ] Add forgetting/decay models for long conversations
- [ ] Create `TemporalCoherence` class for sequential evaluation

### Medium Priority Measures

#### 4. Causal Coherence
**Concept**: Evaluate causal relationships between propositions
- [ ] Use causal inference models to detect cause-effect relationships
- [ ] Implement causal graph construction
- [ ] Add causal consistency checking
- [ ] Create `CausalCoherence` class

#### 5. Uncertainty-Aware Coherence  
**Concept**: Factor in model uncertainty when computing coherence
- [ ] Implement uncertainty estimation (ensemble, dropout, etc.)
- [ ] Weight coherence by uncertainty levels
- [ ] Add uncertainty calibration
- [ ] Create `UncertaintyCoherence` class

#### 6. Multi-Modal Coherence
**Concept**: Coherence across text, images, and other modalities
- [ ] Add image-text coherence evaluation
- [ ] Implement cross-modal embedding alignment
- [ ] Support multi-modal proposition sets
- [ ] Create `MultiModalCoherence` class

## Implementation Strategy

### Phase 1: Graph-Based Coherence (Highest Impact)
```python
class GraphCoherence(CoherenceMeasure):
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 graph_metric: str = "clustering_coefficient"):
        # Build graph where edges represent coherence relationships
        # Calculate graph-theoretic measures of coherence
    
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        # 1. Build proposition graph
        # 2. Calculate graph metrics  
        # 3. Return coherence based on connectivity/structure
```

### Phase 2: Enhanced Probabilistic Coherence
- Build on existing `ShogunjiCoherence` 
- Add calibrated probability estimators
- Implement information-theoretic measures

### Phase 3: Temporal Coherence  
- Extend `PropositionSet` to support temporal metadata
- Implement conversation-aware coherence tracking
- Add to multi-turn benchmark evaluation

## Integration Points

### Easy Integration (Existing Patterns)
- All new measures inherit from `CoherenceMeasure`
- Add to `HybridCoherence` as additional components
- Include in benchmark evaluation workflows
- Support in approximation algorithms

### New Infrastructure Needed
- **Graph Processing**: NetworkX integration for graph measures
- **Multi-Modal Support**: Vision model integration  
- **Temporal Support**: Conversation state tracking
- **Uncertainty Estimation**: Model uncertainty quantification

## Success Criteria
- 3+ new coherence measures implemented and tested
- All measures integrate with existing benchmark workflows
- Performance benchmarking shows reasonable computation time
- Measures provide complementary insights (low correlation with existing measures)
- Documentation and examples for each measure

## Specific Implementation Tasks

### Graph-Based Coherence
- [ ] Add networkx dependency to pyproject.toml
- [ ] Implement proposition graph construction algorithms
- [ ] Add graph coherence metrics (clustering, centrality, connectivity)
- [ ] Create visualization tools for proposition graphs
- [ ] Add tests and integration with benchmarks

### Temporal Coherence
- [ ] Extend PropositionSet with temporal metadata support
- [ ] Implement conversation history tracking
- [ ] Add temporal consistency checking algorithms
- [ ] Create multi-turn evaluation examples
- [ ] Integration with chat/dialogue benchmarks

### Enhanced Probabilistic
- [ ] Research and implement calibrated confidence estimation
- [ ] Add information-theoretic coherence measures
- [ ] Improve conditional probability estimation
- [ ] Add Bayesian coherence variants
- [ ] Compare with existing Shogenji implementation

## Files to Create
- `coherify/measures/graph_coherence.py` - Graph-based measures
- `coherify/measures/temporal_coherence.py` - Temporal measures  
- `coherify/measures/bayesian_coherence.py` - Enhanced probabilistic measures
- `coherify/measures/uncertainty_coherence.py` - Uncertainty-aware measures
- `coherify/measures/causal_coherence.py` - Causal relationship measures
- `tests/test_*_coherence.py` - Tests for each measure
- `examples/compare_coherence_measures.py` - Comparative analysis

## Dependencies
- networkx (for graph measures)
- scipy.stats (for information theory)
- Additional ML models for specific measures
- Fixed test suite (Task 01)

## Estimated Effort  
**4-5 days** - 2-3 measures with full integration