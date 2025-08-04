# Coherify UI Development Lessons & Best Practices

## Session Summary
Successfully fixed major Shogenji coherence measure bug and completed comprehensive UI optimization. Transformed UI from basic prototype to professional-grade application with proper error handling and performance optimization.

## Critical Technical Lessons

### 🔧 Shogenji Coherence Measure Debugging
**Major Bug Fixed**: Shogenji measure was producing absurdly high scores (640M) for contradictory statements instead of < 1.

**Root Cause**: Probability estimation was fundamentally flawed:
- Individual probabilities too low (~0.001) 
- Product of probabilities became microscopic (1.56e-12)
- Division by tiny number created massive scores

**Solution Implemented**:
```python
# Fixed probability estimator with proper ranges
ConfidenceBasedProbabilityEstimator(
    baseline_prob=0.5,
    prob_range=(0.2, 0.8)  # Prevents extreme values
)

# Strong contradiction penalty in joint probability
if max_contradiction_score > 0.7:
    penalty = np.exp(-8 * max_contradiction_score * (1 + contradiction_count * 0.3))
    joint_prob = independence_prob * penalty
```

**Key Insight**: Shogenji measure C_S = P(H₁∧...∧Hₙ) / ∏P(Hᵢ) is extremely sensitive to probability estimation quality. Small errors in individual probabilities create exponentially wrong results.

### 🎨 Streamlit UI Development Best Practices

#### Testing UI in Headless Mode
```bash
# Start headless UI for testing
python -m streamlit run ui/coherence_app_v2.py --server.headless true --server.port 8501 &

# Check if running
curl -s -I http://localhost:8501 | head -1  # Should return HTTP/1.1 200 OK

# Kill UI processes
pkill -f streamlit

# Test UI modules directly without starting server
python -c "from ui.performance import get_advanced_measures; print(get_advanced_measures())"
```

#### Streamlit Caching Issues & Solutions
**Problem**: Code changes not reflected in UI due to aggressive caching
**Solutions**:
1. Restart entire UI server (most reliable)
2. Clear specific caches: `st.cache_data.clear()`
3. Use TTL on cache decorators: `@st.cache_data(ttl=300)`
4. Reload modules in testing: `importlib.reload(ui.performance)`

#### Performance Optimization Patterns
```python
# Model singleton pattern for expensive resources
class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def get_measure(self, measure_name, measure_class, *args, **kwargs):
        if measure_name not in self._models:
            self._models[measure_name] = measure_class(*args, **kwargs)
        return self._models.get(measure_name)

# Cached computation with tuple keys (immutable)
@st.cache_data(ttl=300)
def compute_coherence_optimized(propositions_tuple, context, selected_measures_tuple):
    # Implementation
```

### 🧪 Comprehensive Testing Strategies

#### Test All Measure Categories
```python
# Always test the complete pipeline
fast_measures = get_fast_measures()
slow_measures = get_slow_measures() 
advanced_measures = get_advanced_measures()
all_measures = {**fast_measures, **slow_measures, **advanced_measures}

# Verify measure count expectations
expected_counts = {"fast": 2, "slow": 1, "advanced": 1}
```

#### Test Coherence Measure Edge Cases
```python
# Essential test cases for any coherence measure
test_cases = {
    "contradictory": {
        "statements": ["Sunny", "Raining", "Perfect picnic", "Stay indoors"],
        "expected_score": "< 1",
        "context": "Weather conditions"
    },
    "coherent": {
        "statements": ["Temp risen 1.1°C", "Due to emissions", "Scientists agree"],
        "expected_score": "> 1", 
        "context": "Climate change"
    },
    "neutral": {
        "statements": ["Sky is blue", "Pizza from Italy", "Purple favorite color"],
        "expected_score": "≈ 1",
        "context": "Random facts"
    }
}
```

### 📊 UI Architecture Insights

#### Measure Loading Pipeline
```python
# Critical: compute function must include ALL measure types
def compute_coherence_optimized(propositions_tuple, context, selected_measures_tuple):
    # BUG: Missing advanced_measures caused Shogenji to never compute
    all_measures = {**fast_measures, **slow_measures, **advanced_measures, **api_measures}
```

#### Error Handling & User Feedback
```python
# Graceful degradation pattern
try:
    measure = create_measure()
    if measure:
        measures[name] = {"measure": measure, "description": "..."}
except Exception as e:
    print(f"Failed to load {name}: {e}")  # Log but continue
    
# UI feedback for partial failures
missing = [name for name in expected if name not in available_measures]
if missing:
    st.warning(f"⚠️ Some measures failed to load: {', '.join(missing)}")
```

#### Professional UI Design Principles
- **No emoji in professional mode**: Business-appropriate interface
- **Colorblind-aware palettes**: Use tools like Viridis, avoid red/green only
- **Progressive disclosure**: Fast → Balanced → Complete → Advanced modes
- **Real-time feedback**: Show loading states, computation times, error details

### 🔬 Advanced Coherence Theory Insights

#### Unbounded vs Bounded Measures
- **Traditional measures** (Shogenji): Unbounded, philosophically meaningful
- **Modern NLP measures**: Bounded 0-1, user-friendly but lose theoretical precision
- **UI handling**: Special formatting and educational explanations for unbounded measures

#### Probability Estimation Challenges
1. **Language Model Probabilities**: Often miscalibrated, need normalization
2. **Contradiction Detection**: NLI models excel at this, use for joint probability penalties
3. **Independence Assumptions**: Geometric mean better than product for stability

### 🚀 Development Workflow Best Practices

#### Systematic Debugging Process
1. **Test individual components**: Probability estimator → Coherence measure → UI integration
2. **Use detailed logging**: Print intermediate calculations, not just final results
3. **Test edge cases first**: Contradictions, empty inputs, single propositions
4. **Verify mathematical correctness**: Check that C_S formula produces expected ranges

#### Code Quality Maintenance
```bash
# Pre-commit routine
black coherify/          # Code formatting
flake8 coherify/         # Linting
mypy coherify/           # Type checking
pytest tests/            # Run test suite
```

#### Documentation Strategy
- **Technical context**: Document mathematical formulas and theoretical basis
- **User guidance**: Provide interpretation of unbounded scores
- **Development notes**: Record debugging insights and architectural decisions

## Key Files Modified This Session

### Core Implementation
- `coherify/measures/shogenji.py`: Complete probability estimation rewrite
- `ui/performance.py`: Updated to use improved Shogenji measure
- `ui/coherence_app_v2.py`: Fixed compute function to include all measure types

### Critical Code Patterns Established
```python
# Probability estimator with contradiction detection
class ConfidenceBasedProbabilityEstimator(ProbabilityEstimator):
    def __init__(self, baseline_prob=0.5, prob_range=(0.2, 0.8)):
        # Prevents extreme probability values
        
    def estimate_joint_probability(self, propositions, context=None):
        # Uses NLI for pairwise contradiction detection
        # Applies exponential penalty for strong contradictions
```

## Future Development Guidelines

1. **Always test contradictory cases** when modifying coherence measures
2. **Use headless UI testing** for automated verification 
3. **Clear Streamlit cache** after significant code changes
4. **Implement progressive loading** for expensive models
5. **Provide educational context** for complex theoretical measures
6. **Test complete pipeline** from UI interaction to final results

## Commands for Quick Testing
```bash
# Test all measures load correctly
python -c "from ui.performance import get_advanced_measures; print(len(get_advanced_measures()))"

# Test Shogenji with contradictory statements  
python -c "from coherify.measures.shogenji import *; from coherify.core.base import *; props=[Proposition('Sunny'), Proposition('Raining')]; result=ShogunjiCoherence().compute(PropositionSet(props)); print(f'Score: {result.score:.4f}')"

# Start UI for testing
python -m streamlit run ui/coherence_app_v2.py --server.headless true --server.port 8501 &

# Verify UI accessibility
curl -s -I http://localhost:8501
```

This session demonstrated that systematic debugging, comprehensive testing, and understanding the underlying mathematics are essential for building reliable coherence analysis tools.