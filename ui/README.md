# Coherify Interactive UI

A web-based interface for exploring coherence measures and understanding how they analyze text coherence.

## 🚀 Quick Start

### Launch the UI
```bash
# Option 1: Using the launcher script
python run_ui.py

# Option 2: Using make
make ui

# Option 3: Direct streamlit command
streamlit run ui/coherence_app.py
```

The app will be available at: http://localhost:8501

### Try the Demo
```bash
# See a command-line demo of the functionality
python demo_ui.py
```

## 🎯 Features

### 🔬 Coherence Workbench
The main interactive analysis tool where you can:

- **Input Custom Text**: Enter your own propositions to analyze
- **Use Preset Examples**: Choose from educational examples that demonstrate different coherence patterns
- **Compare Measures**: See how different coherence theories evaluate the same text
- **Real-time Analysis**: Get instant feedback as you modify text

#### Available Coherence Measures

1. **Semantic Coherence** 
   - Uses sentence embeddings to measure semantic similarity
   - Good for detecting topical coherence
   - Based on transformer models like MiniLM

2. **Entailment Coherence**
   - Uses Natural Language Inference (NLI) models
   - Detects logical relationships and contradictions
   - Effective for finding inconsistencies

3. **Hybrid Coherence**
   - Combines semantic and entailment measures
   - Provides balanced analysis
   - Weighted combination of multiple approaches

4. **Shogenji Coherence**
   - Traditional probability-based measure from philosophy
   - Theoretical foundation in formal coherence theory
   - May require additional model setup

### 📊 Visualizations

- **Score Comparison Charts**: Bar charts showing coherence scores across measures
- **Pairwise Similarity Heatmaps**: Matrix view of proposition relationships
- **Interactive Plots**: Powered by Plotly for rich interactions

### 📈 Benchmark Dashboard

Performance overview showing:

- **FEVER**: Fact verification benchmark results
- **TruthfulQA**: Question answering focused on truthfulness
- **FaithBench**: AI hallucination detection performance
- **SelfCheckGPT**: Self-consistency evaluation

Compares baseline vs coherence-guided approaches across all benchmarks.

## 🎓 Educational Examples

### Coherent Text
```
Context: Climate change news
Propositions:
1. Global temperatures have risen by 1.1°C since pre-industrial times.
2. The increase is primarily due to greenhouse gas emissions.
3. Climate scientists agree that immediate action is needed.
4. Renewable energy adoption is accelerating worldwide.
```
**Expected**: High coherence scores across all measures

### Incoherent Text  
```
Context: Mixed random facts
Propositions:
1. The sky is blue on a clear day.
2. Pizza was invented in Italy.
3. Quantum computers use quantum bits called qubits.
4. My favorite color is purple.
```
**Expected**: Very low semantic coherence, moderate entailment scores

### Contradictory Text
```
Context: Weather conditions
Propositions:
1. Today is sunny and bright.
2. It's raining heavily outside.
3. The weather is perfect for a picnic.
4. Everyone should stay indoors due to the storm.
```
**Expected**: Low entailment coherence due to contradictions

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web interface framework
- **Plotly**: Interactive visualizations  
- **Altair**: Additional charting support
- **Coherify**: Core coherence analysis library

### Architecture
```
ui/
├── coherence_app.py    # Main Streamlit application
├── README.md          # This documentation
└── __init__.py        # Module initialization

run_ui.py              # UI launcher script
demo_ui.py             # Command-line demo
```

### Adding New Measures
To add a new coherence measure to the UI:

1. Import the measure in `coherence_app.py`
2. Add it to the `create_coherence_measures()` function
3. Provide a description and color for visualization

### Customizing Examples
Edit the `get_example_texts()` function to add new preset examples with:
- Context description
- List of propositions
- Educational purpose

## 🎨 UI Design Principles

### Educational Focus
- Clear explanations of each coherence measure
- Intuitive examples that demonstrate concepts
- Progressive disclosure of complexity

### Visual Clarity
- Color-coded coherence measures
- Consistent styling and layout
- Responsive design for different screen sizes

### Interactive Exploration
- Real-time feedback on text changes
- Multiple visualization types
- Side-by-side measure comparisons

## 🐛 Troubleshooting

### Common Issues

**ImportError on launch**: Install UI dependencies
```bash
pip install streamlit plotly altair
# or
pip install -e ".[ui]"
```

**Model download errors**: Some measures require downloading transformer models on first use. This is normal and happens once.

**Slow analysis**: Large texts or complex measures may take time. The UI shows progress indicators.

### Performance Tips

- Start with shorter texts (2-5 propositions)
- Use preset examples to verify functionality
- Check system resources if analysis is very slow

## 📚 Further Reading

- [Coherify Documentation](../README.md)
- [Coherence Theory Background](../docs/README.md)
- [Benchmark Results](../docs/MULTI_FORMAT_BENCHMARKS.md)

---

Built with ❤️ using Streamlit and the Coherify library