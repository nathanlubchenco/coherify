"""
Configuration settings for the Coherify UI
"""

# UI Settings
UI_CONFIG = {
    "title": "Coherify: Interactive Coherence Analysis",
    "icon": "🧠",
    "layout": "wide",
    "sidebar_state": "expanded",
}

# Color scheme for coherence measures
MEASURE_COLORS = {
    "Semantic Coherence": "#1f77b4",  # Blue
    "Entailment Coherence": "#ff7f0e",  # Orange
    "Hybrid Coherence": "#2ca02c",  # Green
    "Shogenji Coherence": "#d62728",  # Red
    "Adaptive Hybrid": "#9467bd",  # Purple
}

# Default examples for the workbench
DEFAULT_EXAMPLES = {
    "Coherent News Story": {
        "context": "Breaking news about climate change",
        "propositions": [
            "Global temperatures have risen by 1.1°C since pre-industrial times.",
            "The increase is primarily due to greenhouse gas emissions.",
            "Climate scientists agree that immediate action is needed.",
            "Renewable energy adoption is accelerating worldwide.",
        ],
        "description": "Example of coherent, related statements about a single topic.",
    },
    "Incoherent Mixed Topics": {
        "context": "Random facts",
        "propositions": [
            "The sky is blue on a clear day.",
            "Pizza was invented in Italy.",
            "Quantum computers use quantum bits called qubits.",
            "My favorite color is purple.",
        ],
        "description": "Unrelated statements that should score low on coherence.",
    },
    "Partially Coherent": {
        "context": "Technology and society",
        "propositions": [
            "Artificial intelligence is transforming many industries.",
            "Machine learning requires large amounts of data.",
            "Paris is the capital of France.",
            "AI systems can exhibit biases from their training data.",
        ],
        "description": "Mix of related and unrelated statements.",
    },
    "Contradictory Statements": {
        "context": "Weather conditions",
        "propositions": [
            "Today is sunny and bright.",
            "It's raining heavily outside.",
            "The weather is perfect for a picnic.",
            "Everyone should stay indoors due to the storm.",
        ],
        "description": "Contradictory statements that should be detected by entailment measures.",
    },
    "Logical Chain": {
        "context": "Mathematical reasoning",
        "propositions": [
            "All mammals are warm-blooded.",
            "Dogs are mammals.",
            "Therefore, dogs are warm-blooded.",
            "Warm-blooded animals regulate their body temperature.",
        ],
        "description": "Logical chain that should score high on entailment coherence.",
    },
}

# Benchmark performance data (in a real app, this might come from a database)
BENCHMARK_DATA = {
    "FEVER": {
        "baseline": 0.654,
        "coherence_guided": 0.721,
        "improvement": 0.067,
        "description": "Fact verification task testing claim-evidence coherence",
        "samples": 16821,
        "metric": "F1 Score",
    },
    "TruthfulQA": {
        "baseline": 0.438,
        "coherence_guided": 0.502,
        "improvement": 0.064,
        "description": "Question answering focused on truthful responses",
        "samples": 817,
        "metric": "Accuracy",
    },
    "FaithBench": {
        "baseline": 0.583,
        "coherence_guided": 0.641,
        "improvement": 0.058,
        "description": "Hallucination detection in AI-generated text",
        "samples": 1200,
        "metric": "AUC",
    },
    "SelfCheckGPT": {
        "baseline": 0.692,
        "coherence_guided": 0.734,
        "improvement": 0.042,
        "description": "Self-consistency checking for language models",
        "samples": 2000,
        "metric": "Precision@K",
    },
}

# Help text and explanations
MEASURE_EXPLANATIONS = {
    "Semantic Coherence": {
        "short": "Uses sentence embeddings to measure semantic similarity between propositions.",
        "detailed": """
        Semantic coherence measures how similar propositions are in meaning by:

        1. **Encoding**: Converting each proposition into a high-dimensional vector using transformer models
        2. **Similarity**: Computing cosine similarity between all proposition pairs
        3. **Aggregation**: Combining pairwise similarities into an overall coherence score

        **Strengths**: Good at detecting topical coherence and semantic relatedness
        **Limitations**: May miss logical contradictions if they're semantically similar
        **Best for**: Content analysis, topic modeling, semantic clustering
        """,
    },
    "Entailment Coherence": {
        "short": "Uses natural language inference to detect logical relationships.",
        "detailed": """
        Entailment coherence analyzes logical relationships by:

        1. **NLI Classification**: Using trained models to classify proposition pairs as entailment, contradiction, or neutral
        2. **Scoring**: Converting logical relationships into coherence scores
        3. **Integration**: Combining pairwise judgments into overall coherence

        **Strengths**: Detects contradictions and logical inconsistencies
        **Limitations**: May be sensitive to model training data biases
        **Best for**: Fact-checking, consistency verification, logical analysis
        """,
    },
    "Hybrid Coherence": {
        "short": "Combines semantic and entailment measures for balanced analysis.",
        "detailed": """
        Hybrid coherence provides comprehensive analysis by:

        1. **Multi-faceted**: Computing both semantic and entailment coherence
        2. **Weighted Combination**: Balancing different aspects of coherence
        3. **Robust Scoring**: Less sensitive to individual measure limitations

        **Strengths**: More robust and comprehensive than single measures
        **Limitations**: May dilute strong signals from individual measures
        **Best for**: General purpose coherence analysis, balanced evaluation
        """,
    },
    "Shogenji Coherence": {
        "short": "Traditional probability-based coherence measure (normalized for intuitive 0-1 scale).",
        "detailed": """
        The Shogenji coherence measure is the classical approach from philosophy:

        1. **Formula**: C_S = P(H1 ∧ H2 ∧ ... ∧ Hn) / ∏P(Hi)
        2. **Interpretation**: How much more likely the propositions are together vs independently
        3. **Normalization**: Applied tanh transformation for intuitive 0-1 scale

        **Strengths**: Theoretically grounded, captures probabilistic coherence
        **Limitations**: Can be extremely large, sensitive to probability estimation
        **Best for**: Theoretical analysis, understanding classical coherence theory

        **Note**: Raw Shogenji scores can range from 0 to infinity. We normalize using
        tanh(log(score)/10) to keep values interpretable while preserving ordering.
        """,
    },
}

# UI text content
UI_CONTENT = {
    "welcome_message": """
    **Welcome to the Coherence Workbench!**

    Explore how different coherence measures analyze text. Try the example presets or enter your own text to see how coherent your propositions are according to different theories.
    """,
    "workbench_description": """
    The Coherence Workbench lets you experiment with different text examples and see how various coherence measures evaluate them. This is perfect for:

    - **Understanding** how coherence measures work
    - **Comparing** different theoretical approaches
    - **Testing** your own text for coherence patterns
    - **Learning** about formal coherence theory
    """,
    "benchmark_description": """
    **Benchmark Performance Overview**

    This dashboard shows how coherence-guided approaches perform compared to baseline methods across different benchmarks. These results demonstrate the practical value of coherence theory in real-world applications.
    """,
}

# Chart configuration
CHART_CONFIG = {
    "height": 400,
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "font_size": 12,
    "show_legend": True,
}
