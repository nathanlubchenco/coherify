"""
Interactive Coherence Analysis UI

A Streamlit-based web application for exploring coherence measures
and understanding how they work on different text examples.
"""

import os

# Add the parent directory to the path to import coherify
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherify.core.base import Proposition, PropositionSet
from coherify.measures.entailment import EntailmentCoherence
from coherify.measures.hybrid import HybridCoherence
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.shogenji import ShogunjiCoherence
from coherify.utils.clean_output import enable_clean_output

# Import UI configuration
try:
    from .config import (
        BENCHMARK_DATA,
        DEFAULT_EXAMPLES,
        MEASURE_COLORS,
        MEASURE_EXPLANATIONS,
        UI_CONFIG,
        UI_CONTENT,
    )
except ImportError:
    # Handle direct script execution
    from config import (
        BENCHMARK_DATA,
        DEFAULT_EXAMPLES,
        MEASURE_COLORS,
        MEASURE_EXPLANATIONS,
        UI_CONFIG,
        UI_CONTENT,
    )

# Enable clean output to suppress warnings
enable_clean_output()

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["title"],
    page_icon=UI_CONFIG["icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["sidebar_state"],
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables."""
    if "coherence_results" not in st.session_state:
        st.session_state.coherence_results = {}
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = "Custom"


def get_example_texts():
    """Get predefined example texts for testing."""
    examples = {
        "Custom": {
            "context": "",
            "propositions": [],
            "description": "Enter your own text",
        }
    }
    examples.update(DEFAULT_EXAMPLES)
    return examples


def create_coherence_measures():
    """Create and return available coherence measures."""
    measures = {
        "Semantic Coherence": {
            "measure": SemanticCoherence(),
            "description": MEASURE_EXPLANATIONS["Semantic Coherence"]["short"],
            "color": MEASURE_COLORS["Semantic Coherence"],
        },
        "Entailment Coherence": {
            "measure": EntailmentCoherence(),
            "description": MEASURE_EXPLANATIONS["Entailment Coherence"]["short"],
            "color": MEASURE_COLORS["Entailment Coherence"],
        },
        "Hybrid Coherence": {
            "measure": HybridCoherence(),
            "description": MEASURE_EXPLANATIONS["Hybrid Coherence"]["short"],
            "color": MEASURE_COLORS["Hybrid Coherence"],
        },
    }

    # Add Shogenji if available (might fail due to model requirements)
    try:
        # Create a wrapper class for normalized Shogenji coherence
        class NormalizedShogunjiCoherence:
            def __init__(self):
                self.base_measure = ShogunjiCoherence()

            def compute(self, prop_set):
                result = self.base_measure.compute(prop_set)
                # Normalize the score to 0-1 using tanh transformation
                # This keeps the relative ordering but makes scores more interpretable
                import math

                normalized_score = math.tanh(math.log(max(result.score, 1e-10)) / 10)
                normalized_score = max(0, normalized_score)  # Ensure non-negative

                # Create new result with normalized score
                from coherify.core.base import CoherenceResult

                return CoherenceResult(
                    score=normalized_score,
                    measure_name="NormalizedShogunjiCoherence",
                    details={
                        **result.details,
                        "original_shogenji_score": result.score,
                        "normalization": "tanh(log(score)/10)",
                    },
                    computation_time=result.computation_time,
                )

        measures["Shogenji Coherence"] = {
            "measure": NormalizedShogunjiCoherence(),
            "description": "Traditional probability-based coherence measure (normalized for intuitive 0-1 scale).",
            "color": MEASURE_COLORS["Shogenji Coherence"],
        }
    except Exception:
        pass  # Skip if model not available

    return measures


def compute_coherence_scores(prop_set, measures):
    """Compute coherence scores for all measures."""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, config) in enumerate(measures.items()):
        status_text.text(f"Computing {name}...")
        try:
            result = config["measure"].compute(prop_set)
            # Extract pairwise scores from different measures
            pairwise_scores = []
            if "pairwise_similarities" in result.details:
                pairwise_scores = result.details["pairwise_similarities"]
            elif "pairwise_scores" in result.details:
                pairwise_scores = result.details["pairwise_scores"]
            elif "pairwise_entailments" in result.details:
                pairwise_scores = result.details["pairwise_entailments"]

            # Add additional info for normalized measures
            description = config["description"]
            if "original_shogenji_score" in result.details:
                original_score = result.details["original_shogenji_score"]
                description += f" (Original: {original_score:.1f})"

            results[name] = {
                "score": result.score,
                "pairwise_scores": pairwise_scores,
                "color": config["color"],
                "description": description,
            }
        except Exception as e:
            st.warning(f"Could not compute {name}: {str(e)}")
            results[name] = {
                "score": 0.0,
                "pairwise_scores": [],
                "color": config["color"],
                "description": config["description"],
                "error": str(e),
            }

        progress_bar.progress((i + 1) / len(measures))

    status_text.empty()
    progress_bar.empty()
    return results


def create_score_visualization(results):
    """Create a bar chart of coherence scores."""
    if not results:
        return None

    names = list(results.keys())
    scores = [results[name]["score"] for name in names]
    colors = [results[name]["color"] for name in names]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=scores,
                marker_color=colors,
                text=[f"{score:.3f}" for score in scores],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Coherence Scores Comparison",
        xaxis_title="Coherence Measure",
        yaxis_title="Coherence Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False,
    )

    return fig


def create_pairwise_heatmap(results, propositions):
    """Create a heatmap of pairwise coherence scores."""
    if not results or len(propositions) < 2:
        return None

    # Use the first available measure with pairwise scores
    measure_name = None
    pairwise_scores = None

    for name, data in results.items():
        if (
            "error" not in data
            and data.get("pairwise_scores")
            and len(data["pairwise_scores"]) > 0
        ):
            measure_name = name
            pairwise_scores = data["pairwise_scores"]
            break

    if not pairwise_scores or not measure_name:
        return None

    # Create pairwise matrix
    n = len(propositions)
    matrix = np.eye(n)  # Identity matrix (1.0 on diagonal)

    # Fill upper triangle with pairwise scores
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(pairwise_scores):
                matrix[i][j] = pairwise_scores[idx]
                matrix[j][i] = pairwise_scores[idx]  # Symmetric
                idx += 1

    # Create shortened labels for propositions
    labels = [
        f"P{i+1}: {prop[:30]}..." if len(prop) > 30 else f"P{i+1}: {prop}"
        for i, prop in enumerate(propositions)
    ]

    fig = px.imshow(
        matrix,
        labels=dict(x="Proposition", y="Proposition", color="Similarity"),
        x=labels,
        y=labels,
        color_continuous_scale="RdYlBu_r",
        title=f"Pairwise Similarity Matrix ({measure_name})",
    )

    fig.update_layout(height=500)
    return fig


def render_workbench():
    """Render the main coherence workbench."""
    st.markdown(
        '<div class="main-header">🧠 Coherence Analysis Workbench</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="explanation-box">
    {UI_CONTENT["welcome_message"]}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")

        # Example selection
        examples = get_example_texts()
        selected_example = st.selectbox(
            "Choose an example:",
            options=list(examples.keys()),
            index=0,
            key="example_selector",
        )

        if selected_example != st.session_state.selected_example:
            st.session_state.selected_example = selected_example
            st.rerun()

        st.markdown("---")

        # Measure selection
        st.subheader("📊 Coherence Measures")
        measures = create_coherence_measures()

        selected_measures = {}
        for name, config in measures.items():
            if st.checkbox(name, value=True, key=f"measure_{name}"):
                selected_measures[name] = config

        # Information about measures
        st.markdown("---")
        st.subheader("ℹ️ About Measures")
        for name, config in selected_measures.items():
            with st.expander(f"{name}"):
                st.write(config["description"])

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<div class="section-header">📝 Text Input</div>', unsafe_allow_html=True
        )

        # Context input
        example_data = examples[selected_example]
        context = st.text_input(
            "Context (optional):",
            value=example_data["context"],
            help="Provide context for your propositions",
        )

        # Propositions input
        st.write("**Propositions to analyze:**")
        propositions = []

        if selected_example == "Custom":
            # Allow manual input for custom example
            num_props = st.number_input(
                "Number of propositions:", min_value=1, max_value=10, value=3
            )
            for i in range(num_props):
                prop = st.text_input(f"Proposition {i+1}:", key=f"prop_{i}")
                if prop.strip():
                    propositions.append(prop.strip())
        else:
            # Show predefined propositions for examples
            for i, prop in enumerate(example_data["propositions"]):
                edited_prop = st.text_input(
                    f"Proposition {i+1}:", value=prop, key=f"prop_{i}"
                )
                if edited_prop.strip():
                    propositions.append(edited_prop.strip())

        # Analysis button
        if st.button("🔍 Analyze Coherence", type="primary", use_container_width=True):
            if len(propositions) < 2:
                st.error("Please provide at least 2 propositions for analysis.")
            else:
                # Create proposition set
                prop_objects = [Proposition(text=prop) for prop in propositions]
                prop_set = PropositionSet(propositions=prop_objects, context=context)

                # Compute coherence scores
                with st.spinner("Computing coherence scores..."):
                    results = compute_coherence_scores(prop_set, selected_measures)
                    st.session_state.coherence_results = results

    with col2:
        st.markdown(
            '<div class="section-header">📊 Results</div>', unsafe_allow_html=True
        )

        if st.session_state.coherence_results:
            results = st.session_state.coherence_results

            # Display scores
            for name, data in results.items():
                if "error" not in data:
                    st.metric(
                        label=name,
                        value=f"{data['score']:.3f}",
                        help=data["description"],
                    )
                else:
                    st.error(f"{name}: {data['error']}")
        else:
            st.info("Click 'Analyze Coherence' to see results here.")

    # Visualizations
    if st.session_state.coherence_results:
        st.markdown(
            '<div class="section-header">📈 Visualizations</div>',
            unsafe_allow_html=True,
        )

        results = st.session_state.coherence_results

        # Score comparison chart
        col1, col2 = st.columns(2)

        with col1:
            score_fig = create_score_visualization(results)
            if score_fig:
                st.plotly_chart(score_fig, use_container_width=True)

        with col2:
            # Pairwise similarity heatmap
            if len(propositions) >= 2:
                heatmap_fig = create_pairwise_heatmap(results, propositions)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    st.info(
                        "Pairwise similarity data not available for selected measures."
                    )


def render_benchmark_dashboard():
    """Render the benchmark performance dashboard."""
    st.markdown(
        '<div class="main-header">📊 Benchmark Performance Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="explanation-box">
    {UI_CONTENT["benchmark_description"]}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load benchmark data from configuration
    benchmark_data = BENCHMARK_DATA

    # Performance overview
    col1, col2, col3, col4 = st.columns(4)

    avg_baseline = np.mean([data["baseline"] for data in benchmark_data.values()])
    avg_coherence = np.mean(
        [data["coherence_guided"] for data in benchmark_data.values()]
    )
    avg_improvement = avg_coherence - avg_baseline
    total_benchmarks = len(benchmark_data)

    with col1:
        st.metric("Average Baseline", f"{avg_baseline:.3f}")
    with col2:
        st.metric("Average Coherence-Guided", f"{avg_coherence:.3f}")
    with col3:
        st.metric("Average Improvement", f"+{avg_improvement:.3f}")
    with col4:
        st.metric("Benchmarks Tested", total_benchmarks)

    # Detailed comparison
    st.markdown(
        '<div class="section-header">🏆 Detailed Performance Comparison</div>',
        unsafe_allow_html=True,
    )

    # Create comparison chart
    benchmarks = list(benchmark_data.keys())
    baseline_scores = [benchmark_data[b]["baseline"] for b in benchmarks]
    coherence_scores = [benchmark_data[b]["coherence_guided"] for b in benchmarks]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=benchmarks,
            y=baseline_scores,
            marker_color="lightcoral",
            text=[f"{score:.3f}" for score in baseline_scores],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Coherence-Guided",
            x=benchmarks,
            y=coherence_scores,
            marker_color="lightblue",
            text=[f"{score:.3f}" for score in coherence_scores],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Benchmark Performance: Baseline vs Coherence-Guided",
        xaxis_title="Benchmark",
        yaxis_title="Performance Score",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Improvement analysis
    st.markdown(
        '<div class="section-header">📈 Improvement Analysis</div>',
        unsafe_allow_html=True,
    )

    improvements = [benchmark_data[b]["improvement"] for b in benchmarks]

    fig_improvement = go.Figure()
    fig_improvement.add_trace(
        go.Bar(
            x=benchmarks,
            y=improvements,
            marker_color="green",
            text=[f"+{imp:.3f}" for imp in improvements],
            textposition="auto",
        )
    )

    fig_improvement.update_layout(
        title="Performance Improvement with Coherence Guidance",
        xaxis_title="Benchmark",
        yaxis_title="Improvement (Coherence - Baseline)",
        height=400,
    )

    st.plotly_chart(fig_improvement, use_container_width=True)

    # Benchmark details
    st.markdown(
        '<div class="section-header">📋 Benchmark Details</div>', unsafe_allow_html=True
    )

    for benchmark, data in benchmark_data.items():
        with st.expander(f"{benchmark} - {data['description']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baseline Score", f"{data['baseline']:.3f}")
            with col2:
                st.metric("Coherence Score", f"{data['coherence_guided']:.3f}")
            with col3:
                st.metric("Improvement", f"+{data['improvement']:.3f}")

            st.write(f"**Description:** {data['description']}")


def main():
    """Main application entry point."""
    initialize_session_state()

    # Navigation
    st.sidebar.title("🧠 Coherify")
    st.sidebar.markdown("Interactive Coherence Analysis")

    page = st.sidebar.selectbox(
        "Navigate to:", ["🔬 Coherence Workbench", "📊 Benchmark Dashboard"]
    )

    if page == "🔬 Coherence Workbench":
        render_workbench()
    elif page == "📊 Benchmark Dashboard":
        render_benchmark_dashboard()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <small>
        Built with Streamlit<br>
        Powered by Coherify
        </small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
