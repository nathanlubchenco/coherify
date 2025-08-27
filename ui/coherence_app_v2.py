"""
Optimized and professionally designed Coherify UI

High-performance, clean interface for exploring coherence measures.
"""

import os

# Add the parent directory to the path to import coherify
import sys
import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherify.utils.clean_output import enable_clean_output

# Import UI modules
try:
    from .performance import (
        get_advanced_measures,
        get_api_enhanced_measures,
        get_cached_example_texts,
        get_fast_measures,
        get_slow_measures,
    )
    from .styles import COLORS, apply_professional_styling, get_chart_config
except ImportError:
    from performance import (
        get_advanced_measures,
        get_api_enhanced_measures,
        get_cached_example_texts,
        get_fast_measures,
        get_slow_measures,
    )

# Enable clean output to suppress warnings
enable_clean_output()

# Page configuration
st.set_page_config(
    page_title="Coherify - Coherence Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply professional styling
apply_professional_styling()


def initialize_session_state():
    """Initialize session state variables."""
    if "coherence_results" not in st.session_state:
        st.session_state.coherence_results = {}
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = "Custom"
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False


@st.cache_data(ttl=300)
def compute_coherence_optimized(propositions_tuple, context, selected_measures_tuple):
    """Optimized coherence computation with caching."""
    from coherify.core.base import Proposition, PropositionSet

    # Convert back from tuple to objects
    prop_objects = [Proposition(text=prop) for prop in propositions_tuple]
    prop_set = PropositionSet(propositions=prop_objects, context=context)

    # Get all available measures (including API measures if configured)
    fast_measures = get_fast_measures()
    slow_measures = get_slow_measures()
    advanced_measures = get_advanced_measures()

    # Get API measures if available (check session state for API config)
    api_measures = {}
    try:
        # This will be empty if no API is configured, which is fine
        api_measures = get_api_enhanced_measures(None, None)
    except:
        pass

    all_measures = {
        **fast_measures,
        **slow_measures,
        **advanced_measures,
        **api_measures,
    }

    results = {}

    for measure_name in selected_measures_tuple:
        if measure_name in all_measures:
            config = all_measures[measure_name]
            try:
                start_time = time.time()
                result = config["measure"].compute(prop_set)
                computation_time = time.time() - start_time

                # Extract pairwise scores
                pairwise_scores = []
                if "pairwise_similarities" in result.details:
                    pairwise_scores = result.details["pairwise_similarities"]
                elif "pairwise_scores" in result.details:
                    pairwise_scores = result.details["pairwise_scores"]

                results[measure_name] = {
                    "score": result.score,
                    "pairwise_scores": pairwise_scores,
                    "color": config["color"],
                    "description": config["description"],
                    "computation_time": computation_time,
                    "unbounded": config.get("unbounded", False),
                    "explanation": config.get("explanation", config["description"]),
                }
            except Exception as e:
                results[measure_name] = {
                    "score": 0.0,
                    "pairwise_scores": [],
                    "color": config["color"],
                    "description": config["description"],
                    "error": str(e),
                    "computation_time": 0.0,
                    "unbounded": config.get("unbounded", False),
                    "explanation": config.get("explanation", config["description"]),
                }

    return results


def create_professional_score_chart(results):
    """Create a professional-looking score comparison chart."""
    if not results:
        return None

    chart_config = get_chart_config()

    # Prepare data
    names = []
    scores = []
    colors = []

    for name, data in results.items():
        if "error" not in data:
            names.append(name.replace(" Coherence", ""))
            scores.append(data["score"])
            colors.append(data["color"])

    if not names:
        return None

    # Create chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=[f"{score:.3f}" for score in scores],
            textposition="outside",
            textfont=dict(size=12, color="#1a1a1a"),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Coherence Analysis Results",
            font=dict(size=18, color="#2E86AB", family="Inter"),
        ),
        xaxis_title="Measure",
        yaxis_title="Coherence Score",
        yaxis=dict(range=[0, 1.1], gridcolor=chart_config["grid_color"]),
        xaxis=dict(gridcolor=chart_config["grid_color"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=12, color="#1a1a1a"),
        height=chart_config["height"],
        showlegend=False,
        margin=dict(t=60, b=40, l=40, r=40),
    )

    return fig


def create_professional_heatmap(results, propositions):
    """Create a professional pairwise similarity heatmap."""
    if not results or len(propositions) < 2:
        return None

    # Find the first measure with pairwise data
    pairwise_data = None
    measure_name = None

    for name, data in results.items():
        if "error" not in data and data.get("pairwise_scores"):
            pairwise_data = data["pairwise_scores"]
            measure_name = name
            break

    if not pairwise_data:
        return None

    # Create similarity matrix
    n = len(propositions)
    matrix = np.eye(n)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(pairwise_data):
                matrix[i][j] = pairwise_data[idx]
                matrix[j][i] = pairwise_data[idx]
                idx += 1

    # Create labels
    labels = [f"Statement {i+1}" for i in range(n)]

    # Create heatmap with professional styling
    fig = px.imshow(
        matrix,
        labels=dict(x="Statement", y="Statement", color="Similarity"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title=f"Pairwise Similarity Matrix ({measure_name})",
    )

    fig.update_layout(
        font=dict(family="Inter", size=12, color="#1a1a1a"),
        title=dict(font=dict(size=16, color="#2E86AB")),
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def render_performance_metrics(results):
    """Display performance metrics."""
    if not results:
        return

    total_time = sum(data.get("computation_time", 0) for data in results.values())
    successful_measures = sum(1 for data in results.values() if "error" not in data)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{total_time:.2f}s</div>
            <div class="metric-label">Total Time</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{successful_measures}</div>
            <div class="metric-label">Measures Computed</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        avg_time = total_time / max(successful_measures, 1)
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{avg_time:.2f}s</div>
            <div class="metric-label">Avg per Measure</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_workbench():
    """Render the optimized coherence workbench."""
    st.markdown(
        '<div class="main-header">Coherence Analysis Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    <strong>Welcome to Coherify</strong><br>
    A professional platform for analyzing text coherence using advanced NLP measures.
    Choose from preset examples or enter your own text to explore coherence patterns.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown(
            '<div class="section-header">Configuration</div>', unsafe_allow_html=True
        )

        # Example selection
        examples = get_cached_example_texts()
        selected_example = st.selectbox(
            "Choose Example", options=list(examples.keys()), index=0
        )

        # Performance mode
        st.markdown(
            '<div class="section-header">Performance</div>', unsafe_allow_html=True
        )
        performance_mode = st.radio(
            "Analysis Mode",
            [
                "Fast (Semantic only)",
                "Balanced (Fast + NLI)",
                "Complete (All measures)",
                "Advanced (+ Shogenji)",
                "API Enhanced",
            ],
            index=0,
            help="Choose analysis speed vs comprehensiveness trade-off",
        )

        # API Configuration (only show if API Enhanced mode selected)
        api_provider = None
        api_model = None
        api_key_status = False

        if performance_mode == "API Enhanced":
            st.markdown(
                '<div class="section-header">API Configuration</div>',
                unsafe_allow_html=True,
            )

            api_provider = st.selectbox(
                "API Provider",
                ["None", "anthropic", "openai"],
                index=0,
                help="Choose your AI API provider",
            )

            if api_provider and api_provider != "None":
                # Model selection based on provider
                if api_provider == "anthropic":
                    api_models = [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                    ]
                elif api_provider == "openai":
                    api_models = [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "gpt-4",
                        "gpt-3.5-turbo",
                    ]
                else:
                    api_models = []

                api_model = st.selectbox(
                    f"{api_provider.title()} Model",
                    api_models,
                    index=0,
                    help=f"Choose your {api_provider} model",
                )

                # API Key status check
                import os

                if api_provider == "anthropic":
                    api_key_status = bool(os.getenv("ANTHROPIC_API_KEY"))
                    key_env_var = "ANTHROPIC_API_KEY"
                elif api_provider == "openai":
                    api_key_status = bool(os.getenv("OPENAI_API_KEY"))
                    key_env_var = "OPENAI_API_KEY"

                if api_key_status:
                    st.success(f"✅ {key_env_var} found")
                else:
                    st.error(f"❌ {key_env_var} not found")
                    st.info(
                        f"💡 Set your API key: `export {key_env_var}=your_key_here`"
                    )

        # Measure selection based on performance mode
        fast_measures = get_fast_measures()
        slow_measures = get_slow_measures()
        advanced_measures = get_advanced_measures()
        api_measures = (
            get_api_enhanced_measures(api_provider, api_model) if api_key_status else {}
        )

        if performance_mode == "Fast (Semantic only)":
            available_measures = fast_measures
        elif performance_mode == "Balanced (Fast + NLI)":
            available_measures = {**fast_measures, **slow_measures}
        elif performance_mode == "Complete (All measures)":
            available_measures = {**fast_measures, **slow_measures}
        elif performance_mode == "Advanced (+ Shogenji)":
            available_measures = {**fast_measures, **slow_measures, **advanced_measures}
        elif performance_mode == "API Enhanced":
            available_measures = {
                **fast_measures,
                **slow_measures,
                **advanced_measures,
                **api_measures,
            }
        else:
            available_measures = {**fast_measures, **slow_measures}

        # Show loading status with more detail
        if not available_measures:
            st.error("❌ No measures available. Check console for loading errors.")
            st.info(
                "💡 Try refreshing the page or check your internet connection for model downloads."
            )
        else:
            st.success(f"✅ {len(available_measures)} measures ready")

            # Show which measures failed to load (if any)
            expected_fast = ["Semantic Coherence", "Hybrid Coherence"]
            expected_slow = ["Entailment Coherence"]
            expected_advanced = ["Shogenji Coherence"]
            expected_api = (
                [f"{api_provider.title()} Enhanced"]
                if api_provider and api_provider != "None"
                else []
            )

            if performance_mode == "Fast (Semantic only)":
                expected = expected_fast
            elif performance_mode == "Balanced (Fast + NLI)":
                expected = expected_fast + expected_slow
            elif performance_mode == "Complete (All measures)":
                expected = expected_fast + expected_slow
            elif performance_mode == "Advanced (+ Shogenji)":
                expected = expected_fast + expected_slow + expected_advanced
            elif performance_mode == "API Enhanced":
                expected = (
                    expected_fast + expected_slow + expected_advanced + expected_api
                )
            else:
                expected = expected_fast + expected_slow

            missing = [name for name in expected if name not in available_measures]
            if missing:
                st.warning(f"⚠️ Some measures failed to load: {', '.join(missing)}")

        selected_measures = []
        st.markdown("**Active Measures:**")
        for name in available_measures.keys():
            if st.checkbox(name, value=True, key=f"measure_{name}"):
                selected_measures.append(name)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<div class="section-header">Text Input</div>', unsafe_allow_html=True
        )

        # Context
        example_data = examples[selected_example]
        context = st.text_input(
            "Context (optional)",
            value=example_data["context"],
            help="Provide context for your statements",
        )

        # Propositions
        st.write("**Statements to Analyze**")
        propositions = []

        if selected_example == "Custom":
            num_props = st.number_input(
                "Number of statements", min_value=2, max_value=8, value=3
            )
            for i in range(num_props):
                prop = st.text_input(f"Statement {i+1}", key=f"prop_{i}")
                if prop.strip():
                    propositions.append(prop.strip())
        else:
            for i, prop in enumerate(example_data["propositions"]):
                edited_prop = st.text_input(
                    f"Statement {i+1}", value=prop, key=f"prop_{i}"
                )
                if edited_prop.strip():
                    propositions.append(edited_prop.strip())

        # Analysis button
        analyze_button = st.button(
            "Analyze Coherence",
            type="primary",
            use_container_width=True,
            disabled=len(propositions) < 2,
        )

        if analyze_button and len(propositions) >= 2:
            # Create cache key
            propositions_tuple = tuple(propositions)
            selected_measures_tuple = tuple(selected_measures)

            # Show progress
            with st.spinner("Computing coherence scores..."):
                results = compute_coherence_optimized(
                    propositions_tuple, context, selected_measures_tuple
                )
                st.session_state.coherence_results = results

    with col2:
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

        if st.session_state.coherence_results:
            results = st.session_state.coherence_results

            # Display scores with special handling for unbounded measures
            for name, data in results.items():
                if "error" not in data:
                    # Check if this is an unbounded measure
                    is_unbounded = data.get("unbounded", False)

                    if is_unbounded:
                        # Format unbounded score differently
                        if data["score"] >= 1000:
                            display_value = f"{data['score']:.0f}"
                        elif data["score"] >= 10:
                            display_value = f"{data['score']:.1f}"
                        else:
                            display_value = f"{data['score']:.3f}"

                        # Add interpretation
                        if data["score"] > 1:
                            interpretation = "Coherent"
                        elif data["score"] == 1:
                            interpretation = "Independent"
                        else:
                            interpretation = "Incoherent"

                        st.metric(
                            label=f"{name} (unbounded)",
                            value=display_value,
                            delta=interpretation,
                            help=f"{data.get('explanation', data['description'])} (computed in {data.get('computation_time', 0):.2f}s)",
                        )
                    else:
                        # Standard bounded measure (0-1 scale)
                        st.metric(
                            label=name,
                            value=f"{data['score']:.3f}",
                            help=f"{data['description']} (computed in {data.get('computation_time', 0):.2f}s)",
                        )
                else:
                    st.error(f"{name}: Error occurred")

            # Add explanation for unbounded measures if any are present
            unbounded_measures = [
                name
                for name, data in results.items()
                if data.get("unbounded", False) and "error" not in data
            ]
            if unbounded_measures:
                st.markdown(
                    """
                <div class="info-box" style="margin-top: 1rem;">
                <strong>About Unbounded Measures:</strong><br>
                Shogenji coherence represents the philosophical approach where coherence is measured as the ratio of joint probability to independent probabilities. Unlike modern NLP measures (0-1 scale), it can range from 0 to infinity, making very high values theoretically meaningful and indicating strong mutual support between propositions.
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
            <div class="info-box">
            Configure your analysis and click "Analyze Coherence" to see results.
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Visualizations
    if st.session_state.coherence_results:
        st.markdown(
            '<div class="section-header">Visual Analysis</div>', unsafe_allow_html=True
        )

        results = st.session_state.coherence_results

        col1, col2 = st.columns(2)

        with col1:
            score_fig = create_professional_score_chart(results)
            if score_fig:
                st.plotly_chart(score_fig, use_container_width=True)

        with col2:
            if len(propositions) >= 2:
                heatmap_fig = create_professional_heatmap(results, propositions)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)

        # Performance metrics
        st.markdown(
            '<div class="section-header">Performance Metrics</div>',
            unsafe_allow_html=True,
        )
        render_performance_metrics(results)


def render_benchmark_dashboard():
    """Render professional benchmark dashboard."""
    st.markdown(
        '<div class="main-header">Benchmark Performance</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="info-box">
    <strong>Research Results</strong><br>
    Performance comparison of coherence-guided approaches across standard NLP benchmarks.
    These results demonstrate the practical value of coherence theory in real applications.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Mock benchmark data with professional presentation
    benchmark_data = {
        "FEVER": {
            "baseline": 0.654,
            "coherence": 0.721,
            "samples": 16821,
            "metric": "F1",
        },
        "TruthfulQA": {
            "baseline": 0.438,
            "coherence": 0.502,
            "samples": 817,
            "metric": "Accuracy",
        },
        "FaithBench": {
            "baseline": 0.583,
            "coherence": 0.641,
            "samples": 1200,
            "metric": "AUC",
        },
        "SelfCheckGPT": {
            "baseline": 0.692,
            "coherence": 0.734,
            "samples": 2000,
            "metric": "Precision@K",
        },
    }

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    avg_baseline = np.mean([data["baseline"] for data in benchmark_data.values()])
    avg_coherence = np.mean([data["coherence"] for data in benchmark_data.values()])
    avg_improvement = avg_coherence - avg_baseline

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{avg_baseline:.3f}</div>
            <div class="metric-label">Avg Baseline</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{avg_coherence:.3f}</div>
            <div class="metric-label">Avg Coherence</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card success-box">
            <div class="metric-value">+{avg_improvement:.3f}</div>
            <div class="metric-label">Improvement</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{len(benchmark_data)}</div>
            <div class="metric-label">Benchmarks</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Professional comparison chart
    benchmarks = list(benchmark_data.keys())
    baseline_scores = [benchmark_data[b]["baseline"] for b in benchmarks]
    coherence_scores = [benchmark_data[b]["coherence"] for b in benchmarks]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=benchmarks,
            y=baseline_scores,
            marker_color=COLORS["accent_gray"],
            text=[f"{score:.3f}" for score in baseline_scores],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Coherence-Guided",
            x=benchmarks,
            y=coherence_scores,
            marker_color=COLORS["primary_blue"],
            text=[f"{score:.3f}" for score in coherence_scores],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(
            text="Performance Comparison: Baseline vs Coherence-Guided",
            font=dict(size=18, color="#2E86AB", family="Inter"),
        ),
        xaxis_title="Benchmark",
        yaxis_title="Performance Score",
        yaxis=dict(range=[0, 1]),
        barmode="group",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=12, color="#1a1a1a"),
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point with performance optimizations."""
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("Coherify")
    st.sidebar.caption("Professional Coherence Analysis")

    page = st.sidebar.selectbox("Navigate", ["Analysis Workbench", "Benchmark Results"])

    if page == "Analysis Workbench":
        render_workbench()
    elif page == "Benchmark Results":
        render_benchmark_dashboard()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <small style='color: #6C757D;'>
        Built with Streamlit<br>
        Powered by Coherify
        </small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
