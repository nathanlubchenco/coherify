"""
Visualization tools for coherence analysis and patterns.
Provides plotting and analysis tools to understand coherence relationships.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from coherify.core.base import (
    CoherenceMeasure,
    CoherenceResult,
    Proposition,
    PropositionSet,
)


class CoherenceVisualizer:
    """
    Main visualization class for coherence analysis.
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style for plots
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize

        # Set style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default if style not available
            pass

        # Configure seaborn
        sns.set_palette("husl")

    def plot_coherence_scores(
        self,
        results: List[CoherenceResult],
        labels: Optional[List[str]] = None,
        title: str = "Coherence Scores Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot coherence scores as a bar chart.

        Args:
            results: List of coherence results to plot
            labels: Labels for each result (default: measure names)
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        scores = [r.score for r in results]
        if labels is None:
            labels = [r.measure_name for r in results]

        # Create bar plot
        bars = ax.bar(labels, scores, alpha=0.7)

        # Color bars based on score
        for bar, score in zip(bars, scores):
            if score > 0.7:
                bar.set_color("green")
            elif score > 0.4:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        ax.set_ylabel("Coherence Score")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_component_analysis(
        self,
        hybrid_result: CoherenceResult,
        title: str = "Hybrid Coherence Component Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot component analysis for hybrid coherence results.

        Args:
            hybrid_result: Hybrid coherence result with component scores
            title: Plot title
            save_path: Path to save plot
        """
        if "component_scores" not in hybrid_result.details:
            raise ValueError("Result must contain component_scores for analysis")

        component_scores = hybrid_result.details["component_scores"]
        component_weights = hybrid_result.details.get("component_weights", {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Component scores
        components = list(component_scores.keys())
        scores = list(component_scores.values())

        bars1 = ax1.bar(components, scores, alpha=0.7)
        ax1.set_ylabel("Component Score")
        ax1.set_title("Individual Component Scores")
        ax1.set_ylim(0, 1.0)

        # Add value labels
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # Component contributions (weighted scores)
        if component_weights:
            contributions = [
                component_scores[comp] * component_weights[comp] for comp in components
            ]

            bars2 = ax2.bar(components, contributions, alpha=0.7)
            ax2.set_ylabel("Weighted Contribution")
            ax2.set_title("Component Contributions to Final Score")

            # Add value labels
            for bar, contrib in zip(bars2, contributions):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{contrib:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        proposition_texts: List[str],
        title: str = "Proposition Similarity Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot similarity matrix as a heatmap.

        Args:
            similarity_matrix: NxN similarity matrix
            proposition_texts: Text labels for propositions
            title: Plot title
            save_path: Path to save plot
        """
        # Truncate long texts for labels
        labels = [
            text[:50] + "..." if len(text) > 50 else text for text in proposition_texts
        ]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            ax=ax,
        )

        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_coherence_network(
        self,
        propositions: List[Proposition],
        similarity_matrix: np.ndarray,
        threshold: float = 0.5,
        title: str = "Coherence Network",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot coherence relationships as a network graph.

        Args:
            propositions: List of propositions
            similarity_matrix: Similarity matrix between propositions
            threshold: Minimum similarity to show edge
            title: Plot title
            save_path: Path to save plot
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for network visualization. Install with: pip install networkx"
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create graph
        G = nx.Graph()

        # Add nodes
        for i, prop in enumerate(propositions):
            # Truncate text for display
            label = prop.text[:30] + "..." if len(prop.text) > 30 else prop.text
            G.add_node(i, label=label, text=prop.text)

        # Add edges based on similarity
        n = len(propositions)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_color="lightblue", node_size=1000, alpha=0.7, ax=ax
        )

        # Draw edges with thickness based on similarity
        edges = G.edges(data=True)
        for u, v, d in edges:
            weight = d["weight"]
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=weight * 5, alpha=0.6, ax=ax
            )

        # Draw labels
        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_coherence_evolution(
        self,
        results_over_time: List[Tuple[Any, CoherenceResult]],
        x_label: str = "Time/Step",
        title: str = "Coherence Evolution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot how coherence changes over time or iterations.

        Args:
            results_over_time: List of (x_value, result) tuples
            x_label: Label for x-axis
            title: Plot title
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        x_values = [x for x, _ in results_over_time]
        scores = [result.score for _, result in results_over_time]

        ax.plot(x_values, scores, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Coherence Score")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add trend line
        z = np.polyfit(range(len(scores)), scores, 1)
        p = np.poly1d(z)
        ax.plot(x_values, p(range(len(scores))), "--", alpha=0.7, label=f"Trend")
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


class CoherenceAnalyzer:
    """
    Analysis tools for understanding coherence patterns.
    """

    def __init__(self):
        self.visualizer = CoherenceVisualizer()

    def analyze_proposition_set(
        self,
        prop_set: PropositionSet,
        coherence_measure: CoherenceMeasure,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a proposition set.

        Args:
            prop_set: Proposition set to analyze
            coherence_measure: Coherence measure to use
            detailed: Whether to include detailed analysis
        """
        # Basic coherence computation
        result = coherence_measure.compute(prop_set)

        analysis = {
            "overall_coherence": result.score,
            "measure_name": result.measure_name,
            "num_propositions": len(prop_set),
            "computation_time": result.computation_time,
        }

        if detailed and len(prop_set) > 1:
            # Pairwise analysis
            pairwise_scores = []
            propositions = prop_set.propositions

            for i in range(len(propositions)):
                for j in range(i + 1, len(propositions)):
                    pairwise_score = coherence_measure.compute_pairwise(
                        propositions[i], propositions[j]
                    )
                    pairwise_scores.append(
                        {
                            "prop1_idx": i,
                            "prop2_idx": j,
                            "prop1_text": propositions[i].text[:50],
                            "prop2_text": propositions[j].text[:50],
                            "coherence": pairwise_score,
                        }
                    )

            analysis["pairwise_analysis"] = pairwise_scores
            analysis["mean_pairwise_coherence"] = np.mean(
                [p["coherence"] for p in pairwise_scores]
            )
            analysis["std_pairwise_coherence"] = np.std(
                [p["coherence"] for p in pairwise_scores]
            )

            # Find most/least coherent pairs
            sorted_pairs = sorted(pairwise_scores, key=lambda x: x["coherence"])
            analysis["least_coherent_pair"] = sorted_pairs[0] if sorted_pairs else None
            analysis["most_coherent_pair"] = sorted_pairs[-1] if sorted_pairs else None

        return analysis

    def compare_measures(
        self,
        prop_set: PropositionSet,
        measures: List[CoherenceMeasure],
        measure_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare different coherence measures on the same proposition set.

        Args:
            prop_set: Proposition set to evaluate
            measures: List of coherence measures
            measure_names: Names for measures (optional)
        """
        if measure_names is None:
            measure_names = [m.__class__.__name__ for m in measures]

        results = []
        for measure, name in zip(measures, measure_names):
            result = measure.compute(prop_set)
            results.append(result)

        comparison = {
            "proposition_set": {
                "num_propositions": len(prop_set),
                "context": prop_set.context,
                "propositions": [p.text for p in prop_set.propositions],
            },
            "measure_results": [
                {
                    "name": name,
                    "score": result.score,
                    "computation_time": result.computation_time,
                    "details": result.details,
                }
                for name, result in zip(measure_names, results)
            ],
            "score_statistics": {
                "mean": np.mean([r.score for r in results]),
                "std": np.std([r.score for r in results]),
                "min": min([r.score for r in results]),
                "max": max([r.score for r in results]),
                "range": max([r.score for r in results])
                - min([r.score for r in results]),
            },
        }

        return comparison

    def analyze_benchmark_performance(
        self, benchmark_results: List[Dict[str, Any]], groupby: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze coherence performance across benchmark samples.

        Args:
            benchmark_results: List of evaluation results from benchmark
            groupby: Field to group results by (e.g., 'category')
        """
        scores = [r.get("coherence_score", 0) for r in benchmark_results]

        analysis = {
            "overall_statistics": {
                "mean_coherence": np.mean(scores),
                "std_coherence": np.std(scores),
                "min_coherence": np.min(scores),
                "max_coherence": np.max(scores),
                "num_samples": len(scores),
            },
            "score_distribution": {
                "high_coherence": sum(1 for s in scores if s > 0.7) / len(scores),
                "medium_coherence": sum(1 for s in scores if 0.3 <= s <= 0.7)
                / len(scores),
                "low_coherence": sum(1 for s in scores if s < 0.3) / len(scores),
            },
        }

        # Group analysis
        if groupby:
            groups = defaultdict(list)
            for result in benchmark_results:
                group = result.get(groupby, "unknown")
                groups[group].append(result.get("coherence_score", 0))

            group_analysis = {}
            for group, group_scores in groups.items():
                group_analysis[group] = {
                    "mean": np.mean(group_scores),
                    "std": np.std(group_scores),
                    "count": len(group_scores),
                }

            analysis["group_analysis"] = group_analysis

        return analysis

    def create_comprehensive_report(
        self,
        prop_set: PropositionSet,
        measures: List[CoherenceMeasure],
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a comprehensive coherence analysis report with visualizations.

        Args:
            prop_set: Proposition set to analyze
            measures: List of coherence measures to apply
            save_dir: Directory to save visualizations (optional)
        """
        # Run analysis
        comparison = self.compare_measures(prop_set, measures)

        # Create visualizations
        results = [measure.compute(prop_set) for measure in measures]
        measure_names = [m.__class__.__name__ for m in measures]

        # Plot comparison
        fig1 = self.visualizer.plot_coherence_scores(
            results, measure_names, title="Coherence Measures Comparison"
        )

        figures = {"comparison": fig1}

        # If we have hybrid results, plot component analysis
        for i, result in enumerate(results):
            if "component_scores" in result.details:
                fig = self.visualizer.plot_component_analysis(
                    result, title=f"{measure_names[i]} Component Analysis"
                )
                figures[f"components_{i}"] = fig

        # If semantic measure available, plot similarity matrix
        for i, measure in enumerate(measures):
            if hasattr(measure, "encoder") and len(prop_set) > 1:
                try:
                    # Get embeddings
                    texts = [p.text for p in prop_set.propositions]
                    embeddings = measure.encoder.encode(texts)

                    # Compute similarity matrix
                    from sklearn.metrics.pairwise import cosine_similarity

                    similarity_matrix = cosine_similarity(embeddings)

                    fig = self.visualizer.plot_similarity_matrix(
                        similarity_matrix,
                        texts,
                        title=f"Similarity Matrix ({measure_names[i]})",
                    )
                    figures[f"similarity_{i}"] = fig

                    # Network plot if networkx available
                    if NETWORKX_AVAILABLE:
                        fig = self.visualizer.plot_coherence_network(
                            prop_set.propositions,
                            similarity_matrix,
                            title=f"Coherence Network ({measure_names[i]})",
                        )
                        figures[f"network_{i}"] = fig

                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")

        # Save figures if directory provided
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")

        report = {
            "analysis": comparison,
            "figures": figures,
            "summary": {
                "best_measure": measure_names[np.argmax([r.score for r in results])],
                "score_range": max([r.score for r in results])
                - min([r.score for r in results]),
                "consistent_measures": comparison["score_statistics"]["std"] < 0.1,
            },
        }

        return report
