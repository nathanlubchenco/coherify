"""
Clustering-based approximation algorithms for large proposition sets.
Groups similar propositions and computes coherence on cluster representatives.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from coherify.core.base import (
    CoherenceMeasure,
    PropositionSet,
    Proposition,
)
from coherify.measures.semantic import SemanticCoherence


@dataclass
class ClusteringResult:
    """Result of clustering-based coherence approximation."""

    approximate_score: float
    num_clusters: int
    total_propositions: int
    clustering_time: float
    computation_time: float
    cluster_coherences: List[float]
    cluster_sizes: List[int]
    clustering_method: str = "unknown"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PropositionClusterer:
    """Base clustering functionality for propositions."""

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        max_clusters: Optional[int] = None,
    ):
        """
        Initialize clusterer.

        Args:
            coherence_measure: Measure for computing embeddings/similarities
            max_clusters: Maximum number of clusters
        """
        self.coherence_measure = coherence_measure or SemanticCoherence()
        self.max_clusters = max_clusters

    def _get_embeddings(self, propositions: List[Proposition]) -> np.ndarray:
        """Get embeddings for propositions."""
        if hasattr(self.coherence_measure, "encoder"):
            texts = [p.text for p in propositions]
            embeddings = self.coherence_measure.encoder.encode(texts)
            return np.array(embeddings)
        else:
            # Fallback to simple text features
            return self._text_features(propositions)

    def _text_features(self, propositions: List[Proposition]) -> np.ndarray:
        """Simple text-based features when embeddings unavailable."""
        features = []

        for prop in propositions:
            text = prop.text.lower()
            feature_vector = [
                len(text),  # Length
                len(text.split()),  # Word count
                text.count("."),  # Sentence count
                text.count(","),  # Comma count
                len(set(text.split())),  # Unique words
            ]
            features.append(feature_vector)

        return np.array(features)


class ClusterBasedApproximator(PropositionClusterer):
    """
    Cluster-based coherence approximation.

    Groups similar propositions and computes coherence using cluster representatives.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        clustering_method: str = "kmeans",
        num_clusters: Optional[int] = None,
        cluster_selection_strategy: str = "auto",
    ):
        """
        Initialize cluster-based approximator.

        Args:
            coherence_measure: Base coherence measure
            clustering_method: Clustering algorithm ('kmeans', 'hierarchical', 'spectral')
            num_clusters: Number of clusters (auto-determined if None)
            cluster_selection_strategy: How to select representatives ('centroid', 'medoid', 'random')
        """
        super().__init__(coherence_measure)
        self.clustering_method = clustering_method
        self.num_clusters = num_clusters
        self.cluster_selection_strategy = cluster_selection_strategy

    def approximate_coherence(
        self, prop_set: PropositionSet, target_clusters: Optional[int] = None
    ) -> ClusteringResult:
        """
        Approximate coherence using clustering.

        Args:
            prop_set: Proposition set to analyze
            target_clusters: Target number of clusters (overrides default)

        Returns:
            ClusteringResult with approximation
        """
        start_time = time.time()

        propositions = prop_set.propositions
        n_props = len(propositions)

        # Determine number of clusters
        n_clusters = (
            target_clusters
            or self.num_clusters
            or self._determine_optimal_clusters(n_props)
        )
        n_clusters = min(n_clusters, n_props)

        if n_clusters >= n_props:
            # No clustering needed
            clustering_time = time.time() - start_time
            comp_start = time.time()
            result = self.coherence_measure.compute(prop_set)
            comp_time = time.time() - comp_start

            return ClusteringResult(
                approximate_score=result.score,
                num_clusters=n_props,
                total_propositions=n_props,
                clustering_time=clustering_time,
                computation_time=comp_time,
                cluster_coherences=[result.score],
                cluster_sizes=[n_props],
                clustering_method="none",
                metadata={"exact_computation": True},
            )

        # Get embeddings
        embeddings = self._get_embeddings(propositions)

        # Perform clustering
        cluster_labels = self._cluster(embeddings, n_clusters)
        clustering_time = time.time() - start_time

        # Get cluster representatives
        representatives = self._get_cluster_representatives(
            propositions, embeddings, cluster_labels, n_clusters
        )

        # Compute coherence on representatives
        comp_start = time.time()
        representative_set = PropositionSet(
            propositions=representatives,
            context=prop_set.context,
            metadata=prop_set.metadata,
        )
        result = self.coherence_measure.compute(representative_set)
        comp_time = time.time() - comp_start

        # Compute per-cluster coherences
        cluster_coherences, cluster_sizes = self._compute_cluster_coherences(
            propositions, cluster_labels, n_clusters
        )

        return ClusteringResult(
            approximate_score=result.score,
            num_clusters=n_clusters,
            total_propositions=n_props,
            clustering_time=clustering_time,
            computation_time=comp_time,
            cluster_coherences=cluster_coherences,
            cluster_sizes=cluster_sizes,
            clustering_method=self.clustering_method,
            metadata={
                "reduction_ratio": len(representatives) / n_props,
                "cluster_selection": self.cluster_selection_strategy,
            },
        )

    def _determine_optimal_clusters(self, n_propositions: int) -> int:
        """Determine optimal number of clusters based on proposition count."""
        if n_propositions <= 10:
            return n_propositions
        elif n_propositions <= 50:
            return max(5, n_propositions // 3)
        elif n_propositions <= 200:
            return max(10, n_propositions // 5)
        else:
            return max(20, int(np.sqrt(n_propositions)))

    def _cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform clustering on embeddings."""
        try:
            if self.clustering_method == "kmeans":
                from sklearn.cluster import KMeans

                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                return clusterer.fit_predict(embeddings)

            elif self.clustering_method == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering

                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                return clusterer.fit_predict(embeddings)

            elif self.clustering_method == "spectral":
                from sklearn.cluster import SpectralClustering

                clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
                return clusterer.fit_predict(embeddings)

            else:
                # Fallback to simple partitioning
                cluster_size = len(embeddings) // n_clusters
                labels = []
                for i in range(len(embeddings)):
                    labels.append(min(i // cluster_size, n_clusters - 1))
                return np.array(labels)

        except ImportError:
            # Fallback clustering if sklearn not available
            return self._simple_clustering(embeddings, n_clusters)

    def _simple_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple clustering when sklearn unavailable."""
        # K-means-like clustering using basic numpy
        n_points = len(embeddings)

        # Initialize centroids randomly
        np.random.seed(42)
        centroid_indices = np.random.choice(n_points, n_clusters, replace=False)
        centroids = embeddings[centroid_indices]

        labels = np.zeros(n_points, dtype=int)

        # Simple iterative assignment
        for iteration in range(10):  # Max 10 iterations
            # Assign points to nearest centroid
            for i, point in enumerate(embeddings):
                distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                labels[i] = np.argmin(distances)

            # Update centroids
            new_centroids = []
            for k in range(n_clusters):
                cluster_points = embeddings[labels == k]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    new_centroids.append(centroids[k])  # Keep old centroid

            centroids = np.array(new_centroids)

        return labels

    def _get_cluster_representatives(
        self,
        propositions: List[Proposition],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
    ) -> List[Proposition]:
        """Get representative propositions for each cluster."""
        representatives = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            if self.cluster_selection_strategy == "centroid":
                # Find proposition closest to cluster centroid
                cluster_embeddings = embeddings[cluster_mask]
                centroid = np.mean(cluster_embeddings, axis=0)

                distances = [
                    np.linalg.norm(emb - centroid) for emb in cluster_embeddings
                ]
                closest_idx = np.argmin(distances)
                representative_idx = cluster_indices[closest_idx]

            elif self.cluster_selection_strategy == "medoid":
                # Find proposition with minimum total distance to all others in cluster
                cluster_embeddings = embeddings[cluster_mask]
                min_total_distance = float("inf")
                medoid_idx = 0

                for i, emb1 in enumerate(cluster_embeddings):
                    total_distance = sum(
                        np.linalg.norm(emb1 - emb2) for emb2 in cluster_embeddings
                    )
                    if total_distance < min_total_distance:
                        min_total_distance = total_distance
                        medoid_idx = i

                representative_idx = cluster_indices[medoid_idx]

            else:  # random
                representative_idx = np.random.choice(cluster_indices)

            representatives.append(propositions[representative_idx])

        return representatives

    def _compute_cluster_coherences(
        self,
        propositions: List[Proposition],
        cluster_labels: np.ndarray,
        n_clusters: int,
    ) -> Tuple[List[float], List[int]]:
        """Compute coherence within each cluster."""
        cluster_coherences = []
        cluster_sizes = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_props = [propositions[i] for i in cluster_indices]

            cluster_sizes.append(len(cluster_props))

            if len(cluster_props) <= 1:
                cluster_coherences.append(
                    1.0
                )  # Single proposition is perfectly coherent
            else:
                cluster_set = PropositionSet(propositions=cluster_props)
                result = self.coherence_measure.compute(cluster_set)
                cluster_coherences.append(result.score)

        return cluster_coherences, cluster_sizes


class HierarchicalCoherenceApproximator(PropositionClusterer):
    """
    Hierarchical coherence approximation.

    Builds a hierarchy of clusters and computes coherence at different levels.
    """

    def __init__(
        self,
        coherence_measure: Optional[CoherenceMeasure] = None,
        max_depth: int = 4,
        min_cluster_size: int = 2,
    ):
        """
        Initialize hierarchical approximator.

        Args:
            coherence_measure: Base coherence measure
            max_depth: Maximum hierarchy depth
            min_cluster_size: Minimum propositions per cluster
        """
        super().__init__(coherence_measure)
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size

    def approximate_coherence(
        self, prop_set: PropositionSet, target_depth: Optional[int] = None
    ) -> ClusteringResult:
        """
        Approximate coherence using hierarchical clustering.

        Args:
            prop_set: Proposition set to analyze
            target_depth: Target hierarchy depth

        Returns:
            ClusteringResult with hierarchical approximation
        """
        start_time = time.time()

        propositions = prop_set.propositions
        depth = target_depth or min(self.max_depth, int(np.log2(len(propositions))))

        # Build hierarchy
        hierarchy = self._build_hierarchy(propositions, depth)
        clustering_time = time.time() - start_time

        # Compute coherence at target level
        comp_start = time.time()
        target_level = hierarchy[min(depth - 1, len(hierarchy) - 1)]

        # Get representatives from target level
        representatives = [cluster["representative"] for cluster in target_level]
        representative_set = PropositionSet(
            propositions=representatives,
            context=prop_set.context,
            metadata=prop_set.metadata,
        )

        result = self.coherence_measure.compute(representative_set)
        comp_time = time.time() - comp_start

        # Extract cluster information
        cluster_coherences = [cluster["coherence"] for cluster in target_level]
        cluster_sizes = [len(cluster["propositions"]) for cluster in target_level]

        return ClusteringResult(
            approximate_score=result.score,
            num_clusters=len(target_level),
            total_propositions=len(propositions),
            clustering_time=clustering_time,
            computation_time=comp_time,
            cluster_coherences=cluster_coherences,
            cluster_sizes=cluster_sizes,
            clustering_method="hierarchical",
            metadata={
                "hierarchy_depth": depth,
                "levels_built": len(hierarchy),
                "reduction_ratio": len(representatives) / len(propositions),
            },
        )

    def _build_hierarchy(
        self, propositions: List[Proposition], max_depth: int
    ) -> List[List[Dict[str, Any]]]:
        """Build hierarchical clustering."""
        hierarchy = []

        # Level 0: Individual propositions
        current_level = []
        for prop in propositions:
            current_level.append(
                {
                    "propositions": [prop],
                    "representative": prop,
                    "coherence": 1.0,
                    "embedding": None,
                }
            )

        # Get embeddings for all propositions
        embeddings = self._get_embeddings(propositions)
        for i, cluster in enumerate(current_level):
            cluster["embedding"] = embeddings[i]

        hierarchy.append(current_level)

        # Build higher levels
        for depth in range(1, max_depth):
            if len(current_level) <= 1:
                break

            # Target number of clusters for this level
            target_clusters = max(1, len(current_level) // 2)

            # Merge most similar clusters
            next_level = self._merge_clusters(current_level, target_clusters)

            if len(next_level) >= len(current_level):
                break  # No meaningful clustering possible

            hierarchy.append(next_level)
            current_level = next_level

        return hierarchy

    def _merge_clusters(
        self, clusters: List[Dict[str, Any]], target_count: int
    ) -> List[Dict[str, Any]]:
        """Merge clusters to reach target count."""
        current_clusters = clusters.copy()

        while len(current_clusters) > target_count:
            # Find most similar pair
            best_pair = None
            best_similarity = -1

            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    # Compute similarity between cluster embeddings
                    emb1 = current_clusters[i]["embedding"]
                    emb2 = current_clusters[j]["embedding"]

                    similarity = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, j)

            if best_pair is None:
                break

            # Merge the best pair
            i, j = best_pair
            cluster1 = current_clusters[i]
            cluster2 = current_clusters[j]

            # Combine propositions
            merged_props = cluster1["propositions"] + cluster2["propositions"]

            # Compute merged coherence
            if len(merged_props) > 1:
                merged_set = PropositionSet(propositions=merged_props)
                result = self.coherence_measure.compute(merged_set)
                merged_coherence = result.score
            else:
                merged_coherence = 1.0

            # Choose representative (highest individual coherence)
            rep_scores = []
            for prop in merged_props:
                single_set = PropositionSet(propositions=[prop])
                score = 1.0  # Single propositions are perfectly coherent
                rep_scores.append((prop, score))

            representative = max(rep_scores, key=lambda x: x[1])[0]

            # Compute merged embedding
            merged_embedding = (cluster1["embedding"] + cluster2["embedding"]) / 2

            # Create merged cluster
            merged_cluster = {
                "propositions": merged_props,
                "representative": representative,
                "coherence": merged_coherence,
                "embedding": merged_embedding,
            }

            # Remove original clusters and add merged
            current_clusters = [
                c for idx, c in enumerate(current_clusters) if idx not in (i, j)
            ]
            current_clusters.append(merged_cluster)

        return current_clusters

    def get_hierarchy_summary(self, prop_set: PropositionSet) -> Dict[str, Any]:
        """Get summary of hierarchical structure."""
        hierarchy = self._build_hierarchy(prop_set.propositions, self.max_depth)

        summary = {
            "total_propositions": len(prop_set.propositions),
            "hierarchy_depth": len(hierarchy),
            "levels": [],
        }

        for level_idx, level in enumerate(hierarchy):
            level_info = {
                "level": level_idx,
                "num_clusters": len(level),
                "cluster_sizes": [len(c["propositions"]) for c in level],
                "coherence_scores": [c["coherence"] for c in level],
                "reduction_ratio": len(level) / len(prop_set.propositions),
            }
            summary["levels"].append(level_info)

        return summary
