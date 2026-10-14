# PURPOSE: Cluster correspondence disparities to separate distinct cloud layers
#          and reject outlying matches, and provide a stable colour map for those
#          clusters. This is disparity-space clustering; image-space coherence
#          checking lives in clustering/spatial.py.
# INPUTS: an (N,2) array of disparity vectors; HDBSCAN parameters.
# OUTPUTS: cluster centroids sorted by size, their standard deviations, the
#          per-match label array and a label->colour map.
# KEYWORDS: clustering, hdbscan, disparity, layers, outliers
from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN


def cluster_shifts(shift_arr: np.ndarray, hdbscan_config: dict, show_clusters=False) -> tuple:
    """
    Clusters shift vectors using HDBSCAN to find dominant motion patterns.

    Args:
        shift_arr (np.ndarray): An array of (dx, dy) shift vectors.
        hdbscan_config (dict): Configuration for the HDBSCAN algorithm.
        show_clusters (bool): If True, plots the clustered shift vectors.

    Returns:
        A tuple containing:
        - sorted_centroids (list): Cluster centroids, sorted by cluster size (largest first).
        - std_devs (list): Standard deviations for each corresponding cluster.
        - labels (np.ndarray): The cluster label for each input shift vector.
        - colors (dict): A mapping from cluster label to a color.
    """
    hdbscan = HDBSCAN(
        min_cluster_size=hdbscan_config["min_cluster_size"],
        min_samples=hdbscan_config["min_samples"],
        allow_single_cluster=hdbscan_config.get("allow_single_cluster", True),
        store_centers='centroid'
    )
    
    if len(shift_arr) < hdbscan_config.get("min_samples", 1):
        labels = np.full(len(shift_arr), -1, dtype=int)
        colors = get_cluster_colors(labels)
        return [], [], labels, colors

    try:
        hdbscan.fit(shift_arr)
    except Exception as exc:
        # A silent all-noise fallback would look exactly like a clear sky, so the
        # reason is printed. The previous epsilon fallback was removed with the
        # epsilon knob itself: sklearn 1.9's HDBSCAN raises
        # "TypeError: only 0-dimensional arrays can be converted to Python scalars"
        # for any cluster_selection_epsilon > 0 on this data, so the knob could
        # never legitimately be non-zero.
        print(f"  HDBSCAN failed ({exc}); returning no clusters for this frame")
        labels = np.full(len(shift_arr), -1, dtype=int)
        colors = get_cluster_colors(labels)
        return [], [], labels, colors
    labels = hdbscan.labels_
    colors = get_cluster_colors(labels)

    if show_clusters:
        # Visualization call is now cleanly separated
        from ..viz.plots import plot_shift_vectors
        plot_shift_vectors(shift_arr, labels, hdbscan.probabilities_, colors, hdbscan_config)

    # Calculate centroids and standard deviations for valid clusters
    power_of_cluster = collections.Counter(labels)
    valid_clusters = {label: count for label, count in power_of_cluster.items() if label >= 0}

    if not valid_clusters:
        return [], [], labels, colors

    # Sort cluster IDs by size
    sorted_cluster_ids = sorted(valid_clusters.keys(), key=lambda k: valid_clusters[k], reverse=True)
    
    centroids = hdbscan.centroids_
    sorted_centroids = [centroids[i] for i in sorted_cluster_ids]
    
    std_devs = []
    for cid in sorted_cluster_ids:
        cluster_points = shift_arr[labels == cid]
        std_devs.append(np.std(cluster_points, axis=0))

    return sorted_centroids, std_devs, labels, colors

def get_cluster_colors(labels: np.ndarray) -> dict:
    """Generates a consistent color map for a set of cluster labels."""
    unique_labels = sorted(list(set(labels)))
    n_clusters = len(unique_labels)
    # Using 'viridis' which is perceptually uniform and good for data visualization
    colormap = plt.get_cmap('viridis', n_clusters)
    colors = {}
    for i, label in enumerate(unique_labels):
        if label == -1:
            colors[label] = (0.5, 0.5, 0.5)  # Gray for noise
        else:
            colors[label] = colormap(i)[:3]  # RGB in 0-1 range
    return colors
