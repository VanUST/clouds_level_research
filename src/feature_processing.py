# /src/feature_processing.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import collections
import torch

# Global cache for the active matcher instance
_matcher_instance = None
_matcher_backend = None


# PURPOSE: Create or retrieve the cached feature matcher function for the active backend.
# INPUTS: backend_name (str), config (dict)
# OUTPUTS: callable (img0, img1, visualize=False) -> (np.ndarray, np.ndarray)
# KEYWORDS: lazy_init, matcher, singleton, cache, backend
def _get_matcher(backend_name, config):
    global _matcher_instance, _matcher_backend
    if _matcher_instance is not None and _matcher_backend == backend_name:
        return _matcher_instance
    from .feature_matchers import create_matcher
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _matcher_instance = create_matcher(backend_name, config, device)
    _matcher_backend = backend_name
    return _matcher_instance


# PURPOSE: Find corresponding feature points between two images using the configured matcher backend,
#          with optional match filtering (epipolar, confidence, disparity-MAD, cycle-consistency).
# INPUTS: img1 (np.ndarray), img2 (np.ndarray), visualize (bool), match_thresh (float),
#         backend (str) – one of 'kornia_loftr', 'matchanything_eloftr',
#         filter_config (dict|None) – controls post-match filtering
# OUTPUTS: Tuple[np.ndarray, np.ndarray] keypoints from image1 and image2.
# KEYWORDS: feature_matching, matcher, loftr, eloftr, stereo, correspondences, filtering
def get_corresponding_feature_points(img1, img2, visualize=False, match_thresh=0.5,
                                     backend="kornia_loftr", filter_config=None, **kwargs):
    """
    Finds corresponding feature points between two images.

    Args:
        img1 (np.ndarray): The first image (BGR or grayscale).
        img2 (np.ndarray): The second image (BGR or grayscale).
        visualize (bool): If True, displays the matches.
        match_thresh (float): Confidence threshold for the matcher.
        backend (str): Matcher backend: 'kornia_loftr' or 'matchanything_eloftr'.
        filter_config (dict|None): Match filtering configuration with keys:
            - 'y_thresh_px' (float): Max allowed y-disparity in pixels (epipolar).
            - 'conf_thresh' (float): Min match confidence score.
            - 'mad_n' (float): Disparity MAD multiplier for outlier removal.
            - 'cycle_consistency' (bool): Enable bidirectional cycle check.
            - 'cycle_pixel_thresh' (float): Pixel tolerance for cycle consistency.
        **kwargs: Additional backend-specific config.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Keypoints from image 1, keypoints from image 2.
    """
    from .feature_matchers import (filter_mask_epipolar, filter_mask_confidence,
                                    filter_mask_disparity_mad, apply_filter_masks,
                                    filter_matches_cycle_consistency)

    config = {"match_threshold": match_thresh, **kwargs}
    matcher = _get_matcher(backend, config)

    fc = filter_config or {}

    # Cycle-consistency: runs matcher bidirectionally, takes priority
    if fc.get("cycle_consistency"):
        pixel_thresh = fc.get("cycle_pixel_thresh", 2.0)
        kp1, kp2 = filter_matches_cycle_consistency(
            matcher, img1, img2, pixel_thresh=pixel_thresh
        )
        # Apply additional filters on the cycle-consistent result
        masks = []
        if fc.get("y_thresh_px") is not None:
            masks.append(filter_mask_epipolar(kp1, kp2, fc["y_thresh_px"]))
        if fc.get("mad_n") is not None and len(kp1) > 0:
            masks.append(filter_mask_disparity_mad(kp1, kp2, n_mad=fc["mad_n"]))
        kp1, kp2 = apply_filter_masks(kp1, kp2, masks)
        return kp1, kp2

    # Standard matching with optional confidence return
    result = matcher(img1, img2, visualize=visualize, return_conf=fc.get("conf_thresh") is not None)
    if isinstance(result, tuple) and len(result) == 3:
        kp1, kp2, conf = result
    else:
        kp1, kp2 = result
        conf = None

    # Build filter masks
    masks = []
    if fc.get("y_thresh_px") is not None and len(kp1) > 0:
        masks.append(filter_mask_epipolar(kp1, kp2, fc["y_thresh_px"]))
    if fc.get("conf_thresh") is not None and conf is not None:
        masks.append(filter_mask_confidence(conf, fc["conf_thresh"]))
    if fc.get("mad_n") is not None and len(kp1) > 0:
        masks.append(filter_mask_disparity_mad(kp1, kp2, n_mad=fc["mad_n"]))

    kp1, kp2 = apply_filter_masks(kp1, kp2, masks)
    return kp1, kp2


# PURPOSE: Backward-compatible wrapper that delegates to the Kornia LoFTR backend.
# INPUTS: Same as get_corresponding_feature_points with backend='kornia_loftr'.
# OUTPUTS: Tuple[np.ndarray, np.ndarray]
# KEYWORDS: legacy, loftr, kornia, backward_compat
def get_corresponding_feature_points_loftr(img1, img2, visualize=False, match_thresh=0.5):
    """
    Legacy wrapper around Kornia LoFTR matcher (backward compatible).

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.
        visualize (bool): If True, displays the matches.
        match_thresh (float): The confidence threshold for LoFTR matches.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Keypoints from image 1, keypoints from image 2.
    """
    return get_corresponding_feature_points(
        img1, img2, visualize=visualize, match_thresh=match_thresh,
        backend="kornia_loftr"
    )

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
        cluster_selection_epsilon=hdbscan_config.get("cluster_selection_epsilon", hdbscan_config.get("epsilon", 0.0)),
        allow_single_cluster=hdbscan_config.get("allow_single_cluster", True),
        store_centers='centroid'
    )
    
    if len(shift_arr) < hdbscan_config.get("min_samples", 1):
        labels = np.full(len(shift_arr), -1, dtype=int)
        colors = get_cluster_colors(labels)
        return [], [], labels, colors

    try:
        hdbscan.fit(shift_arr)
    except Exception:
        print(f"  HDBSCAN failed with epsilon={hdbscan.cluster_selection_epsilon}, falling back")
        labels = np.full(len(shift_arr), -1, dtype=int)
        colors = get_cluster_colors(labels)
        return [], [], labels, colors
    labels = hdbscan.labels_
    colors = get_cluster_colors(labels)

    if show_clusters:
        # Visualization call is now cleanly separated
        from .visualization import plot_shift_vectors
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
    colormap = plt.cm.get_cmap('viridis', n_clusters)
    colors = {}
    for i, label in enumerate(unique_labels):
        if label == -1:
            colors[label] = (0.5, 0.5, 0.5)  # Gray for noise
        else:
            colors[label] = colormap(i)[:3]  # RGB in 0-1 range
    return colors