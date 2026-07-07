# PURPOSE: Validate HDBSCAN clusters for spatial coherence before they enter
#          the temporal tracker. Rejects clusters whose keypoints are scattered
#          across the image (likely false matches on sky or different cloud regions
#          at coincidentally-similar disparity).
# INPUTS: Cluster keypoints (kp1 np.ndarray Nx2), cluster disparities (dx array),
#         cluster disparity std, configuration dict
# OUTPUTS: Boolean mask (N,) indicating which clusters pass spatial validation
# KEYWORDS: spatial_coherence, DBSCAN_subclustering, density, convex_hull,
#           spatial_disparity_consistency, sky_rejection, cloud_validation
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, cKDTree
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# PURPOSE: Validate all clusters in a frame via spatial coherence checks.
#          Each cluster is independently tested; clusters that fail are removed.
# INPUTS: kp1 (np.ndarray Nx2), labels (np.ndarray N,), shifts (np.ndarray Nx2),
#         centers (np.ndarray Mx2), config (dict)
# OUTPUTS: valid_mask (np.ndarray bool M,) — True for clusters that pass
# KEYWORDS: spatial_filter, cluster_validation, coherence, frame
def validate_clusters_spatial(
    kp1: np.ndarray,
    labels: np.ndarray,
    shifts: np.ndarray,
    centers: np.ndarray,
    config: dict,
) -> dict:
    """
    Validate each HDBSCAN cluster for spatial coherence in image space.

    Returns a dict mapping cluster_label (int) → bool (True = passes).
    """
    if not config.get("spatial_filter_enabled", True):
        result = {}
        for label in sorted(set(labels)):
            if label >= 0:
                result[label] = True
        return result

    unique_labels = sorted(set(labels))
    result = {}

    for label in unique_labels:
        if label < 0:
            continue

        mask = labels == label
        cluster_kp1 = kp1[mask]
        cluster_shifts = shifts[mask]
        cluster_size = len(cluster_kp1)

        if cluster_size < 3:
            result[label] = False
            logger.debug("Cluster %d REJECTED: too few points (%d)",
                         label, cluster_size)
            continue

        # Stage 1: Image-space DBSCAN sub-clustering
        sub_ok, sub_kp1, sub_shifts = _check_sub_cluster(
            cluster_kp1, cluster_shifts, config
        )
        if not sub_ok:
            result[label] = False
            logger.debug("Cluster %d REJECTED: failed sub-clustering", label)
            continue

        # Stage 2: Density / convex hull check
        if not _check_density(sub_kp1, config):
            result[label] = False
            logger.debug("Cluster %d REJECTED: density too low", label)
            continue

        # Stage 3: Joint spatial-disparity consistency
        if not _check_spatial_disparity(sub_kp1, sub_shifts, config):
            result[label] = False
            logger.debug("Cluster %d REJECTED: spatial-disparity inconsistent", label)
            continue

        result[label] = True
        logger.debug("Cluster %d PASSED spatial filter | size=%d",
                     label, len(sub_kp1))

    n_rejected = sum(1 for v in result.values() if not v)
    if n_rejected > 0:
        logger.info("Spatial filter: %d/%d clusters rejected",
                    n_rejected, len(result))
    return result


# ---------------------------------------------------------------------------
# Stage 1: Image-space DBSCAN sub-clustering
# ---------------------------------------------------------------------------

# PURPOSE: Check if a cluster's keypoints form a single contiguous cloud
#          or several disconnected blobs (different clouds at same disparity).
#          Runs DBSCAN in image space and keeps only the dominant sub-cluster.
# INPUTS: kp1 (Nx2), shifts (Nx2), config
# OUTPUTS: (passes, filtered_kp1, filtered_shifts) — boolean and filtered arrays
# KEYWORDS: sub_clustering, DBSCAN, image_space, contiguity, fragmentation
def _check_sub_cluster(kp1: np.ndarray, shifts: np.ndarray, config: dict):
    eps = config.get("spatial_dbscan_eps", 30.0)
    min_samples = config.get("spatial_dbscan_min_samples", 3)

    if len(kp1) < min_samples:
        return False, kp1, shifts

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    sub_labels = db.fit_predict(kp1)

    unique_subs = sorted(set(sub_labels))
    # Remove noise label (-1)
    real_subs = [s for s in unique_subs if s >= 0]

    if len(real_subs) == 0:
        return False, kp1, shifts

    # Find largest sub-cluster
    sub_sizes = {s: int(np.sum(sub_labels == s)) for s in real_subs}
    largest_label = max(real_subs, key=lambda s: sub_sizes[s])

    keep_mask = sub_labels == largest_label
    filtered_kp1 = kp1[keep_mask]
    filtered_shifts = shifts[keep_mask]

    if len(filtered_kp1) < min_samples:
        return False, filtered_kp1, filtered_shifts

    return True, filtered_kp1, filtered_shifts


# ---------------------------------------------------------------------------
# Stage 2: Density / convex hull check
# ---------------------------------------------------------------------------

# PURPOSE: Verify that cluster keypoints are densely packed relative to their
#          spatial extent. Scattered points with large convex hull area but
#          few points indicate a non-physical cluster (e.g., false matches
#          at opposite edges of the frame).
# INPUTS: kp1 (Nx2), config
# OUTPUTS: bool — True if density is sufficient
# KEYWORDS: density, convex_hull, area, compactness, scatter
def _check_density(kp1: np.ndarray, config: dict) -> bool:
    density_min = config.get("spatial_density_min", 0.002)

    if len(kp1) < 4:
        return len(kp1) >= 3

    try:
        hull = ConvexHull(kp1)
        area = hull.volume  # In 2D, `volume` is area
    except Exception:
        return len(kp1) >= 3

    if area <= 0:
        return True

    density = len(kp1) / area
    return density >= density_min


# ---------------------------------------------------------------------------
# Stage 3: Joint spatial-disparity consistency
# ---------------------------------------------------------------------------

# PURPOSE: Verify the physical property that points close in image space
#          should have similar disparities (continuous 3D surface).
#          For each keypoint, checks its K nearest image-space neighbors
#          for disparity consistency; rejects cluster if too many points
#          have inconsistent neighbors.
# INPUTS: kp1 (Nx2), shifts (Nx2), config
# OUTPUTS: bool — True if the cluster is sufficiently consistent
# KEYWORDS: spatial_disparity, consistency, kdtree, neighbors, surface
def _check_spatial_disparity(
    kp1: np.ndarray, shifts: np.ndarray, config: dict
) -> bool:
    k = config.get("spatial_k_neighbors", 5)
    max_inconsistent_frac = config.get("spatial_inconsistency_max", 0.30)
    std_mult = config.get("spatial_disparity_std_mult", 2.0)

    n = len(kp1)
    if n < k + 1:
        return n >= 3

    dx = shifts[:, 0]
    cluster_std = float(np.std(dx))

    if cluster_std < 1e-8:
        return True

    tree = cKDTree(kp1)
    k_eff = min(k + 1, n)
    _, indices = tree.query(kp1, k=k_eff)

    n_inconsistent = 0
    for i in range(n):
        neighbor_idx = indices[i][1:]  # exclude self (index 0)
        neighbor_dx = dx[neighbor_idx]
        neighbor_std = float(np.std(neighbor_dx))
        if neighbor_std > std_mult * cluster_std:
            n_inconsistent += 1

    inconsistent_frac = n_inconsistent / n
    return inconsistent_frac <= max_inconsistent_frac
