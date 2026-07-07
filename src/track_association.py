# PURPOSE: Data association between tracked cloud layers and new frame cluster
#          measurements. Uses Hungarian algorithm on Mahalanobis distance cost
#          with chi-squared gating and physical plausibility pre-checks.
# INPUTS: List of CloudTrack, list of ClusterMeasurement, config
# OUTPUTS: (assignments, unassigned_cluster_indices, unassigned_tracks)
# KEYWORDS: association, hungarian, mahalanobis, gating, chi_squared, data_association
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from typing import Optional

from .spatial_filter import validate_clusters_spatial

logger = logging.getLogger(__name__)

# Pre-compute chi-squared gate values
_CHI2_GATE_4DOF = chi2.ppf(0.99, 4)  # ≈ 13.28


# ---------------------------------------------------------------------------
# Build Kalman matrices (7-DOF constant-velocity model)
# ---------------------------------------------------------------------------

# PURPOSE: Construct the Kalman filter matrices F, H, Q for the 7-DOF
#          constant-velocity cloud track model.
# INPUTS: dt (float seconds), process noise sigmas
# OUTPUTS: dict with keys F, H, Q (all np.ndarray)
# KEYWORDS: kalman, matrices, constant_velocity, process_noise, 7DOF
def _build_matrices(dt: float, sigma_h: float, sigma_u: float,
                    sigma_v: float, sigma_s: float) -> dict:
    F = np.eye(7)
    F[0, 1] = dt
    F[2, 3] = dt
    F[4, 5] = dt

    H = np.zeros((4, 7))
    H[0, 0] = 1.0  # h
    H[1, 2] = 1.0  # u
    H[2, 3] = 1.0  # v
    H[3, 6] = 1.0  # s

    # Piecewise constant white acceleration model (Bar-Shalom §6.3.2)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    Q_h = sigma_h**2 * np.array([
        [dt4 / 4, dt3 / 2],
        [dt3 / 2, dt2],
    ])

    Q_u = sigma_u**2 * np.array([
        [dt4 / 4, dt3 / 2],
        [dt3 / 2, dt2],
    ])

    Q_v = sigma_v**2 * np.array([
        [dt4 / 4, dt3 / 2],
        [dt3 / 2, dt2],
    ])

    Q = np.zeros((7, 7))
    Q[0:2, 0:2] = Q_h
    Q[2:4, 2:4] = Q_u
    Q[4:6, 4:6] = Q_v
    Q[6, 6] = sigma_s**2 * dt

    return {"F": F, "H": H, "Q": Q}


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------

# PURPOSE: Compute squared Mahalanobis distance between track prediction
#          and cluster measurement for association gating.
# INPUTS: track (CloudTrack), cluster (ClusterMeasurement), H (4x7),
#         R (4x4 measurement noise covariance)
# OUTPUTS: squared Mahalanobis distance (float), or inf if singular
# KEYWORDS: mahalanobis, innovation, covariance, gating
def compute_mahalanobis_sq(track, cluster, H: np.ndarray,
                           R: np.ndarray) -> float:
    z = np.array([cluster.h, cluster.u, cluster.v,
                  float(cluster.size)], dtype=np.float64)
    innovation = z - H @ track.x
    S = H @ track.P @ H.T + R
    try:
        d2 = float(innovation.T @ np.linalg.solve(S, innovation))
        if not np.isfinite(d2):
            return 1e9
        return d2
    except np.linalg.LinAlgError:
        return 1e9


# ---------------------------------------------------------------------------
# Physical plausibility gating
# ---------------------------------------------------------------------------

# PURPOSE: Reject physically-impossible track-cluster associations before
#          computing the Mahalanobis distance.
# INPUTS: track, cluster, dt (float seconds), config
# OUTPUTS: bool — True if the association is physically plausible
# KEYWORDS: gating, physical_plausibility, speed_limit, size_ratio
def gate_physical(track, cluster, dt: float, config: dict) -> bool:
    # Height rate check
    max_dhdt = config.get("max_height_rate", 10.0)
    dhdt = abs(cluster.h - track.x[0]) / max(dt, 1e-6)
    if dhdt > max_dhdt:
        logger.debug("Gate FAIL height rate: %.1f m/s > %.1f",
                     dhdt, max_dhdt)
        return False

    # Image-plane speed check
    max_app = config.get("max_apparent_speed", 100.0)
    du = cluster.u - track.x[2]
    dv = cluster.v - track.x[3]
    app_speed = np.sqrt(du**2 + dv**2) / max(dt, 1e-6)
    if app_speed > max_app:
        logger.debug("Gate FAIL apparent speed: %.1f px/s > %.1f",
                     app_speed, max_app)
        return False

    # Cluster size ratio check
    size_pred = max(track.x[6], 1.0)
    ratio = cluster.size / size_pred
    min_r = config.get("min_size_ratio", 0.2)
    max_r = config.get("max_size_ratio", 5.0)
    if ratio < min_r or ratio > max_r:
        logger.debug("Gate FAIL size ratio: %.2f not in [%.2f, %.2f]",
                     ratio, min_r, max_r)
        return False

    return True


# ---------------------------------------------------------------------------
# Hungarian association
# ---------------------------------------------------------------------------

# PURPOSE: Run Hungarian algorithm over the gate-validated Mahalanobis cost
#          matrix to optimally assign clusters to existing tracks.
# INPUTS: tracks (list[CloudTrack]), clusters (list[ClusterMeasurement]),
#         config (dict)
# OUTPUTS: (assignments, unassigned_cluster_idx, unassigned_tracks)
# KEYWORDS: hungarian, assignment, optimal, bipartite_matching
def associate_frame(
    tracks: list,
    clusters: list,
    config: dict,
    H: Optional[np.ndarray] = None,
) -> tuple:
    """
    Associate clusters to tracks using Hungarian algorithm.

    Returns:
        assignments: list of (track_idx, cluster_idx) pairs
        unassigned_clusters: list of cluster indices
        unassigned_tracks: list of track references
    """
    n_tracks = len(tracks)
    n_clusters = len(clusters)

    if n_tracks == 0 or n_clusters == 0:
        return [], list(range(n_clusters)), list(tracks)

    chi2_gate = config.get("chi2_gate", _CHI2_GATE_4DOF)
    dt = config.get("_dt", 10.0)

    if H is None:
        H = np.zeros((4, 7))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        H[2, 3] = 1.0
        H[3, 6] = 1.0

    cost = np.full((n_tracks, n_clusters), 1e9, dtype=np.float64)

    for i, track in enumerate(tracks):
        for j, cluster in enumerate(clusters):
            if not gate_physical(track, cluster, dt, config):
                continue
            R = _build_R_for_cluster(cluster, config)
            d2 = compute_mahalanobis_sq(track, cluster, H, R)
            if d2 <= chi2_gate:
                cost[i, j] = d2

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = []
    assigned_clusters = set()
    assigned_tracks = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 1e9:
            assignments.append((i, j))
            assigned_clusters.add(j)
            assigned_tracks.add(i)

    unassigned_clusters = [j for j in range(n_clusters)
                           if j not in assigned_clusters]
    unassigned_tracks = [tracks[i] for i in range(n_tracks)
                         if i not in assigned_tracks]

    return assignments, unassigned_clusters, unassigned_tracks


# PURPOSE: Build measurement noise covariance R for a cluster based on its
#          height uncertainty, size, and image-space position.
# INPUTS: cluster (ClusterMeasurement), config
# OUTPUTS: 4x4 np.ndarray diagonal covariance matrix
# KEYWORDS: measurement_noise, covariance, adaptive, R_matrix
def _build_R_for_cluster(cluster, config: dict) -> np.ndarray:
    scale = config.get("centroid_uncertainty_scale", 1.0)
    img_w = config.get("image_width", 1920)
    s = max(cluster.size, 1)
    sigma_u_sq = scale * (img_w / s) ** 2
    return np.diag([
        max(cluster.h_err ** 2, 1.0),
        sigma_u_sq,
        sigma_u_sq,
        max(float(s), 1.0),
    ])
