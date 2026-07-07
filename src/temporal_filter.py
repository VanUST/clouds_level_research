# PURPOSE: Temporal Kalman filter for cloud base height estimation.
#          Tracks cloud layers across frames using a constant-velocity model,
#          filters out physically-implausible height jumps via gating and innovation
#          checks, and emits smoothed, physically-consistent CBH estimates.
#          Applies spatial coherence pre-filtering per frame before tracking.
# INPUTS: Per-frame cluster measurements (height, centroid, size, uncertainty)
# OUTPUTS: Filtered height estimates per frame, track trajectories, track quality scores
# KEYWORDS: temporal, kalman, tracking, cloud_base_height, filtering, smoothing, rts
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from .track_association import (
    gate_physical, compute_mahalanobis_sq, associate_frame,
)
from .spatial_filter import validate_clusters_spatial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class TrackStatus(Enum):
    PROBATION = auto()
    CONFIRMED = auto()
    DEAD = auto()


@dataclass
class ClusterMeasurement:
    h: float
    h_err: float
    u: float
    v: float
    size: int
    disp_x: float
    disp_std: float
    cluster_id: int


# ---------------------------------------------------------------------------
# CloudTrack
# ---------------------------------------------------------------------------

# PURPOSE: Represent one tracked cloud layer with its Kalman state, covariance,
#          lifecycle status, quality score, and measurement history.
# INPUTS: track_id (int), initial_state (np.ndarray 7,), initial_cov (np.ndarray 7x7),
#         init_timestamp (float), init_measurement (ClusterMeasurement)
# OUTPUTS: Managed track with predict/update/lifecycle methods.
# KEYWORDS: track, kalman_state, covariance, lifecycle, prediction, update
class CloudTrack:
    __slots__ = (
        "track_id", "status", "x", "P", "history", "miss_count",
        "outlier_count", "consecutive_outliers", "probation_hits",
        "probation_frames", "innovation_history", "last_timestamp",
        "init_timestamp", "is_spatial_pass",
    )

    def __init__(self, track_id: int, initial_state: np.ndarray,
                 initial_cov: np.ndarray, init_timestamp: float,
                 init_measurement: ClusterMeasurement, spatial_pass: bool = True):
        self.track_id = track_id
        self.status = TrackStatus.PROBATION
        self.x = initial_state.copy()
        self.P = initial_cov.copy()
        self.history: list[dict] = []
        self.miss_count = 0
        self.outlier_count = 0
        self.consecutive_outliers = 0
        self.probation_hits = 1
        self.probation_frames = 1
        self.innovation_history: list[float] = []
        self.last_timestamp = init_timestamp
        self.init_timestamp = init_timestamp
        self.is_spatial_pass = spatial_pass

        self._record(init_timestamp, init_measurement, "init", float("nan"))

    def _record(self, timestamp: float, meas: Optional[ClusterMeasurement],
                event: str, innovation_mah: float):
        entry = {
            "timestamp": timestamp,
            "h": float(self.x[0]),
            "h_dot": float(self.x[1]),
            "u": float(self.x[2]),
            "v": float(self.x[3]),
            "u_dot": float(self.x[4]),
            "v_dot": float(self.x[5]),
            "size": float(self.x[6]),
            "status": self.status.name,
            "event": event,
            "innovation_mah": innovation_mah,
        }
        if meas is not None:
            entry["meas_h"] = meas.h
            entry["meas_u"] = meas.u
            entry["meas_v"] = meas.v
            entry["meas_size"] = meas.size
        self.history.append(entry)

    def predict(self, dt: float, Q: np.ndarray, F: np.ndarray):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas: ClusterMeasurement, R: np.ndarray,
               H: np.ndarray, config: dict) -> bool:
        z = np.array([meas.h, meas.u, meas.v, float(meas.size)], dtype=np.float64)
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            d2 = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            d2 = 1e9

        self.innovation_history.append(d2)
        outlier_thresh = config.get("innovation_outlier_thresh", 5.0)

        if d2 > outlier_thresh ** 2:
            self.consecutive_outliers += 1
            self.outlier_count += 1
            self._record(self.last_timestamp, meas, "outlier_rejected", np.sqrt(d2))
            max_consec = config.get("max_consecutive_outliers", 3)
            if self.consecutive_outliers >= max_consec:
                self.status = TrackStatus.DEAD
                logger.debug("Track %d killed after %d consecutive outliers",
                             self.track_id, self.consecutive_outliers)
            return False

        self.consecutive_outliers = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(7) - K @ H) @ self.P

        if self.status == TrackStatus.PROBATION:
            self.probation_hits += 1
            self.probation_frames += 1

        self._record(self.last_timestamp, meas, "updated", np.sqrt(d2))
        return True

    def mark_miss(self, timestamp: float):
        self.miss_count += 1
        self._record(timestamp, None, "missed", float("nan"))

    def quality_score(self, config: dict) -> float:
        w = config.get("quality_weights", {"w_len": 0.3, "w_innov": 0.4, "w_size": 0.3})
        max_len = config.get("quality_max_len", 50)
        innov_max = config.get("quality_innov_max", 15.0)
        size_max = config.get("quality_size_max", 200)

        track_len = len([e for e in self.history if e["event"] in ("updated", "init")])
        score_len = min(1.0, track_len / max_len)

        innovs = [v for v in self.innovation_history if not np.isnan(v)]
        avg_innov = float(np.mean(innovs)) if innovs else float(innov_max)
        score_innov = max(0.0, 1.0 - avg_innov / innov_max)

        sizes = [e["size"] for e in self.history if e["event"] in ("updated", "init")]
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        score_size = min(1.0, avg_size / size_max)

        return w["w_len"] * score_len + w["w_innov"] * score_innov + w["w_size"] * score_size

    def lifetime_seconds(self) -> float:
        return self.last_timestamp - self.init_timestamp


# ---------------------------------------------------------------------------
# Cluster merging utilities (height-based, for HDBSCAN fragmentation)
# ---------------------------------------------------------------------------

# PURPOSE: Merge clusters with similar heights into composite "super-clusters"
#          to counter HDBSCAN fragmentation of a single cloud layer into
#          multiple small clusters.
# INPUTS: clusters (list[ClusterMeasurement]), height_tolerance_m (float)
# OUTPUTS: merged list[ClusterMeasurement]
# KEYWORDS: merge, height, cluster, fragmentation, consolidation
def _merge_clusters_by_height(clusters: list, tol_m: float) -> list:
    if len(clusters) <= 1:
        return clusters
    sorted_c = sorted(clusters, key=lambda c: c.h)
    merged = []
    current_group = [sorted_c[0]]
    for c in sorted_c[1:]:
        group_avg_h = np.mean([m.h for m in current_group])
        if abs(c.h - group_avg_h) < tol_m:
            current_group.append(c)
        else:
            merged.append(_merge_group(current_group))
            current_group = [c]
    if current_group:
        merged.append(_merge_group(current_group))
    return merged


def _merge_group(group: list) -> ClusterMeasurement:
    total_size = sum(c.size for c in group)
    weights = np.array([c.size for c in group], dtype=float)
    weights /= weights.sum()
    return ClusterMeasurement(
        h=float(np.average([c.h for c in group], weights=weights)),
        h_err=float(np.average([c.h_err for c in group], weights=weights)),
        u=float(np.average([c.u for c in group], weights=weights)),
        v=float(np.average([c.v for c in group], weights=weights)),
        size=total_size,
        disp_x=float(np.average([c.disp_x for c in group], weights=weights)),
        disp_std=float(np.average([c.disp_std for c in group], weights=weights)),
        cluster_id=group[0].cluster_id,
    )


# ---------------------------------------------------------------------------
# TemporalFilter
# ---------------------------------------------------------------------------

# PURPOSE: Orchestrate the full temporal filtering pipeline across all frames.
#          Manages track lifecycle (probation/confirmed/dead), data association,
#          Kalman predict/update cycles, innovation outlier rejection, and
#          multi-layer CBH decision logic. Applies spatial coherence filter
#          to each frame's clusters before tracking.
# INPUTS: config (dict from TEMPORAL_FILTER_CONFIG), stereo_system (StereoCameraSystem)
# OUTPUTS: Per-frame CBH estimates, full track histories, quality metrics
# KEYWORDS: orchestrator, multi_track, kalman_filter, cbH_estimation, lifecycle
class TemporalFilter:
    def __init__(self, config: dict, stereo_system=None):
        self.cfg = config
        self.stereo_system = stereo_system
        self.tracks: list[CloudTrack] = []
        self.dead_tracks: list[CloudTrack] = []
        self.next_track_id = 0
        self.frame_count = 0
        self.last_timestamp: Optional[float] = None
        self.spatial_reject_count = 0
        self.total_clusters_seen = 0
        self.image_width = config.get("image_width", 1920)
        self.mode = config.get("mode", "multi_track")

        # Simple 1D Kalman state
        self._kf_x: Optional[float] = None  # filtered height
        self._kf_P: float = 10000.0          # state uncertainty

        # TQI (Trust Quality Index) state
        self._tqi_history: list[float] = []
        self._recent_heights: list[float] = []
        self._recent_innovations: list[float] = []
        self._recent_sizes: list[int] = []
        self._recent_spatial_pass: list[bool] = []
        self._tqi_window: int = self.cfg.get("tqi_window", 5)

        self._H = np.zeros((4, 7))
        self._H[0, 0] = 1.0
        self._H[1, 2] = 1.0
        self._H[2, 3] = 1.0
        self._H[3, 6] = 1.0

        mode_label = "simple_1d" if self.mode == "simple" else "multi_track"
        logger.info("TemporalFilter initialized | mode=%s | spatial_filter=%s",
                     mode_label,
                     config.get("spatial_filter_enabled", True))

    def _compute_Q(self, dt: float) -> np.ndarray:
        if dt <= 0:
            dt = 1e-6
        pn = self.cfg.get("process_noise", {})
        sigma_h = pn.get("sigma_h_accel", 0.5)
        sigma_u = pn.get("sigma_u_accel", 2.0)
        sigma_v = pn.get("sigma_v_accel", 2.0)
        sigma_s = pn.get("sigma_s_walk", 5.0)

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        Q = np.zeros((7, 7))
        Q[0:2, 0:2] = sigma_h**2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q[2:4, 2:4] = sigma_u**2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q[4:6, 4:6] = sigma_v**2 * np.array([
            [dt4 / 4, dt3 / 2],
            [dt3 / 2, dt2],
        ])
        Q[6, 6] = sigma_s**2 * dt
        return Q

    def _compute_F(self, dt: float) -> np.ndarray:
        if dt <= 0:
            dt = 1e-6
        F = np.eye(7)
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        return F

    def _initial_state(self, meas: ClusterMeasurement) -> np.ndarray:
        return np.array([meas.h, 0.0, meas.u, meas.v, 0.0, 0.0,
                         float(meas.size)], dtype=np.float64)

    def _initial_covariance(self, meas: ClusterMeasurement) -> np.ndarray:
        P = np.eye(7) * 100.0
        P[0, 0] = max(meas.h_err ** 2, 100.0)
        P[2, 2] = 25.0
        P[3, 3] = 25.0
        P[6, 6] = max(float(meas.size), 1.0)
        P[1, 1] = 1.0
        P[4, 4] = 1.0
        P[5, 5] = 1.0
        return P

    def _build_R(self, meas: ClusterMeasurement) -> np.ndarray:
        scale = self.cfg.get("centroid_uncertainty_scale", 1.0)
        img_w = self.image_width
        s = max(meas.size, 1)
        sigma_u_sq = scale * (img_w / s) ** 2
        return np.diag([
            max(meas.h_err ** 2, 1.0),
            sigma_u_sq,
            sigma_u_sq,
            max(float(s), 1.0),
        ])

    # -----------------------------------------------------------------------
    # TQI — Unsupervised Trust Quality Index
    # -----------------------------------------------------------------------

    # PURPOSE: Compute the Trust Quality Index — an unsupervised per-frame
    #          confidence score that predicts whether the CBH estimate is
    #          trustworthy, without requiring ceilometer ground truth.
    #          Combines five independent signals: innovation consistency,
    #          cluster cardinality, spatial coherence, temporal stability,
    #          and layer persistence.
    # INPUTS: best (ClusterMeasurement), innovation_raw (float), spatial_pass (bool)
    # OUTPUTS: float [0,1] — higher = more trustworthy
    # KEYWORDS: trust, quality, unsupervised, confidence, TQI, runtime
    # PURPOSE: Compute the Reliability Index — an unsupervised per-frame
    #          confidence score that predicts CBH estimate quality using
    #          measurement quality + selection ambiguity signals (NOT Kalman
    #          model fit, which can converge to wrong layers).
    # INPUTS: best (selected ClusterMeasurement), all_clusters (list),
    #         innovation_raw (float), spatial_pass (bool)
    # OUTPUTS: float [0,1] — higher = more reliable
    # KEYWORDS: reliability, selection_quality, competition, dominance,
    #           unsupervised, confidence, RI
    def _compute_ri(self, best, all_clusters: list, innovation_raw: float,
                    spatial_pass: bool) -> float:
        """Compute unsupervised Reliability Index (selection-quality based)."""
        w = self.cfg.get("ri_weights", {
            "competition": 0.30, "dominance": 0.25, "precision": 0.20,
            "cardinality": 0.15, "spatial": 0.10,
        })

        # 1. q_competition: penalty when other large clusters exist at
        #    very different heights (multi-layer ambiguity)
        competing_size = 0.0
        for c in all_clusters:
            if c.cluster_id == best.cluster_id:
                continue
            if abs(c.h - best.h) > 500.0:  # different cloud layer
                competing_size += c.size
        if best.size > 0:
            q_competition = max(0.0, 1.0 - competing_size / best.size)
        else:
            q_competition = 0.0

        # 2. q_dominance: size ratio of selected vs 2nd-largest cluster
        #    (any height — unclear winner = unstable selection)
        sizes = sorted([c.size for c in all_clusters], reverse=True)
        if len(sizes) >= 2 and sizes[0] > 0:
            dominance_ratio = sizes[0] / max(sizes[1], 1)
            # ratio=1.0 (tie) → score=0.0; ratio=10.0 (clear) → score=1.0
            q_dominance = min(1.0, np.log10(max(dominance_ratio, 1.0)))
        else:
            q_dominance = 1.0

        # 3. q_precision: disparity tightness (disp_std / |disp_x|)        
        if abs(best.disp_x) > 1e-6:
            cv = best.disp_std / abs(best.disp_x)  # coefficient of variation
            q_precision = max(0.0, 1.0 - cv / 0.5)  # cv=0.5 → score=0
        else:
            q_precision = 0.0

        # 4. q_cardinality: cluster point count
        q_cardinality = min(1.0, best.size / 100.0)

        # 5. q_spatial: spatial filter
        q_spatial = 1.0 if spatial_pass else 0.0

        ri = (w["competition"] * q_competition + w["dominance"] * q_dominance +
              w["precision"] * q_precision + w["cardinality"] * q_cardinality +
              w["spatial"] * q_spatial)

        return float(np.clip(ri, 0.0, 1.0))

    def _compute_tqi(self, best, innovation_raw, spatial_pass):
        """Backward-compat wrapper for old TQI. Calls _compute_ri internally."""
        # Dummy clusters list (only best) for backward compat
        return self._compute_ri(best, [best], innovation_raw, spatial_pass)

    def get_last_tqi(self) -> Optional[float]:
        """Return the TQI from the most recently processed frame."""
        if self._tqi_history:
            return self._tqi_history[-1]
        return None

    def get_tqi_trust_label(self, tqi: Optional[float] = None) -> str:
        """Convert TQI score to human-readable trust label."""
        if tqi is None:
            tqi = self.get_last_tqi()
        if tqi is None:
            return "UNKNOWN"
        if tqi >= 0.70:
            return "TRUSTED"
        elif tqi >= 0.40:
            return "CONDITIONAL"
        return "UNTRUSTED"

    # -----------------------------------------------------------------------
    # Main per-frame processing
    # -----------------------------------------------------------------------

    def process_frame(self, clusters: list[ClusterMeasurement],
                      timestamp: float,
                      kp1: Optional[np.ndarray] = None,
                      labels: Optional[np.ndarray] = None,
                      shifts: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Process one frame of cluster measurements with optional spatial filtering.

        In 'simple' mode: picks the largest cluster and runs 1D Kalman smoothing.
        In 'multi_track' mode: runs full Hungarian-association multi-layer tracker.

        Returns: CBH estimate (meters) or None
        """
        if self.mode == "simple":
            return self._process_simple(clusters, timestamp, kp1, labels, shifts)

        return self._process_multi_track(clusters, timestamp, kp1, labels, shifts)

    # -----------------------------------------------------------------------
    # Simple mode: 1D Kalman on dominant cluster
    # -----------------------------------------------------------------------

    def _process_simple(self, clusters, timestamp, kp1, labels, shifts) -> Optional[float]:
        dt = (timestamp - self.last_timestamp) if self.last_timestamp is not None else 10.0
        dt = max(dt, 1e-6)
        self.last_timestamp = timestamp
        self.frame_count += 1
        self.total_clusters_seen += len(clusters)

        # Size filter
        min_size = self.cfg.get("min_cluster_size_for_tracking", 8)
        clusters = [c for c in clusters if c.size >= min_size]

        if not clusters:
            # Predict-only
            if self._kf_x is not None:
                Q = self.cfg.get("process_noise", {}).get("sigma_h_accel", 2.0)**2 * dt
                self._kf_P += Q
            return self._kf_x

        # Spatial filter if enabled
        if (self.cfg.get("spatial_filter_enabled", True)
                and kp1 is not None and labels is not None):
            valid_dict = validate_clusters_spatial(kp1, labels, shifts,
                                                   np.empty((0, 2)), self.cfg)
            n_before = len(clusters)
            clusters = [c for c in clusters if valid_dict.get(c.cluster_id, True)]
            self.spatial_reject_count += (n_before - len(clusters))

        if not clusters:
            if self._kf_x is not None:
                Q = self.cfg.get("process_noise", {}).get("sigma_h_accel", 2.0)**2 * dt
                self._kf_P += Q
            return self._kf_x

        # Pick the appropriate cluster for CBH selection
        strategy = self.cfg.get("cbh_strategy", "largest")
        if strategy == "lowest":
            clusters.sort(key=lambda c: c.h)
        elif strategy == "lowest_reliable":
            clusters.sort(key=lambda c: (c.h, -c.size))
        else:
            clusters.sort(key=lambda c: c.size, reverse=True)
        best = clusters[0]
        z = best.h
        innovation_raw = 0.0
        # Innovation-adaptive R: trusts measurements close to prediction,
        # downweights large jumps (likely measurement noise or layer switching)
        R_scale = self.cfg.get("measurement_noise_scale", 10.0)
        if self._kf_x is not None:
            innovation_raw = abs(z - self._kf_x)
            innovation_raw = max(innovation_raw, 10.0)
            R_adaptive = R_scale * (innovation_raw ** 2) * 0.01
        else:
            R_adaptive = R_scale * best.h_err ** 2
        R = float(np.clip(R_adaptive, 100.0, 1e6))

        if self._kf_x is None:
            self._kf_x = z
            self._kf_P = float(np.clip(R, 100.0, 100000.0))
        else:
            # Constant-acceleration process noise: Q = sigma_h_accel² * dt²
            sigma_h = self.cfg.get("process_noise", {}).get("sigma_h_accel", 3.0)
            Q_h = (sigma_h * dt) ** 2
            self._kf_P += Q_h
            K = self._kf_P / (self._kf_P + R)
            self._kf_x += K * (z - self._kf_x)
            self._kf_P = (1.0 - K) * self._kf_P

        logger.debug("Frame %d simple: z=%.1f filt=%.1f (cluster_size=%d)",
                     self.frame_count, z, self._kf_x, best.size)

        # --- Compute Reliability Index (RI) ---
        spatial_pass = valid_dict.get(best.cluster_id, True) if 'valid_dict' in dir() else True
        ri = self._compute_ri(best, clusters, innovation_raw, spatial_pass)
        self._tqi_history.append(ri)  # reuse history buffer for backward compat

        return self._kf_x

    # -----------------------------------------------------------------------
    # Multi-track mode (original)
    # -----------------------------------------------------------------------

    def _process_multi_track(self, clusters: list[ClusterMeasurement],
                             timestamp: float,
                             kp1: Optional[np.ndarray] = None,
                             labels: Optional[np.ndarray] = None,
                             shifts: Optional[np.ndarray] = None) -> Optional[float]:
        dt = (timestamp - self.last_timestamp) if self.last_timestamp is not None else 10.0
        dt = max(dt, 1e-6)
        self.last_timestamp = timestamp
        self.frame_count += 1

        n_raw = len(clusters)
        self.total_clusters_seen += n_raw

        # --- 1a. Minimum cluster size + per-frame top-N filter ---
        min_size = self.cfg.get("min_cluster_size_for_tracking", 15)
        max_per_frame = self.cfg.get("max_clusters_per_frame", 0)
        merge_tol = self.cfg.get("cluster_merge_height_tol_m", 0)

        before = len(clusters)

        if min_size > 0:
            clusters = [c for c in clusters if c.size >= min_size]

        # Merge clusters at similar heights (same cloud layer fragmentation fix)
        if merge_tol > 0 and len(clusters) > 1:
            clusters = _merge_clusters_by_height(clusters, merge_tol)

        if max_per_frame > 0 and len(clusters) > max_per_frame:
            clusters.sort(key=lambda c: c.size, reverse=True)
            clusters = clusters[:max_per_frame]

        dropped = before - len(clusters)
        if dropped > 0:
            self.spatial_reject_count += dropped
            logger.debug("Frame %d: size/merge/top-N dropped %d/%d clusters "
                         "(min_size=%d)",
                         self.frame_count, dropped, before, min_size)

        # --- 1b. Spatial coherence filter ---
        if (self.cfg.get("spatial_filter_enabled", True)
                and kp1 is not None and labels is not None and shifts is not None
                and len(clusters) > 0):
            # Build a minimal centers array (not really needed by the new API,
            # but passed for backward compatibility)
            centers_for_api = np.empty((0, 2))

            valid_dict = validate_clusters_spatial(kp1, labels, shifts,
                                                   centers_for_api, self.cfg)

            filtered_clusters = []
            for c in clusters:
                if valid_dict.get(c.cluster_id, True):
                    filtered_clusters.append(c)

            n_rejected = len(clusters) - len(filtered_clusters)
            self.spatial_reject_count += n_rejected
            if n_rejected > 0:
                logger.info("Frame %d: spatial filter rejected %d/%d clusters",
                            self.frame_count, n_rejected, len(clusters))
            clusters = filtered_clusters

        logger.debug("--- Frame %d | t=%.1f | dt=%.1fs | %d clusters (raw=%d) ---",
                     self.frame_count, timestamp, dt, len(clusters), n_raw)

        # 2. Predict all active tracks
        active = [t for t in self.tracks if t.status != TrackStatus.DEAD]
        for track in active:
            track.predict(dt, self._compute_Q(dt), self._compute_F(dt))
            track.last_timestamp = timestamp

        # 3-4. Associate clusters to tracks
        confirmed = [t for t in active if t.status == TrackStatus.CONFIRMED]
        prob = [t for t in active if t.status == TrackStatus.PROBATION]
        all_tracks_for_assoc = confirmed + prob

        # Set _dt for gate_physical
        self.cfg["_dt"] = dt

        assignments, unassigned_clusters, unassigned_tracks = associate_frame(
            all_tracks_for_assoc, clusters, self.cfg, self._H,
        )

        # 5. Update assigned tracks
        assigned_track_ids = set()
        for track_idx, cluster_idx in assignments:
            track = all_tracks_for_assoc[track_idx]
            cluster = clusters[cluster_idx]
            R = self._build_R(cluster)
            track.update(cluster, R, self._H, self.cfg)
            assigned_track_ids.add(track.track_id)

        # Mark missed tracks
        for track in unassigned_tracks:
            track.mark_miss(timestamp)
            max_miss = self.cfg.get("max_missed_frames", 5)
            if track.status == TrackStatus.CONFIRMED and track.miss_count >= max_miss:
                track.status = TrackStatus.DEAD
                self.dead_tracks.append(track)
                logger.info("Track %d died after %d consecutive misses",
                            track.track_id, track.miss_count)

        # 6. Initiate new tracks for unassigned clusters
        for idx in unassigned_clusters:
            cluster = clusters[idx]
            init_state = self._initial_state(cluster)
            init_cov = self._initial_covariance(cluster)
            track = CloudTrack(self.next_track_id, init_state, init_cov,
                              timestamp, cluster)
            self.next_track_id += 1
            self.tracks.append(track)
            logger.debug("New track %d initiated (PROBATION) | h=%.1f | size=%d",
                         track.track_id, cluster.h, cluster.size)

        # 7. Manage probation tracks
        for track in self.tracks:
            if track.status != TrackStatus.PROBATION:
                continue
            M = self.cfg.get("probation_M", 2)
            N = self.cfg.get("probation_N", 3)
            if track.probation_frames >= N:
                if track.probation_hits >= M:
                    track.status = TrackStatus.CONFIRMED
                    logger.info("Track %d CONFIRMED | hits=%d/%d | quality=%.3f",
                                track.track_id, track.probation_hits,
                                track.probation_frames, track.quality_score(self.cfg))
                else:
                    track.status = TrackStatus.DEAD
                    self.dead_tracks.append(track)
                    logger.debug("Track %d failed probation | hits=%d/%d",
                                 track.track_id, track.probation_hits,
                                 track.probation_frames)

        # 8. Select CBH
        cbh = self._select_cbh(timestamp)
        logger.debug("Frame CBH: %s", f"{cbh:.1f}" if cbh is not None else "None")
        return cbh

    def _select_cbh(self, timestamp: float) -> Optional[float]:
        min_points = self.cfg.get("min_points_for_cbh", 30)
        quality_min = self.cfg.get("quality_threshold", 0.3)

        candidates = []
        # Prefer confirmed tracks, but allow probation as fallback
        for track in self.tracks:
            if track.status == TrackStatus.DEAD:
                continue
            size = float(track.x[6])
            if size < min_points:
                continue
            q = track.quality_score(self.cfg)
            if track.status == TrackStatus.CONFIRMED and q >= quality_min:
                candidates.append((track.x[0], q, track.track_id, 0))
            elif track.status == TrackStatus.PROBATION and q >= quality_min * 0.6:
                candidates.append((track.x[0], q, track.track_id, 1))

        if not candidates:
            return None

        # Sort by: status_preference (confirmed first), then height
        candidates.sort(key=lambda x: (x[3], x[0]))
        best_h, best_q, best_id, pref = candidates[0]
        status_str = "confirmed" if pref == 0 else "probation"
        logger.debug("CBH selected | h=%.1f | quality=%.3f | track_id=%d | %s | candidates=%d",
                     best_h, best_q, best_id, status_str, len(candidates))
        return best_h

    # -----------------------------------------------------------------------
    # RTS Smoother
    # -----------------------------------------------------------------------

    def rts_smooth(self):
        if not self.cfg.get("enable_rts_smoother", True):
            logger.info("RTS smoother disabled")
            return

        for track in self.tracks:
            if track.status not in (TrackStatus.CONFIRMED, TrackStatus.PROBATION):
                continue
            history = track.history
            if len(history) < 3:
                continue

            xs = [np.array([e["h"], e["h_dot"], e["u"], e["v"],
                           e["u_dot"], e["v_dot"], e["size"]]) for e in history]
            xs_smoothed = [xs[-1].copy()]

            for k in range(len(xs) - 2, -1, -1):
                t_k = history[k]["timestamp"]
                t_k1 = history[k + 1]["timestamp"]
                dt = max(t_k1 - t_k, 1e-6)
                F = self._compute_F(dt)
                Q = self._compute_Q(dt)

                P_k = np.eye(7) * 1.0
                P_pred = F @ P_k @ F.T + Q
                try:
                    G = P_k @ F.T @ np.linalg.inv(P_pred)
                except np.linalg.LinAlgError:
                    G = np.zeros((7, 7))
                xs_smoothed.append(xs[k] + G @ (xs_smoothed[-1] - F @ xs[k]))

            xs_smoothed.reverse()

            for i, x_s in enumerate(xs_smoothed):
                history[i]["h"] = float(x_s[0])
                history[i]["h_dot"] = float(x_s[1])
                history[i]["u"] = float(x_s[2])
                history[i]["v"] = float(x_s[3])
                history[i]["u_dot"] = float(x_s[4])
                history[i]["v_dot"] = float(x_s[5])
                history[i]["size"] = float(x_s[6])
                history[i]["smoothed"] = True

        logger.info("RTS smoother applied to %d tracks", len(self.tracks))

    # -----------------------------------------------------------------------
    # Results export
    # -----------------------------------------------------------------------

    def get_cbh_series(self) -> list[dict]:
        frames = {}
        for track in self.tracks:
            for entry in track.history:
                ts = entry["timestamp"]
                key = round(ts, 1)
                if key not in frames:
                    frames[key] = {
                        "timestamp": ts,
                        "cbh": None,
                        "tracks": [],
                    }
        for track in self.tracks:
            for entry in track.history:
                if entry["status"] != "CONFIRMED":
                    continue
                ts = entry["timestamp"]
                key = round(ts, 1)
                if key in frames:
                    frames[key]["tracks"].append({
                        "track_id": track.track_id,
                        "h": entry["h"],
                        "size": entry["size"],
                        "quality": track.quality_score(self.cfg),
                    })

        sorted_keys = sorted(frames.keys())
        result = []
        for key in sorted_keys:
            f = frames[key]
            if f["tracks"]:
                f["tracks"].sort(key=lambda x: x["h"])
                best = f["tracks"][0]
                f["cbh"] = best["h"]
            result.append(f)
        return result

    def get_track_summaries(self) -> list[dict]:
        summaries = []
        for track in self.tracks:
            if track.status == TrackStatus.DEAD:
                continue
            summaries.append({
                "track_id": track.track_id,
                "status": track.status.name,
                "frames": len(track.history),
                "lifetime_s": track.lifetime_seconds(),
                "quality": track.quality_score(self.cfg),
                "avg_h": float(np.mean([e["h"] for e in track.history])),
                "final_h": float(track.x[0]),
                "final_size": float(track.x[6]),
                "outlier_rate": (track.outlier_count /
                                 max(len(track.history), 1)),
            })
        return summaries

    def print_report(self):
        summaries = self.get_track_summaries()
        confirmed = [s for s in summaries if s["status"] == "CONFIRMED"]
        prob = [s for s in summaries if s["status"] == "PROBATION"]

        print(f"\n{'='*70}")
        print(f"  TEMPORAL FILTER REPORT")
        print(f"{'='*70}")
        print(f"  Frames processed:          {self.frame_count}")
        print(f"  Total clusters seen:       {self.total_clusters_seen}")
        print(f"  Clusters rejected (spatial): {self.spatial_reject_count}")
        print(f"  Confirmed tracks:           {len(confirmed)}")
        print(f"  Probation tracks:           {len(prob)}")
        print(f"  Dead tracks:                {len(self.dead_tracks)}")
        print(f"  Total tracks created:       {len(self.tracks)}")
        print(f"{'='*70}")

        if confirmed:
            print(f"\n  {'ID':>4} {'Quality':>8} {'Frames':>7} {'Life(s)':>8} "
                  f"{'Avg H(m)':>9} {'Final H(m)':>10} {'Outlier%':>9}")
            print(f"  {'-'*62}")
            for s in sorted(confirmed, key=lambda x: x["quality"], reverse=True):
                print(f"  {s['track_id']:4d} {s['quality']:8.3f} {s['frames']:7d} "
                      f"{s['lifetime_s']:8.1f} {s['avg_h']:9.1f} {s['final_h']:10.1f} "
                      f"{s['outlier_rate']:9.1%}")
