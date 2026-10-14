# PURPOSE: The temporal filter itself — track lifecycle management, data
#          association, Kalman predict/update cycles, innovation outlier
#          rejection and multi-layer cloud-base selection. Its inputs, tracks,
#          cluster merging and Reliability Index live in sibling modules.
# INPUTS: tracker settings (see .settings), a stereo system, per-frame clusters.
# OUTPUTS: a filtered cloud-base height per frame, track histories and metrics.
# KEYWORDS: kalman, temporal_filter, tracking, cloud_base_height, lifecycle
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

from .association import associate_frame, compute_mahalanobis_sq, gate_physical
from .measurement import ClusterMeasurement
from .merge import merge_clusters_by_height
from .track import CloudTrack, TrackStatus
from .reliability import compute_ri, trust_label
from .report import cbh_series, format_report, track_summaries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TemporalFilter
# ---------------------------------------------------------------------------

# PURPOSE: Orchestrate the full temporal filtering pipeline across all frames.
#          Manages track lifecycle (probation/confirmed/dead), data association,
#          Kalman predict/update cycles, innovation outlier rejection, and
#          multi-layer CBH decision logic.
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
        self._tqi_window: int = self.cfg.get("tqi_window", 5)

        self._H = np.zeros((4, 7))
        self._H[0, 0] = 1.0
        self._H[1, 2] = 1.0
        self._H[2, 3] = 1.0
        self._H[3, 6] = 1.0

        mode_label = "simple_1d" if self.mode == "simple" else "multi_track"
        logger.info("TemporalFilter initialized | mode=%s", mode_label)

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
    def _compute_tqi(self, best, innovation_raw, spatial_pass):
        """Backward-compat wrapper for old TQI. Calls _compute_ri internally."""
        # Dummy clusters list (only best) for backward compat
        return self._compute_ri(best, [best], innovation_raw, spatial_pass)

    def get_last_tqi(self) -> Optional[float]:
        """Return the TQI from the most recently processed frame."""
        if self._tqi_history:
            return self._tqi_history[-1]
        return None

    def process_frame(self, clusters: list[ClusterMeasurement],
                      timestamp: float,
                      kp1: Optional[np.ndarray] = None,
                      labels: Optional[np.ndarray] = None,
                      shifts: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Process one frame of cluster measurements.

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
        # spatial_pass is retained as a constant: the spatial-coherence filter was
        # removed after it was measured to reject 37-43 % of clusters while changing
        # the filtered series by 0.3 % and the raw series not at all. The parameter
        # stays so the RI definition is unchanged and past numbers remain comparable.
        ri = self._compute_ri(best, clusters, innovation_raw, True)
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
            clusters = merge_clusters_by_height(clusters, merge_tol)

        if max_per_frame > 0 and len(clusters) > max_per_frame:
            clusters.sort(key=lambda c: c.size, reverse=True)
            clusters = clusters[:max_per_frame]

        dropped = before - len(clusters)
        if dropped > 0:
            logger.debug("Frame %d: size/merge/top-N dropped %d/%d clusters "
                         "(min_size=%d)",
                         self.frame_count, dropped, before, min_size)

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

    # PURPOSE: Compute the Reliability Index for the current frame. Delegates to
    #          clouds.temporal.reliability so the scoring rule lives in one place.
    # INPUTS: best (ClusterMeasurement), all_clusters (list), innovation_raw
    #         (float), spatial_pass (bool).
    # OUTPUTS: float in [0,1].
    # KEYWORDS: reliability, delegate
    def _compute_ri(self, best, all_clusters: list, innovation_raw: float,
                    spatial_pass: bool) -> float:
        return compute_ri(self.cfg, best, all_clusters, innovation_raw, spatial_pass)

    # PURPOSE: Backward-compatible single-candidate form of the Reliability Index.
    # INPUTS: best, innovation_raw, spatial_pass.
    # OUTPUTS: float in [0,1].
    # KEYWORDS: reliability, compatibility
    def get_tqi_trust_label(self, tqi: Optional[float] = None) -> str:
        score = tqi if tqi is not None else self.get_last_tqi()
        return trust_label(score)

    # PURPOSE: Return the per-frame cloud-base series. Delegates to
    #          clouds.temporal.report.
    # INPUTS: none.
    # OUTPUTS: list of per-frame dicts.
    # KEYWORDS: report, delegate
    def get_cbh_series(self) -> list:
        return cbh_series(self.tracks, self.cfg)

    # PURPOSE: Summarise every live track. Delegates to clouds.temporal.report.
    # INPUTS: none.
    # OUTPUTS: list of per-track dicts.
    # KEYWORDS: report, delegate
    def get_track_summaries(self) -> list:
        return track_summaries(self.tracks, self.cfg)

    # PURPOSE: Print the formatted tracking report. Delegates to
    #          clouds.temporal.report.
    # INPUTS: none.
    # OUTPUTS: None (writes to stdout).
    # KEYWORDS: report, delegate
    def print_report(self) -> None:
        print(format_report(self))
