# PURPOSE: A tracked cloud layer — its Kalman state and covariance, lifecycle
#          status, quality score and measurement history.
# INPUTS: an initial state, covariance, timestamp and measurement.
# OUTPUTS: a track exposing predict/update/lifecycle operations.
# KEYWORDS: track, kalman_state, covariance, lifecycle, quality
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

from .measurement import ClusterMeasurement

logger = logging.getLogger(__name__)


class TrackStatus(Enum):
    PROBATION = auto()
    CONFIRMED = auto()
    DEAD = auto()


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
