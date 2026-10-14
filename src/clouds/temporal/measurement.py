# PURPOSE: The per-frame observation the tracker consumes — a cloud cluster's
#          height, its uncertainty, its image-space position and size, and the
#          disparity statistics behind them.
# INPUTS: one HDBSCAN cluster summarised by the stereo pipeline.
# OUTPUTS: an immutable measurement record.
# KEYWORDS: measurement, cluster, cloud_base_height, tracker_input
from __future__ import annotations

from dataclasses import dataclass


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
