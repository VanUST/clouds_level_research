# PURPOSE: The unsupervised Reliability Index — a per-frame confidence that the
#          selected cloud-base estimate is trustworthy, computed without any
#          ground truth from measurement quality and selection ambiguity. It is a
#          reporting signal, never an input to cluster selection.
# INPUTS: the tracker settings, the selected measurement, all candidate clusters,
#         the raw innovation and whether the spatial filter passed.
# OUTPUTS: a score in [0,1] and a human-readable trust label.
# KEYWORDS: reliability, confidence, unsupervised, selection_quality, trust
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_ri(cfg: dict, best, all_clusters: list, innovation_raw: float,
               spatial_pass: bool) -> float:
    """Compute the unsupervised Reliability Index (selection-quality based)."""
    w = cfg.get("ri_weights", {
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


def trust_label(score: Optional[float]) -> str:
    """Map a Reliability Index onto a human-readable trust label.
    INPUTS: score (float|None).
    OUTPUTS: "UNKNOWN" when score is None, else TRUSTED (>=0.70),
             CONDITIONAL (>=0.40) or UNTRUSTED.
    KEYWORDS: reliability, trust, label
    """
    if score is None:
        return "UNKNOWN"
    if score >= 0.70:
        return "TRUSTED"
    if score >= 0.40:
        return "CONDITIONAL"
    return "UNTRUSTED"

    # -----------------------------------------------------------------------
    # Main per-frame processing
    # -----------------------------------------------------------------------
