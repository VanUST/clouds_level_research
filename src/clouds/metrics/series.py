# PURPOSE: Scalar descriptors of a cloud-base-height time series — smoothness and
#          temporal coherence. These are the quantities a Kalman filter is
#          supposed to improve, expressed separately from any comparison against
#          ground truth, so "did the filter smooth the series" and "did it agree
#          with the ceilometer" stay distinguishable.
# INPUTS: sequences of heights (and, where relevant, their timestamps).
# OUTPUTS: floats; NaN (never an exception) when the series is too short or has
#          zero variance, so a degenerate window cannot abort a whole run.
# KEYWORDS: metrics, smoothness, autocorrelation, jump, time_series
from __future__ import annotations

import numpy as np


# PURPOSE: Measure frame-to-frame height instability as the mean and median
#          absolute difference between consecutive samples.
# INPUTS: heights (sequence[float]) — time-ordered, gaps already removed.
# OUTPUTS: dict {mean_dh, median_dh, n}. With fewer than two samples all values
#          are NaN/0 rather than raising.
# KEYWORDS: metric, jump, frame_to_frame, smoothness
def frame_to_frame_jump(heights) -> dict:
    h = np.asarray(heights, dtype=float)
    if h.size < 2:
        return {"mean_dh": float("nan"), "median_dh": float("nan"), "n": int(h.size)}
    d = np.abs(np.diff(h))
    return {"mean_dh": float(np.mean(d)), "median_dh": float(np.median(d)), "n": int(h.size)}


# PURPOSE: Compute the lag-1 autocorrelation of a height series — the correlation
#          of the series with itself one acquisition step later. A smooth,
#          physically coherent trajectory has a value near 1; an incoherent
#          frame-by-frame series has one near 0.
# INPUTS: heights (sequence[float]).
# OUTPUTS: float Pearson r. NaN when the series has fewer than three samples or
#          either half has zero variance.
# KEYWORDS: metric, autocorrelation, smoothness, lag1
def lag1_autocorrelation(heights) -> float:
    h = np.asarray(heights, dtype=float)
    if h.size < 3:
        return float("nan")
    a, b = h[:-1], h[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# PURPOSE: Report the improvement factors a filter achieved, guarding against
#          division by zero and NaN so a degenerate series yields None instead of
#          a misleading number.
# INPUTS: raw_heights, filtered_heights (sequences).
# OUTPUTS: dict {jump_factor, autocorr_factor, raw, filtered} where the factors
#          are None when they are undefined.
# KEYWORDS: metric, improvement, factor, filter
def smoothing_improvement(raw_heights, filtered_heights) -> dict:
    raw = frame_to_frame_jump(raw_heights)
    filt = frame_to_frame_jump(filtered_heights)
    raw_ac = lag1_autocorrelation(raw_heights)
    filt_ac = lag1_autocorrelation(filtered_heights)
    jump_factor = (raw["mean_dh"] / filt["mean_dh"]
                   if filt["mean_dh"] and np.isfinite(filt["mean_dh"]) and filt["mean_dh"] != 0
                   else None)
    autocorr_factor = (filt_ac / raw_ac
                       if raw_ac is not None and np.isfinite(raw_ac) and raw_ac != 0
                       else None)
    return {
        "raw": raw, "filtered": filt,
        "raw_autocorr": raw_ac, "filtered_autocorr": filt_ac,
        "jump_factor": jump_factor, "autocorr_factor": autocorr_factor,
    }
