# PURPOSE: Align a stereo-derived cloud-base-height series with a ceilometer
#          series in time and score the agreement. The two instruments sit at
#          different locations and sample different clocks, and wind moves the
#          cloud field between them, so agreement must be measured after
#          searching a time offset — and the search must report how many pairs
#          supported each candidate offset, otherwise a high correlation resting
#          on a handful of overlapping samples looks like a result.
# INPUTS: two time series as (datetime, value) pairs; window length, hop, shift
#         search range and pairing tolerance.
# OUTPUTS: per-window results, summary statistics, and a shift-scan curve.
# KEYWORDS: alignment, time_shift, correlation, windowed, cross_correlation
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np


# PURPOSE: Convert a camera filename timestamp to the ceilometer clock.
# INPUTS: ts (datetime) parsed from an image filename; offset_h (float) — hours
#         to add (3.0 for the 2025 archives, whose filenames are UTC while the
#         DOL-2 logs are MSK).
# OUTPUTS: datetime in the ceilometer clock.
# KEYWORDS: timezone, utc, msk, alignment
def to_ceilometer_clock(ts: dt.datetime, offset_h: float) -> dt.datetime:
    return ts + dt.timedelta(hours=float(offset_h))


# PURPOSE: Pair each algorithm sample with the nearest-in-time ceilometer sample
#          after applying a candidate clock shift, and score the agreement.
# INPUTS: algo_ts, algo_h, ceilo_ts, ceilo_h (np.ndarray, epoch seconds /
#         heights); shift_s (float) added to algo_ts; tol_s (float) maximum
#         accepted separation.
# OUTPUTS: (r, n_pairs). r is NaN when fewer than 3 pairs survive or either side
#          has zero variance; n_pairs is always returned so the caller can judge
#          how much evidence supports r.
# KEYWORDS: pairing, nearest, correlation, gating
def paired_r(algo_ts, algo_h, ceilo_ts, ceilo_h, shift_s: float, tol_s: float):
    algo_ts = np.asarray(algo_ts, dtype=float)
    algo_h = np.asarray(algo_h, dtype=float)
    ceilo_ts = np.asarray(ceilo_ts, dtype=float)
    ceilo_h = np.asarray(ceilo_h, dtype=float)
    if algo_ts.size == 0 or ceilo_ts.size == 0:
        return float("nan"), 0
    shifted = algo_ts + shift_s
    idx = np.searchsorted(ceilo_ts, shifted)
    idx = np.clip(idx, 0, len(ceilo_ts) - 1)
    left = np.clip(idx - 1, 0, len(ceilo_ts) - 1)
    choose_left = np.abs(ceilo_ts[left] - shifted) < np.abs(ceilo_ts[idx] - shifted)
    idx = np.where(choose_left, left, idx)
    valid = np.abs(ceilo_ts[idx] - shifted) <= tol_s
    n = int(np.sum(valid))
    if n < 3:
        return float("nan"), n
    ch, ah = ceilo_h[idx[valid]], algo_h[valid]
    if np.std(ch) == 0 or np.std(ah) == 0:
        return float("nan"), n
    return float(np.corrcoef(ch, ah)[0, 1]), n


# PURPOSE: Split two series into fixed windows and, inside each window, find the
#          time shift that maximises correlation — the "per-window optimal shift"
#          procedure used to compare instruments decoupled by wind advection.
# INPUTS: times (list[datetime]), heights (list[float]), ceilo times/heights,
#         window_min (int), shift_range_s (float), tol_s (float), step_min (int
#         or None; defaults to the window length, i.e. non-overlapping windows).
# OUTPUTS: list of dicts {start, n_algo, n_ceilo, best_shift, r}. A window is
#          emitted only when both series have at least 5 samples inside it; r may
#          still be None when no shift produced at least 3 usable pairs.
# KEYWORDS: windowed, correlation, dynamic_shift, per_window
def windowed_correlation(times, heights, ceilo_times, ceilo_heights,
                         window_min: int, shift_range_s: float, tol_s: float,
                         step_min: Optional[int] = None) -> list:
    if step_min is None:
        step_min = window_min
    a_ts = np.array([t.timestamp() for t in times], dtype=float)
    a_h = np.array(heights, dtype=float)
    c_ts = np.array([t.timestamp() for t in ceilo_times], dtype=float)
    c_h = np.array(ceilo_heights, dtype=float)
    if a_ts.size == 0 or c_ts.size == 0:
        return []

    start = dt.datetime.fromtimestamp(min(a_ts[0], c_ts[0]))
    end = dt.datetime.fromtimestamp(max(a_ts[-1], c_ts[-1]))
    delta = dt.timedelta(minutes=window_min)
    step = dt.timedelta(minutes=step_min)

    results = []
    current = start
    while current < end:
        nxt = current + delta
        lo, hi = current.timestamp(), nxt.timestamp()
        a_mask = (a_ts >= lo) & (a_ts < hi)
        c_mask = (c_ts >= lo) & (c_ts < hi)
        if int(np.sum(a_mask)) >= 5 and int(np.sum(c_mask)) >= 5:
            best_r, best_shift = float("nan"), 0
            for shift in range(-int(shift_range_s), int(shift_range_s) + 1, 5):
                r, _n = paired_r(a_ts[a_mask], a_h[a_mask], c_ts[c_mask], c_h[c_mask],
                                 shift, tol_s)
                if not np.isnan(r) and (np.isnan(best_r) or r > best_r):
                    best_r, best_shift = r, shift
            results.append({
                "start": current.isoformat(),
                "n_algo": int(np.sum(a_mask)),
                "n_ceilo": int(np.sum(c_mask)),
                "best_shift": int(best_shift),
                "r": None if np.isnan(best_r) else float(best_r),
            })
        current += step
    return results


# PURPOSE: Summarise per-window correlations into the three numbers this kind of
#          comparison is usually quoted with: mean r, fraction of windows with
#          r > 0.5, and the best single window.
# INPUTS: windows (list[dict] as returned by windowed_correlation).
# OUTPUTS: dict {n_windows, n_scored, mean_r, frac_gt_half, max_r}; NaN/None when
#          no window produced a finite correlation.
# KEYWORDS: summary, mean_r, fraction, max_r
def summarise_windows(windows: list) -> dict:
    rs = [w["r"] for w in windows if w.get("r") is not None]
    if not rs:
        return {"n_windows": len(windows), "n_scored": 0, "mean_r": float("nan"),
                "frac_gt_half": float("nan"), "max_r": float("nan")}
    arr = np.array(rs, dtype=float)
    return {
        "n_windows": len(windows),
        "n_scored": int(arr.size),
        "mean_r": float(np.mean(arr)),
        "frac_gt_half": float(np.mean(arr > 0.5)),
        "max_r": float(np.max(arr)),
    }


# PURPOSE: Scan a wide range of global time offsets and report the best one, so a
#          whole-hour clock error cannot hide behind a narrow shift search. Also
#          reports the overlap size at each candidate, because a correlation found
#          on a small overlapping tail is not comparable to one found on hours of
#          data.
# INPUTS: times, heights, ceilo times/heights, tol_s, span_h (int), step_s (int),
#         min_overlap (int) — candidates with fewer pairs are not considered.
# OUTPUTS: dict {best_shift_s, best_shift_h, best_r, best_n, span_h, step_s,
#          min_overlap, r_at_0h, curve}. curve is a list of
#          [shift_s, r, n] triples restricted to shifts meeting min_overlap.
# KEYWORDS: verification, timezone, cross_correlation, alignment
def scan_global_shift(times, heights, ceilo_times, ceilo_heights, tol_s: float = 10.0,
                      span_h: int = 4, step_s: int = 60, min_overlap: int = 30) -> dict:
    a_ts = np.array([t.timestamp() for t in times], dtype=float)
    a_h = np.array(heights, dtype=float)
    c_ts = np.array([t.timestamp() for t in ceilo_times], dtype=float)
    c_h = np.array(ceilo_heights, dtype=float)
    if a_ts.size < 3 or c_ts.size < 3:
        return {"error": "insufficient data"}

    curve = []
    best = (-2.0, 0, 0)
    for shift in range(-span_h * 3600, span_h * 3600 + 1, step_s):
        r, n = paired_r(a_ts, a_h, c_ts, c_h, shift, tol_s)
        if np.isnan(r) or n < min_overlap:
            continue
        curve.append([int(shift), round(float(r), 4), int(n)])
        if r > best[0]:
            best = (float(r), int(shift), int(n))

    def r_at(target):
        if not curve:
            return None
        return min(curve, key=lambda x: abs(x[0] - target))[1]

    return {
        "best_shift_s": best[1], "best_shift_h": round(best[1] / 3600.0, 3),
        "best_r": best[0], "best_n": best[2],
        "span_h": span_h, "step_s": step_s, "min_overlap": min_overlap,
        "r_at_0h": r_at(0), "r_at_plus3h": r_at(3 * 3600), "r_at_minus3h": r_at(-3 * 3600),
        "curve": curve,
    }
