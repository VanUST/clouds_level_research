# PURPOSE: Shared loading and windowing helpers for the temporal-alignment
#          investigation, so the five question scripts all consume the identical
#          stored series and the identical window grid the pipeline used. The
#          series is read from a run directory's series.json; nothing here
#          estimates a height, it only re-shapes already-stored numbers.
# INPUTS: a run directory under /home/iustimov/clouds_data/logs.
# OUTPUTS: numpy arrays of epoch seconds and heights for each series, plus the
#          window grid generator.
# KEYWORDS: alignment, series, windowing, reproducible, diagnostics
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np

# The roll-corrected run is the preferred one (see the task statement): the
# roll=0 companion run is 20261013T113738Z_evaluate_clouds_2025-05-23_7d983a.
RUN_ROLL_CORRECTED = ("/home/iustimov/clouds_data/logs/"
                      "20261013T112649Z_evaluate_clouds_2025-05-23_3565c5")
RUN_ROLL_ZERO = ("/home/iustimov/clouds_data/logs/"
                 "20261013T113738Z_evaluate_clouds_2025-05-23_7d983a")
IMAGES_DIR = "/home/iustimov/clouds_data/images"


# PURPOSE: Convert a series of ISO timestamp strings to epoch seconds.
# INPUTS: stamps (list[str]).
# OUTPUTS: np.ndarray float64 of epoch seconds, same order.
# KEYWORDS: time, epoch, parsing
def stamps_to_epoch(stamps) -> np.ndarray:
    return np.array([dt.datetime.fromisoformat(s).timestamp() for s in stamps],
                    dtype=float)


# PURPOSE: Load one run's stored series (algo times/heights + ceilometer pairs).
# INPUTS: run_dir (str) — a directory containing series.json.
# OUTPUTS: dict with keys a_ts, raw_h, filt_h, c_ts, c_h (all np.ndarray);
#          heights are float64 and contain no NaN in these runs.
# KEYWORDS: series, io, loader
def load_series(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "series.json")) as fh:
        d = json.load(fh)
    c = np.array(d["ceilometer"], dtype=object)
    return {
        "a_ts": stamps_to_epoch(d["times_ceilometer_clock"]),
        "raw_h": np.asarray(d["raw_h"], dtype=float),
        "filt_h": np.asarray(d["filtered_h"], dtype=float),
        "c_ts": np.asarray([float(dt.datetime.fromisoformat(t).timestamp())
                            for t, _ in c], dtype=float),
        "c_h": np.asarray([float(h) for _, h in c], dtype=float),
    }


# PURPOSE: Reproduce the pipeline's window grid exactly — windows of
#          window_min minutes starting at the earliest sample of either series,
#          advanced by hop_min minutes, keeping only windows where both series
#          have at least 5 samples.
# INPUTS: a_ts, c_ts (epoch seconds); window_min, hop_min (ints).
# OUTPUTS: list of dicts {start_s, start_dt, a_idx, c_idx, n_algo, n_ceilo} —
#          time-ordered, matching clouds.metrics.alignment.windowed_correlation.
# KEYWORDS: windowing, grid, reproduction
def window_grid(a_ts, c_ts, window_min: int = 30, hop_min: int = 15) -> list:
    start = min(a_ts[0], c_ts[0])
    end = max(a_ts[-1], c_ts[-1])
    delta = window_min * 60.0
    step = hop_min * 60.0
    out = []
    current = start
    while current < end:
        nxt = current + delta
        a_mask = (a_ts >= current) & (a_ts < nxt)
        c_mask = (c_ts >= current) & (c_ts < nxt)
        if int(a_mask.sum()) >= 5 and int(c_mask.sum()) >= 5:
            out.append({
                "start_s": float(current),
                "start_dt": dt.datetime.fromtimestamp(current),
                "a_ts": a_ts[a_mask], "a_h": None,  # heights filled by caller
                "a_idx": np.where(a_mask)[0], "c_idx": np.where(c_mask)[0],
                "n_algo": int(a_mask.sum()), "n_ceilo": int(c_mask.sum()),
            })
        current += step
    return out
