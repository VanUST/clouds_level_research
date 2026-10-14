# PURPOSE: Feasibility probe for the research workplan. It asks a single question:
#          the estimator's series is reported as significantly ANTI-correlated with
#          the ceilometer, and a correct estimator should at worst be uncorrelated.
#          Before planning research around that result, establish WHAT the estimator
#          reports, WHICH of the several anti-correlation claims survive a gap-safe
#          protocol, and WHERE in the image the reported height comes from.
#
#          Three measurements, all on stored artifacts, no pipeline change:
#            1. Reproduce the evaluation's two correlation claims (level and first
#               differences) and test whether each survives (a) dropping the
#               pseudo-differences that span time gaps, and (b) a moving-block
#               bootstrap that respects autocorrelation.
#            2. Stratify the per-frame height distribution by the ceilometer's own
#               H1 regime (lower < 2750 m vs upper > 3500 m). If the stereo sees the
#               cloud, the two distributions must differ; the SIGN of that
#               difference is the anti-correlation mechanism in miniature.
#            3. Inspect the shipped "lowest cluster" selection: its image position
#               (u, v), radius and stationarity. A constant height at a constant
#               image position is a fixed scene target, not a cloud.
#
#          The ceilometer enters only as a stratification variable for comparisons
#          that already exist; it never selects, weights or filters an estimate
#          (guardrail G1). No file outside experiments/planning/ is written.
# INPUTS: the roll-corrected run series.json and the baseline cluster dump under
#         /home/iustimov/clouds_data/logs/, and the DOL-2 log Zve_Sci_250523.txt.
# OUTPUTS: experiments/planning/anti_correlation_probe.json with every number
#          printed here; console log of the same. Missing inputs raise FileNotFoundError
#          rather than silently skipping a section.
# KEYWORDS: feasibility, anti_correlation, effective_sample_size, ceilometer_regime,
#           cluster_selection, workplan
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "report"))

from clouds.sources import ceilometer as C  # noqa: E402
from metric_power import r_of, regrid  # noqa: E402

# The roll-corrected full-window run: series.json only, no cluster dump.
ROLL_RUN = (
    "/home/iustimov/clouds_data/logs/"
    "20261013T112649Z_evaluate_clouds_2025-05-23_3565c5"
)
# The shipped-configuration full-window run that also dumped per-cluster records.
CLUSTER_RUN = (
    "/home/iustimov/clouds_data/logs/"
    "20261013T092414Z_evaluate_clouds_2025-05-23_63dbc1"
)
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"

# Regime split used throughout, taken from the reference's own reported structure
# (docs/plan/evaluation.md §3: 40 % below 2750 m, 53 % above 3500 m).
REGIME_LOW_MAX_M = 2750.0
REGIME_HIGH_MIN_M = 3500.0

# A cluster must hold at least this many matches to be a candidate, matching the
# configured `temporal.min_cluster_size_for_tracking`.
MIN_CLUSTER_SIZE = 8


# PURPOSE: Fraction of the estimator/ceilometer disagreement that is a slow level
#          effect rather than frame-to-frame tracking.
# INPUTS: none (reads the two runs named above).
# OUTPUTS: dict with the level correlation and its block-bootstrap interval, the
#          two difference-series variants (gap-crossing and strictly contiguous)
#          and their intervals. All correlations are Pearson r on the stated n.
# KEYWORDS: correlation, block_bootstrap, decorrelation, reproducibility
def correlation_audit() -> dict:
    series = json.load(open(os.path.join(ROLL_RUN, "series.json")))
    at = [dt.datetime.fromisoformat(x) for x in series["times_ceilometer_clock"]]
    raw = np.array([np.nan if v is None else v for v in series["raw_h"]], float)

    samples = C.parse_ceilometer_log(LOG)
    lo, hi = dt.time(12, 0, 0), dt.time(17, 0, 0)
    ce = [s for s in samples
          if s.time.date() == dt.date(2025, 5, 23) and lo <= s.time.time() <= hi
          and s.h1 is not None]
    ct = [s.time for s in ce]
    ch = np.array([s.h1 for s in ce], float)

    ta = np.array([x.timestamp() for x in at], float)
    order = np.argsort(ta)
    ta, raw = ta[order], raw[order]
    tc = np.array([x.timestamp() for x in ct], float)
    oc = np.argsort(tc)
    tc, ch = tc[oc], ch[oc]
    t0 = max(ta.min(), tc.min())
    t1 = min(ta.max(), tc.max())

    _tg, a, b = regrid(ta, raw, tc, ch, 30.0, t0, t1)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    r_level, n_level = r_of(a, b)
    level_ci = _block_bootstrap(a, b, 60)

    # The evaluation's protocol: np.diff over the masked array, which silently
    # differences across every time gap (measured below).
    da_all, db_all = np.diff(a), np.diff(b)
    r_gap = float(np.corrcoef(da_all, db_all)[0, 1])
    # Gap-safe: only differences between adjacent 30 s grid cells of the original
    # grid, so no pair spans a missing epoch.
    pos = np.where(m)[0]
    contiguous = np.diff(pos) == 1
    da_c = np.diff(a)[contiguous]
    db_c = np.diff(b)[contiguous]
    r_contig, n_contig = r_of(da_c, db_c)
    contig_ci = _block_bootstrap(da_c, db_c, 20)

    return {
        "level": {"r": r_level, "n": n_level, "bootstrap95": level_ci},
        "diffs_gap_crossing": {"r": r_gap, "n": int(len(da_all)),
                               "n_cross_gap": int((~contiguous).sum())},
        "diffs_contiguous": {"r": r_contig, "n": n_contig,
                             "bootstrap95": contig_ci},
        "grid_s": 30.0,
        "n_valid_cells": int(m.sum()),
    }


# PURPOSE: Percentile interval of Pearson r under a moving-block bootstrap, so an
#          autocorrelated series does not borrow the significance of n i.i.d. samples.
# INPUTS: a, b (N,) paired arrays; block (int) block length in samples.
# OUTPUTS: [lo, hi] 2.5/97.5 percentiles over 1500 resamples; [nan, nan] when the
#          block is longer than the series.
# KEYWORDS: bootstrap, autocorrelation, uncertainty
def _block_bootstrap(a: np.ndarray, b: np.ndarray, block: int) -> list:
    n = len(a)
    if block >= n or n < 3:
        return [float("nan"), float("nan")]
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    rng = np.random.default_rng(11)
    rs = []
    for _ in range(1500):
        s = rng.choice(starts, size=n_blocks)
        idx = np.concatenate([np.arange(i, i + block) for i in s])[:n]
        if np.std(a[idx]) == 0 or np.std(b[idx]) == 0:
            continue
        rs.append(float(np.corrcoef(a[idx], b[idx])[0, 1]))
    if not rs:
        return [float("nan"), float("nan")]
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]


# PURPOSE: Attach the nearest ceilometer H1 to every dumped frame, within a stated
#          tolerance, so frames can be stratified by reference regime.
# INPUTS: none.
# OUTPUTS: (rows, h1_of_row) where rows are the parsed clusters.jsonl records with
#          an added 'H1' key; frames with no ceilometer sample inside the tolerance
#          get NaN and are excluded by the callers. Tolerance is 8 s, i.e. half the
#          instrument's 15 s slot.
# KEYWORDS: ceilometer, join, tolerance
def load_cluster_rows():
    path = os.path.join(CLUSTER_RUN, "clusters.jsonl")
    rows = [json.loads(line) for line in open(path)]
    samples = C.parse_ceilometer_log(LOG)
    ct = np.array([s.time.replace(tzinfo=None).timestamp()
                   for s in samples if s.h1 is not None])
    ch = np.array([s.h1 for s in samples if s.h1 is not None])
    for r in rows:
        t = dt.datetime.fromisoformat(r["ts_ceilometer_clock"])
        t = t.replace(tzinfo=None).timestamp() if t.tzinfo else t.timestamp()
        i = int(np.searchsorted(ct, t))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ct) and abs(ct[j] - t) <= 8.0:
                if best is None or abs(ct[j] - t) < abs(ct[best] - t):
                    best = j
        r["H1"] = float(ch[best]) if best is not None else float("nan")
    return rows


# PURPOSE: Does the stereo height distribution move with the reference's layer, and
#          in which direction?
# INPUTS: none.
# OUTPUTS: per-regime medians of the per-frame median candidate height (candidates
#          are clusters with size >= MIN_CLUSTER_SIZE), the between-regime shift, a
#          Kolmogorov-Smirnov statistic/p-value, and the same restricted to a
#          core radius so a residual radial geometry term cannot explain it.
# KEYWORDS: regime, stratification, ks_test, radial_control
def regime_response() -> dict:
    from scipy.stats import ks_2samp

    rows = load_cluster_rows()
    out = {}
    for label, rmax in (("full", None), ("r_le_250", 250.0)):
        lo, hi, lo_all, hi_all = [], [], [], []
        lo_iqr, hi_iqr, lo_cnt, hi_cnt = [], [], [], []
        for r in rows:
            if not np.isfinite(r["H1"]):
                continue
            cand = [c for c in r["clusters"] if c["size"] >= MIN_CLUSTER_SIZE
                    and (rmax is None
                         or np.hypot(c["u"] - 320.0, c["v"] - 320.0) <= rmax)]
            if not cand:
                continue
            h = np.array([c["h"] for c in cand], float)
            iqr = float(np.percentile(h, 75) - np.percentile(h, 25))
            cnt = float(sum(c["size"] for c in cand))
            if r["H1"] < REGIME_LOW_MAX_M:
                lo.append(float(np.median(h)))
                lo_all.extend(h.tolist())
                lo_iqr.append(iqr)
                lo_cnt.append(cnt)
            elif r["H1"] > REGIME_HIGH_MIN_M:
                hi.append(float(np.median(h)))
                hi_all.extend(h.tolist())
                hi_iqr.append(iqr)
                hi_cnt.append(cnt)
        lo, hi = np.array(lo), np.array(hi)
        ks = ks_2samp(lo, hi)
        out[label] = {
            "n_lower": int(len(lo)), "n_upper": int(len(hi)),
            "median_lower_m": float(np.median(lo)),
            "median_upper_m": float(np.median(hi)),
            "shift_upper_minus_lower_m": float(np.median(hi) - np.median(lo)),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "n_matches_lower": len(lo_all), "n_matches_upper": len(hi_all),
            "median_frame_height_iqr_lower_m": float(np.median(lo_iqr)),
            "median_frame_height_iqr_upper_m": float(np.median(hi_iqr)),
            "median_frame_match_count_lower": float(np.median(lo_cnt)),
            "median_frame_match_count_upper": float(np.median(hi_cnt)),
        }
    # radius distribution must be regime-independent for the core test to be a
    # control rather than a second effect
    rad = {"lower": [], "upper": []}
    for r in rows:
        if not np.isfinite(r["H1"]):
            continue
        cand = [c for c in r["clusters"] if c["size"] >= MIN_CLUSTER_SIZE]
        if not cand:
            continue
        rr = np.array([np.hypot(c["u"] - 320.0, c["v"] - 320.0) for c in cand])
        key = ("lower" if r["H1"] < REGIME_LOW_MAX_M
               else "upper" if r["H1"] > REGIME_HIGH_MIN_M else None)
        if key:
            rad[key].append(float(np.median(rr)))
    out["match_radius_by_regime"] = {
        k: {"median_px": float(np.median(v))} for k, v in rad.items()}
    return out


# PURPOSE: Characterise what the shipped "lowest cluster" rule actually selects, so
#          a fixed scene target can be distinguished from a cloud.
# INPUTS: none.
# OUTPUTS: per-rule height statistics for the hours 12:00-16:00 MSK, the selected
#          cluster's radius/position distribution and stationarity, and the same in
#          image position over time. Numbers only; no selection is changed.
# KEYWORDS: cluster_selection, stationarity, occluder, target_identity
def selection_diagnosis() -> dict:
    rows = load_cluster_rows()
    rules = {
        "lowest": lambda c: min(c, key=lambda x: x["h"])["h"],
        "largest": lambda c: max(c, key=lambda x: x["size"])["h"],
        "median_all": lambda c: float(np.median([x["h"] for x in c])),
    }
    result = {}
    ts = np.array([np.datetime64(r["ts_ceilometer_clock"]) for r in rows])
    for name, fn in rules.items():
        vals = []
        for r in rows:
            cand = [c for c in r["clusters"] if c["size"] >= MIN_CLUSTER_SIZE]
            vals.append(fn(cand) if cand else np.nan)
        vals = np.array(vals, float)
        hourly = []
        for hh in range(12, 17):
            mm = ((ts >= np.datetime64(f"2025-05-23T{hh:02d}:00"))
                  & (ts < np.datetime64(f"2025-05-23T{hh + 1:02d}:00")))
            hourly.append(float(np.nanmedian(vals[mm])) if mm.sum() else float("nan"))
        result[name] = {"median_m": float(np.nanmedian(vals)),
                        "hourly_medians_msk_12_16": hourly}

    sel = []
    for r in rows:
        big = [c for c in r["clusters"] if c["size"] >= MIN_CLUSTER_SIZE]
        if big:
            sel.append(min(big, key=lambda c: c["h"]))
    u = np.array([c["u"] for c in sel], float)
    v = np.array([c["v"] for c in sel], float)
    h = np.array([c["h"] for c in sel], float)
    rad = np.hypot(u - 320.0, v - 320.0)
    result["selected_lowest_cluster"] = {
        "n": len(sel),
        "height_median_m": float(np.median(h)),
        "height_iqr_m": float(np.percentile(h, 75) - np.percentile(h, 25)),
        "radius_median_px": float(np.median(rad)),
        "radius_p90_px": float(np.percentile(rad, 90)),
        "frac_radius_gt_320px": float((rad > 320).mean()),
        "u_median_px": float(np.median(u)), "v_median_px": float(np.median(v)),
        "share_in_top_60px_position_bin": float(
            max(np.bincount((u // 60).astype(int) * 64 + (v // 60).astype(int))
                ) / len(sel)),
    }
    return result


# PURPOSE: Availability of the ceilometer's secondary fields, so a planned
#          comparison cannot be built on a column that is empty.
# INPUTS: none.
# OUTPUTS: valid counts for H1/H2/L/Q on 2025-05-23 in the validation window and
#          over the whole file, plus the number of distinct Q codes.
# KEYWORDS: ceilometer, backscatter_width, data_availability
def ceilometer_fields() -> dict:
    samples = C.parse_ceilometer_log(LOG, quality_min=0)
    lo, hi = dt.time(12, 0, 0), dt.time(17, 0, 0)
    win = [s for s in samples if s.time.date() == dt.date(2025, 5, 23)
           and lo <= s.time.time() <= hi]
    def count(seq, attr):
        return int(sum(1 for s in seq if getattr(s, attr) is not None))
    return {
        "file_slots": len(samples),
        "window_slots": len(win),
        "window_h1": count(win, "h1"), "window_h2": count(win, "h2"),
        "window_l": count(win, "backscatter_l"), "window_q": count(win, "quality"),
        "file_l": count(samples, "backscatter_l"),
        "distinct_q_codes": sorted({s.quality for s in samples
                                    if s.quality is not None}),
    }


# PURPOSE: Decide whether the low heights that the "lowest" rule always reports are
#          localised in one occluder region or are a broad low tail of the field.
#          A localised occluder is removable by a mask; a broad tail is not, and
#          makes "lowest" a noise-extremum estimator rather than a layer estimator.
# INPUTS: none.
# OUTPUTS: selected-lowest median after excluding a symmetric border, after
#          excluding only the top band, and the height medians of matches inside vs
#          outside that band. Excluding a band that removes the low population is
#          the signature of a localised occluder.
# KEYWORDS: tail, order_statistic, border_artifact, occluder
def tail_diagnosis() -> dict:
    rows = load_cluster_rows()

    def selected(pred):
        out = []
        for r in rows:
            c = [x for x in r["clusters"] if x["size"] >= MIN_CLUSTER_SIZE
                 and pred(x)]
            out.append(min(c, key=lambda x: x["h"])["h"] if c else np.nan)
        return np.array(out, float)

    border = {}
    for b in (0, 32, 64, 96, 128):
        h = selected(lambda x, b=b: (x["u"] >= b and x["u"] < 640 - b
                                     and x["v"] >= b and x["v"] < 640 - b))
        border[f"symmetric_{b}px"] = {
            "n": int(np.isfinite(h).sum()), "median_m": float(np.nanmedian(h))}
    top = {}
    for b in (0, 64, 96, 128):
        h = selected(lambda x, b=b: x["v"] >= b)
        top[f"exclude_top_{b}px"] = {
            "n": int(np.isfinite(h).sum()), "median_m": float(np.nanmedian(h))}
    in_band, out_band = [], []
    for r in rows:
        for x in r["clusters"]:
            if x["size"] < MIN_CLUSTER_SIZE:
                continue
            (in_band if x["v"] < 64 else out_band).append(x["h"])
    return {
        "selected_after_symmetric_border": border,
        "selected_after_top_band_exclusion": top,
        "match_height_top_band_v_lt_64_m": {
            "n": len(in_band), "median_m": float(np.median(in_band))},
        "match_height_rest_m": {
            "n": len(out_band), "median_m": float(np.median(out_band))},
    }


def main():
    out = {
        "correlation_audit": correlation_audit(),
        "regime_response": regime_response(),
        "selection_diagnosis": selection_diagnosis(),
        "tail_diagnosis": tail_diagnosis(),
        "ceilometer_fields": ceilometer_fields(),
    }
    print(json.dumps(out, indent=1))
    dest = os.path.join(HERE, "anti_correlation_probe.json")
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
