# PURPOSE: Answer question 1 — is the pipeline's per-window "best time shift"
#          a real maximum? Re-scan each 30-min window over a much wider shift
#          range than the pipeline's +-300 s and characterise the correlation
#          curve (peak height, prominence, width, boundary hits, multiplicity),
#          and compare the observed peak against a null built by destroying the
#          time alignment while preserving each series' autocorrelation.
#          This script never estimates a height and never writes measurement
#          data; it only calls the project's own paired_r on stored series.
# INPUTS: nothing (reads the stored run series and re-scans).
# OUTPUTS: experiments/temporal_alignment/q1_shift_scan.json and
#          q1_shift_scan.png, plus a printed table.
# KEYWORDS: correlation, shift_scan, null_test, boundary_artefact, diagnostics
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

from clouds.metrics.alignment import paired_r  # noqa: E402
import common  # noqa: E402

WINDOW_MIN = 30
HOP_MIN = 15
TOL_S = 10.0
STEP = 5
WIDE_S = 1800          # question 1 asks for at least +-1800 s
PIPE_RANGE_S = 300     # the pipeline's search boundary
N_NULL = 120           # rotations per window for the null
NULL_STEP = 10


# PURPOSE: Scan paired correlation and pair count as a function of applied time
#          shift. The pair count is returned because at large shifts a 30-min
#          window and its shifted copy barely overlap: a correlation computed on
#          a 3-pair tail can reach 1.0 and must not be mistaken for a peak.
# INPUTS: a_ts, a_h, c_ts, c_h (np.ndarray); half_range_s, step_s (ints).
# OUTPUTS: (shifts, rs, ns) np.ndarrays; rs is NaN where < 3 pairs survive.
# KEYWORDS: shift_scan, correlation, curve, overlap
def scan(a_ts, a_h, c_ts, c_h, half_range_s, step_s):
    shifts = np.arange(-half_range_s, half_range_s + 1, step_s)
    rn = [paired_r(a_ts, a_h, c_ts, c_h, float(s), TOL_S) for s in shifts]
    rs = np.array([x[0] for x in rn], dtype=float)
    ns = np.array([x[1] for x in rn], dtype=int)
    return shifts, rs, ns


# PURPOSE: Mark which shifts of a scan have enough overlapping pairs to score,
#          so tail artefacts at large |shift| cannot win the maximum.
# INPUTS: ns (np.ndarray of pair counts); n_at_zero (int, pairs at shift 0).
# OUTPUTS: boolean np.ndarray; True where ns >= max(10, half of n_at_zero).
# KEYWORDS: gating, overlap, quality
def overlap_ok(ns, n_at_zero):
    return ns >= max(10.0, 0.5 * n_at_zero)


# PURPOSE: Extract shape diagnostics from one r(shift) curve.
# INPUTS: shifts, rs (np.ndarray); r_min_consider (float).
# OUTPUTS: dict with global peak, prominence above the curve median, the
#          half-prominence width (seconds), the count of local maxima within
#          0.05 of the peak (separated by >= 150 s), and the fraction of the
#          scanned curve within 0.05 of the peak. nan when the curve is empty.
# KEYWORDS: peak, prominence, width, multiplicity, flatness
def curve_diagnostics(shifts, rs, r_min_consider=-0.99):
    ok = ~np.isnan(rs)
    if ok.sum() < 3:
        return {"peak_r": float("nan"), "peak_shift": None}
    sh, r = shifts[ok], rs[ok]
    i = int(np.argmax(r))
    peak_r, peak_shift = float(r[i]), float(sh[i])
    med = float(np.median(r))
    prominence = peak_r - med
    # Half-prominence width: the contiguous run around the peak inside which the
    # curve stays above median + prominence/2.
    thresh = med + 0.5 * prominence if prominence > 0 else peak_r
    above = r >= thresh
    lo = i
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi + 1 < len(r) and above[hi + 1]:
        hi += 1
    width_s = float(sh[hi] - sh[lo])
    # Multiplicity: local maxima comparable to the global one. Peaks closer than
    # 150 s are the same broad hump, not a second candidate.
    locals_ = []
    for k in range(1, len(r) - 1):
        if r[k] >= r[k - 1] and r[k] >= r[k + 1] and r[k] >= peak_r - 0.05:
            if not locals_ or sh[k] - locals_[-1] >= 150:
                locals_.append(sh[k])
    return {
        "peak_r": peak_r, "peak_shift": peak_shift,
        "median_r": med, "prominence": prominence,
        "half_prom_width_s": width_s,
        "n_comparable_peaks": len(locals_),
        "comparable_peak_shifts": [float(x) for x in locals_],
        "frac_within_0p05_of_peak": float(np.mean(r >= peak_r - 0.05)),
        "frac_above_zero": float(np.mean(r > 0.0)),
    }


# PURPOSE: Build the null distribution of the *scan maximum* by rotating the
#          algorithm series against the fixed ceilometer series, so any real
#          time relation is destroyed while both marginal distributions and
#          their autocorrelation are preserved.
# INPUTS: a_ts, a_h, c_ts, c_h; n_draws, null_step, half_range_s.
# OUTPUTS: np.ndarray of null max-r values (NaN draws dropped).
# KEYWORDS: null, permutation, circular_shift, significance
def null_max_r(a_ts, a_h, c_ts, c_h, n_draws=N_NULL, null_step=NULL_STEP,
               half_range_s=PIPE_RANGE_S):
    rng = np.random.default_rng(20250523)
    n = a_ts.size
    n0 = paired_r(a_ts, a_h, c_ts, c_h, 0.0, TOL_S)[1]
    out = []
    for _ in range(n_draws):
        k = int(rng.integers(1, n))
        a_h_rot = np.roll(a_h, k)
        _, rs, ns = scan(a_ts, a_h_rot, c_ts, c_h, half_range_s, null_step)
        ok = overlap_ok(ns, n0) & ~np.isnan(rs)
        if ok.sum() >= 3:
            out.append(float(np.max(rs[ok])))
    return np.array(out, dtype=float)


# PURPOSE: Run the full question-1 analysis for one height series.
# INPUTS: series dict (from common.load_series); height_key ("raw_h"/"filt_h").
# OUTPUTS: list of per-window dicts with curve diagnostics and null p-values.
# KEYWORDS: windows, diagnostics, null, summary
def analyse(series, height_key: str):
    grid = common.window_grid(series["a_ts"], series["c_ts"], WINDOW_MIN, HOP_MIN)
    h_all = series[height_key]
    rows = []
    for w in grid:
        a_ts = series["a_ts"][w["a_idx"]]
        a_h = h_all[w["a_idx"]]
        c_ts = series["c_ts"][w["c_idx"]]
        c_h = series["c_h"][w["c_idx"]]
        shifts, rs, ns = scan(a_ts, a_h, c_ts, c_h, WIDE_S, STEP)
        n0 = paired_r(a_ts, a_h, c_ts, c_h, 0.0, TOL_S)[1]
        ok = overlap_ok(ns, n0) & ~np.isnan(rs)
        d = curve_diagnostics(shifts[ok], rs[ok])
        # Peak restricted to the pipeline's +-300 s search, same overlap gate.
        m = ok & (np.abs(shifts) <= PIPE_RANGE_S)
        d_pipe = curve_diagnostics(shifts[m], rs[m])
        d["n_pairs_at_0"] = int(n0)
        d["n_scored_shifts"] = int(ok.sum())
        d["n_scored_shifts_in_pipe"] = int(m.sum())
        # Match the null's coarser step (10 s) when reading the observed maximum
        # so the p-value compares like with like.
        m10 = m & (np.mod(shifts, NULL_STEP) == 0)
        d["pipe_peak_r_step10"] = (float(np.nanmax(rs[m10]))
                                   if m10.sum() else float("nan"))
        d["pipe_peak_r"] = d_pipe["peak_r"]
        d["pipe_peak_shift"] = d_pipe["peak_shift"]
        d["pipe_peak_at_boundary"] = bool(
            d_pipe.get("peak_shift") is not None
            and abs(abs(d_pipe["peak_shift"]) - PIPE_RANGE_S) < 1e-6)
        r_at_0 = paired_r(a_ts, a_h, c_ts, c_h, 0.0, TOL_S)[0]
        d["r_at_0"] = None if np.isnan(r_at_0) else float(r_at_0)
        # Wide peak outside the pipeline's window, and better than the pipeline's?
        d["wide_peak_outside_pipe"] = bool(
            d.get("peak_shift") is not None and abs(d["peak_shift"]) > PIPE_RANGE_S)
        d["wide_peak_better_than_pipe"] = bool(
            not np.isnan(d["peak_r"]) and not np.isnan(d_pipe["peak_r"])
            and d["peak_r"] > d_pipe["peak_r"] + 0.02)
        nulls = null_max_r(a_ts, a_h, c_ts, c_h)
        d["null_max_median"] = float(np.median(nulls)) if nulls.size else float("nan")
        d["null_max_p95"] = float(np.percentile(nulls, 95)) if nulls.size else float("nan")
        d["p_scanmax"] = (float(np.mean(nulls >= d["pipe_peak_r_step10"]))
                          if nulls.size else float("nan"))
        d["start"] = w["start_dt"].isoformat()
        d["n_algo"] = w["n_algo"]
        d["n_ceilo"] = w["n_ceilo"]
        d["n_pairs_at_peak"] = int(paired_r(a_ts, a_h, c_ts, c_h,
                                            float(d_pipe["peak_shift"] or 0), TOL_S)[1])
        rows.append(d)
        # Effective degrees of freedom: both series are smooth, so the number of
        # paired samples hugely overstates the independent evidence. Estimate it
        # from the lag-1 autocorrelation of the ceilometer series in the window.
        ch = series["c_h"][w["c_idx"]]
        if ch.size > 3:
            ac1 = float(np.corrcoef(ch[:-1], ch[1:])[0, 1])
            d["ceilo_ac1"] = ac1
            d["n_eff_ceilo"] = float(ch.size * (1 - ac1) / (1 + ac1)) if ac1 < 1 else float("nan")
    return rows


# PURPOSE: Draw and save the question-1 figure (curve overlay, shift histogram,
#          observed vs null peak, peak-height scatter).
# INPUTS: rows_raw, rows_filt (list[dict]); out_png (str).
# OUTPUTS: writes the PNG; returns None.
# KEYWORDS: figure, diagnostics, matplotlib
def make_figure(rows_raw, rows_filt, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    per = []
    for name, rows, colour in (("raw", rows_raw, "tab:blue"),
                               ("filtered", rows_filt, "tab:red")):
        per.append((name, rows, colour))
    # (a) peak shift of the wide scan and of the pipeline scan.
    axa = ax[0, 0]
    for name, rows, colour in per:
        s_pipe = [r["pipe_peak_shift"] for r in rows]
        s_wide = [r["peak_shift"] for r in rows]
        axa.scatter(s_pipe, [f"{name} pipe" for _ in rows], c=colour, alpha=0.4)
        axa.scatter(s_wide, [f"{name} wide" for _ in rows], c=colour, marker="^", alpha=0.6)
    axa.axvline(300, color="k", ls="--", lw=1)
    axa.axvline(-300, color="k", ls="--", lw=1)
    axa.set_xlabel("best shift (s)")
    axa.set_title("(a) best-shift location, per window")
    # (b) peak height vs null max
    axb = ax[0, 1]
    for name, rows, colour in per:
        obs = [r["pipe_peak_r"] for r in rows]
        null95 = [r["null_max_p95"] for r in rows]
        axb.scatter(null95, obs, c=colour, label=name)
    lim = [min(0, min(axb.get_xlim()[0], axb.get_ylim()[0])),
           max(axb.get_xlim()[1], axb.get_ylim()[1])]
    axb.plot(lim, lim, "k--", lw=1, label="observed = null 95th pct")
    axb.set_xlabel("null 95th pct of scan max r")
    axb.set_ylabel("observed pipeline peak r")
    axb.legend(fontsize=8)
    axb.set_title("(b) observed peak vs shuffled-alignment null")
    # (c) fraction of windows whose peak sits at the search boundary
    axc = ax[1, 0]
    width = 0.35
    xs = np.arange(len(per))
    frac_at = [np.mean([r["pipe_peak_at_boundary"] for r in rows]) for _, rows, _ in per]
    frac_out = [np.mean([r["wide_peak_outside_pipe"] for r in rows]) for _, rows, _ in per]
    frac_better = [np.mean([r["wide_peak_better_than_pipe"] for r in rows]) for _, rows, _ in per]
    axc.bar(xs - width, frac_at, width, label="pipe peak at +-300 s")
    axc.bar(xs, frac_out, width, label="wide peak outside +-300 s")
    axc.bar(xs + width, frac_better, width, label="wide peak > pipe peak")
    axc.set_xticks(xs)
    axc.set_xticklabels([name for name, _, _ in per])
    axc.set_ylim(0, 1)
    axc.set_ylabel("fraction of windows")
    axc.legend(fontsize=8)
    axc.set_title("(c) boundary and out-of-range behaviour")
    # (d) curve overlay
    axd = ax[1, 1]
    series = None
    for name, rows, colour in per:
        if name == "filtered":
            series = common.load_series(common.RUN_ROLL_CORRECTED)
    grid = common.window_grid(series["a_ts"], series["c_ts"], WINDOW_MIN, HOP_MIN)
    for w in grid:
        a_ts = series["a_ts"][w["a_idx"]]
        a_h = series["filt_h"][w["a_idx"]]
        c_ts = series["c_ts"][w["c_idx"]]
        c_h = series["c_h"][w["c_idx"]]
        shifts, rs, ns = scan(a_ts, a_h, c_ts, c_h, WIDE_S, STEP)
        n0 = paired_r(a_ts, a_h, c_ts, c_h, 0.0, TOL_S)[1]
        ok = overlap_ok(ns, n0)
        axd.plot(shifts[ok], rs[ok], lw=0.8, alpha=0.7)
    axd.axvspan(-300, 300, color="grey", alpha=0.15)
    axd.axhline(0, color="k", lw=0.8)
    axd.set_xlabel("applied shift (s)")
    axd.set_ylabel("windowed r")
    axd.set_title("(d) r(shift) for all 20 windows (filtered), grey = pipeline search")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"figure written: {out_png}")


# PURPOSE: Print the per-window diagnostic table and the aggregates the finding
#          quotes.
# INPUTS: label (str); rows (list[dict]).
# OUTPUTS: prints a table; returns the aggregate dict.
# KEYWORDS: report, aggregate, console
def report(label, rows):
    print(f"\n=== {label} ===")
    hdr = ("start              pipe_r  pipe_shift  wide_r  wide_shift  prom   "
           "width_s  nPeak  r@0    null95  p")
    print(hdr)
    for r in rows:
        print(f"{r['start'][11:19]}  {r['pipe_peak_r']:6.3f}  "
              f"{str(r['pipe_peak_shift']):>7}  {r['peak_r']:6.3f}  "
              f"{str(r['peak_shift']):>7}  {r['prominence']:6.3f}  "
              f"{r['half_prom_width_s']:7.0f}  {r['n_comparable_peaks']:5d}  "
              f"{r['r_at_0']:6.3f}  {r['null_max_p95']:6.3f}  {r['p_scanmax']:.3f}")
    agg = {
        "n_windows": len(rows),
        "mean_pipe_peak_r": float(np.mean([r["pipe_peak_r"] for r in rows])),
        "mean_wide_peak_r": float(np.mean([r["peak_r"] for r in rows])),
        "mean_r_at_0": float(np.nanmean([r["r_at_0"] for r in rows])),
        "frac_pipe_peak_at_boundary": float(np.mean([r["pipe_peak_at_boundary"] for r in rows])),
        "frac_pipe_peak_within_1step_of_boundary": float(np.mean(
            [abs(abs(r["pipe_peak_shift"]) - PIPE_RANGE_S) <= STEP for r in rows])),
        "frac_pipe_peak_within_25s_of_boundary": float(np.mean(
            [abs(r["pipe_peak_shift"]) >= PIPE_RANGE_S - 25 for r in rows])),
        "frac_wide_peak_outside_pipe": float(np.mean([r["wide_peak_outside_pipe"] for r in rows])),
        "frac_wide_peak_better_than_pipe": float(np.mean([r["wide_peak_better_than_pipe"] for r in rows])),
        "mean_n_comparable_peaks": float(np.mean([r["n_comparable_peaks"] for r in rows])),
        "frac_windows_multi_peak": float(np.mean([r["n_comparable_peaks"] > 1 for r in rows])),
        "median_half_prom_width_s": float(np.median([r["half_prom_width_s"] for r in rows])),
        "median_prominence": float(np.median([r["prominence"] for r in rows])),
        "frac_positive_prominence": float(np.mean([r["prominence"] > 0 for r in rows])),
        "mean_null_max_p95": float(np.nanmean([r["null_max_p95"] for r in rows])),
        "median_p_scanmax": float(np.nanmedian([r["p_scanmax"] for r in rows])),
        "frac_p_scanmax_lt_0p05": float(np.nanmean([r["p_scanmax"] < 0.05 for r in rows])),
        "frac_pipe_peak_exceeds_null95": float(np.nanmean(
            [r["pipe_peak_r"] > r["null_max_p95"] for r in rows])),
        "median_frac_within_0p05_of_peak": float(np.median(
            [r["frac_within_0p05_of_peak"] for r in rows])),
        "median_ceilo_ac1": float(np.nanmedian([r.get("ceilo_ac1", float("nan")) for r in rows])),
        "median_n_eff_ceilo": float(np.nanmedian([r.get("n_eff_ceilo", float("nan")) for r in rows])),
        "median_n_pairs_at_0": float(np.median([r["n_pairs_at_0"] for r in rows])),
    }
    print("\naggregate:", json.dumps(agg, indent=1))
    return agg


# PURPOSE: Entry point: analyse both height series, print, save JSON and figure.
# INPUTS: none.
# OUTPUTS: q1_shift_scan.json and q1_shift_scan.png beside this script.
# KEYWORDS: main, entry_point
def main():
    series = common.load_series(common.RUN_ROLL_CORRECTED)
    rows_raw = analyse(series, "raw_h")
    rows_filt = analyse(series, "filt_h")
    agg_raw = report("raw_h", rows_raw)
    agg_filt = report("filt_h", rows_filt)
    out = {"run": common.RUN_ROLL_CORRECTED, "raw": rows_raw, "filtered": rows_filt,
           "aggregate_raw": agg_raw, "aggregate_filtered": agg_filt}
    with open(os.path.join(HERE, "q1_shift_scan.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(rows_raw, rows_filt, os.path.join(HERE, "q1_shift_scan.png"))


if __name__ == "__main__":
    main()
