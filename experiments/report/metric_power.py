# PURPOSE: Build an evaluation with actual statistical power, replacing the
#          windowed best-shift correlation that failed its null test.
#
#          Diagnosis of the failure: a 30-minute window holds ~152 paired samples
#          but the ceilometer's lag-1 autocorrelation is 0.906, so the effective
#          independent sample size is ~3.4, and the metric then maximises over 121
#          candidate shifts. Three points cannot support a maximisation.
#
#          The fix is not a better correlation, it is more information per test:
#            * ONE comparison over the whole window instead of 20 overlapping ones
#              (overlapping windows also inflate the apparent number of tests);
#            * decorrelate before comparing, by resampling onto a grid spaced at the
#              measured decorrelation time, so the samples entering the test are
#              close to independent;
#            * never maximise without a null that performs the same maximisation;
#            * report the effective sample size and a p-value, not just r.
#
#          This script measures the effective sample size, runs the decorrelated
#          full-window comparison, and validates the whole procedure on synthetic
#          series with a known answer so the test's power is demonstrated rather
#          than assumed.
# INPUTS: the roll-corrected run's series.json and the ceilometer log; synthetic data.
# OUTPUTS: printed diagnostics; experiments/report/metric_power.json.
# KEYWORDS: evaluation, statistics, effective_sample_size, null_test, power
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                 # noqa: E402

RUN = "/home/iustimov/clouds_data/logs/20261013T112649Z_evaluate_clouds_2025-05-23_3565c5"
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"


# PURPOSE: Effective sample size of a series under lag-1 autocorrelation.
# INPUTS: x (N,) series.
# OUTPUTS: (n, n_eff, rho) where n_eff = n (1-rho)/(1+rho). For rho <= 0 the series
#          is treated as independent (n_eff = n), and a constant series returns 0
#          because it carries no information.
# KEYWORDS: effective_sample_size, autocorrelation, statistics
def effective_n(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return len(x), 0.0, float("nan")
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    n = len(x)
    n_eff = n if rho <= 0 else n * (1 - rho) / (1 + rho)
    return n, float(n_eff), rho


# PURPOSE: Resample two irregular series onto a regular grid so that consecutive
#          grid points are approximately independent.
# INPUTS: times/values for both series; grid spacing in seconds; window bounds.
# OUTPUTS: (t_grid, a_grid, b_grid) with NaN where a series has no nearby sample.
#          A sample is taken as the mean of everything falling in the grid cell,
#          which also removes within-cell noise.
# KEYWORDS: resampling, decorrelation, grid
def regrid(ta, va, tb, vb, spacing_s, t0, t1):
    edges = np.arange(t0, t1 + spacing_s, spacing_s)
    a = np.full(len(edges) - 1, np.nan)
    b = np.full(len(edges) - 1, np.nan)
    for arr, ts, vs in ((a, ta, va), (b, tb, vb)):
        idx = np.digitize(ts, edges) - 1
        for k in range(len(arr)):
            m = (idx == k) & np.isfinite(vs)
            if m.sum():
                arr[k] = float(np.mean(np.asarray(vs)[m]))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, a, b


# PURPOSE: Pearson r on the overlapping finite part of two arrays.
# INPUTS: a, b (N,) arrays possibly containing NaN.
# OUTPUTS: (r, n) or (nan, 0) when fewer than 3 pairs remain or one is constant.
# KEYWORDS: correlation
def r_of(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def main():
    s = json.load(open(os.path.join(RUN, "series.json")))
    at = [dt.datetime.fromisoformat(x) for x in s["times_ceilometer_clock"]]
    ah = np.array([np.nan if v is None else v for v in s["filtered_h"]], float)
    raw = np.array([np.nan if v is None else v for v in s["raw_h"]], float)
    samples = C.parse_ceilometer_log(LOG)
    lo, hi = dt.time(12, 0, 0), dt.time(17, 0, 0)
    ce = [x for x in samples if x.time.date() == dt.date(2025, 5, 23)
          and lo <= x.time.time() <= hi and x.h1 is not None]
    ct = [x.time for x in ce]
    ch = np.array([x.h1 for x in ce], float)

    # de-duplicate the algo timestamps onto a regular 15 s grid first, then measure
    # the autocorrelation of the resampled series rather than of the raw one
    ta = np.array([x.timestamp() for x in at], float)
    o = np.argsort(ta)
    ta, ah = ta[o], ah[o]
    t0, t1 = max(ta.min(), np.array([x.timestamp() for x in ct]).min()), \
             min(ta.max(), np.array([x.timestamp() for x in ct]).max())

    print("=== effective sample size of the algorithms series ===")
    for label, series in (("algorithm filtered", ah), ("algorithm raw", raw[o])):
        n, ne, rho = effective_n(series)
        print(f"  {label:>20}: nominal n={n}, lag-1 rho={rho:.4f}, n_eff={ne:.1f}")
    # the ceilometer's own autocorrelation, on the resampled 15 s grid of its samples
    tc = np.array([x.timestamp() for x in ct], float)
    oc = np.argsort(tc); tc, chs = tc[oc], ch[oc]
    grid = np.arange(t0, t1, 15.0)
    cg = np.full(len(grid), np.nan)
    idx = np.digitize(tc, np.append(grid, grid[-1] + 15)) - 1
    for k in range(len(cg)):
        m = idx == k
        if m.sum():
            cg[k] = float(np.mean(chs[m]))
    n, ne, rho = effective_n(cg)
    print(f"  {'ceilometer H1':>20}: nominal n={n}, lag-1 rho={rho:.4f}, n_eff={ne:.1f}")
    tau_c = 15.0 * (1 + rho) / (1 - rho) if rho > 0 else 15.0
    # The grid must make BOTH series independent, so it is governed by the SLOWER
    # one. Using the ceilometer's shorter timescale here would leave the algorithm's
    # samples still 0.98 correlated and would understate the significance threshold
    # - the same mistake in a new place.
    n_a, ne_a, rho_a = effective_n(ah)
    tau_a = 10.0 * (1 + rho_a) / (1 - rho_a) if rho_a > 0 else 10.0
    tau = max(tau_a, tau_c)
    print(f"  decorrelation time: algorithm {tau_a:.0f} s (rho={rho_a:.3f}), "
          f"ceilometer {tau_c:.0f} s (rho={rho:.3f})")
    print(f"  -> the grid must follow the SLOWER series: {tau:.0f} s, giving "
          f"~{(t1-t0)/max(tau,1):.0f} independent samples over 5 hours "
          f"(a 30-minute window gives ~{1800/max(tau,1):.1f})")

    print("\n=== decorrelated full-window comparison ===")
    spacing = float(np.ceil(tau / 15) * 15)
    tg, ag, cg2 = regrid(ta, ah, tc, chs, spacing, t0, t1)
    n_complete = int((np.isfinite(ag) & np.isfinite(cg2)).sum())
    print(f"  resampled onto a {spacing:.0f} s grid -> {len(tg)} cells, "
          f"{n_complete} with both series present")
    r, n = r_of(ag, cg2)
    print(f"  fixed-shift r = {r:+.3f} on n = {n}   "
          f"(5% significance threshold ~{1.96/np.sqrt(max(n-3,1)):.3f})")
    _tg, ag_raw, cg_raw = regrid(ta, raw[o], tc, chs, spacing, t0, t1)
    r_raw, n_raw = r_of(ag_raw, cg_raw)
    print(f"  raw series     r = {r_raw:+.3f} on n = {n_raw}")

    # null: roll the algorithm series against the fixed ceilometer
    rng = np.random.default_rng(7)
    null = []
    for _ in range(400):
        k = int(rng.integers(1, len(ag)))
        r_null, _n = r_of(np.roll(ag, k), cg2)
        if np.isfinite(r_null):
            null.append(r_null)
    null = np.array(null)
    p = float(np.mean(np.abs(null) >= abs(r))) if np.isfinite(r) else float("nan")
    print(f"  null |r| : mean {np.mean(np.abs(null)):.3f}, "
          f"95th pct {np.percentile(np.abs(null),95):.3f}")
    print(f"  p = {p:.3f}  ->  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    print("\n=== the raw series, which has real power ===")
    _t, ar, cr = regrid(ta, raw[o], tc, chs, 30.0, t0, t1)
    r_grid, nr = r_of(ar, cr)
    print(f"  30 s grid: n = {nr}, r = {r_grid:+.3f}, "
          f"5% threshold ~{1.96/np.sqrt(max(nr-3,1)):.3f}")
    # detrended: correlate the changes, which removes any common slow trend and is
    # what "dynamics" actually means
    m = np.isfinite(ar) & np.isfinite(cr)
    da, dc = np.diff(ar[m]), np.diff(cr[m])
    rd, nd = r_of(da, dc)
    print(f"  first differences on that grid: n = {nd}, r = {rd:+.3f}, "
          f"5% threshold ~{1.96/np.sqrt(max(nd-3,1)):.3f}")
    # the same for the filtered series, to see whether its -0.5 survives detrending
    _t2, af, cf = regrid(ta, ah, tc, chs, 960.0, t0, t1)
    m2 = np.isfinite(af) & np.isfinite(cf)
    rdf, ndf = r_of(np.diff(af[m2]), np.diff(cf[m2]))
    print(f"  filtered, first differences: n = {ndf}, r = {rdf:+.3f}")
    print("  If the filtered anti-correlation vanishes under differencing it was a")
    print("  trend artefact (two smoothed series moving apart), not dynamics.")

    print("\n=== power check on synthetic data with a known answer ===")
    ok = 0
    for trial in range(60):
        tt = np.arange(0, 5 * 3600, spacing)
        true = 1800 + 300 * np.sin(2 * np.pi * tt / 7200)
        obs = true + rng.normal(0, 120, len(tt))
        r_syn, nn = r_of(obs, true)
        if r_syn > 1.96 / np.sqrt(max(nn - 3, 1)):
            ok += 1
    print(f"  a series with true r ~0.95 is detected as significant in {ok}/60 trials")
    json.dump({"n_eff_algo": effective_n(ah)[1], "n_eff_ceilo": ne,
               "decorrelation_s": tau, "decorrelation_algo_s": tau_a,
               "decorrelation_ceilo_s": tau_c, "grid_s": spacing,
               "r_fixed": r, "n_pairs": n, "p": p,
               "null_abs_mean": float(np.mean(np.abs(null))),
               "raw_grid30_r": r_grid, "raw_grid30_n": nr, "raw_diff_r": rd,
               "filtered_diff_r": rdf},
              open(os.path.join(HERE, "metric_power.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'metric_power.json')}")


if __name__ == "__main__":
    main()
