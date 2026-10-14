# PURPOSE: Compare the corrected stereo algorithm against the older one on the same
#          validation window, on the same axes as the ceilometer, and score both
#          with the project's existing time-series metrics.
#
#          Three height series are shown:
#            * "as shipped"      — the historical configuration (roll 0, f = xi0/pi)
#            * "roll corrected"  — the measured -1.57 deg residual roll removed
#            * "roll + f"        — additionally scaled by the measured focal constant
#
#          The focal correction is applied here as a pure multiplicative scale rather
#          than by re-running, because D is exactly proportional to f; the scale is
#          printed so the transformation is auditable. The ceilometer is used ONLY to
#          score and to draw; it never feeds an estimate (guardrail G1).
#
#          Dynamics (does it move when the cloud moves) and level (does it give the
#          same height) are reported separately and never merged (guardrail G8).
# INPUTS: two run directories containing series.json, and the project config.
# OUTPUTS: docs/report/comparison_timeseries.png, comparison_metrics.png,
#          comparison.json, and a metrics table on stdout.
# KEYWORDS: comparison, validation, time_series, correlation, ceilometer
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.metrics import alignment as align                        # noqa: E402
OUT = os.path.join(REPO, "docs", "report")

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.autolayout": True,
})
INK = "#1b2430"
CEILO = "#5c6672"
BASE = "#c1440e"
CORR = "#1a7f37"
FSCALE = "#1f6feb"

# measured in docs/plan/w1-calibration.md W1.8: f = 654 +/- 11 px/rad against the
# assumed xi0/pi = 611.15
F_ASSUMED = 1920.0 / math.pi
F_MEASURED = 654.0


def parse(ts):
    return dt.datetime.fromisoformat(ts)


def load(run_dir):
    s = json.load(open(os.path.join(run_dir, "series.json")))
    times = [parse(t) for t in s["times_ceilometer_clock"]]
    raw = np.array([np.nan if v is None else v for v in s["raw_h"]], float)
    filt = np.array([np.nan if v is None else v for v in s["filtered_h"]], float)
    ct = [parse(a) for a, _ in s["ceilometer"]]
    ch = np.array([b for _, b in s["ceilometer"]], float)
    return dict(times=times, raw=raw, filt=filt, ct=ct, ch=ch,
                metrics=json.load(open(os.path.join(run_dir, "metrics.json"))))


# PURPOSE: Pair two irregular time series by nearest timestamp within a tolerance.
# INPUTS: t_a, v_a, t_b, v_b (times and values), tolerance in seconds.
# OUTPUTS: (va_paired, vb_paired) arrays, only for pairs that matched.
# KEYWORDS: alignment, pairing, nearest, time_series
def pair(t_a, v_a, t_b, v_b, tol_s=30.0):
    out_a, out_b = [], []
    j = 0
    order = np.argsort(t_b)
    tb = np.array(t_b, dtype="datetime64[s]")[order]
    vb = np.asarray(v_b, float)[order]
    for ta, va in zip(t_a, v_a):
        if not np.isfinite(va):
            continue
        k = np.searchsorted(tb, np.datetime64(ta, "s"))
        best = None
        for cand in (k - 1, k, k + 1):
            if 0 <= cand < len(tb):
                d = abs((tb[cand] - np.datetime64(ta, "s")) / np.timedelta64(1, "s"))
                if d <= tol_s and (best is None or d < best[0]):
                    best = (d, cand)
        if best and np.isfinite(vb[best[1]]):
            out_a.append(va)
            out_b.append(vb[best[1]])
    return np.array(out_a), np.array(out_b)


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# PURPOSE: Correlation of first differences: "does the estimate move when the cloud
#          moves", with the level removed entirely. This is the cleanest dynamics
#          measure available and it cannot be inflated by a constant offset.
# INPUTS: times, values, ceilometer times/values, tolerance.
# OUTPUTS: Pearson r of the differenced series, or nan.
# KEYWORDS: dynamics, differences, correlation
def diff_r(times, values, ct, ch, tol_s=60):
    a, b = pair(times, values, ct, ch, tol_s=tol_s)
    if len(a) < 10:
        return float("nan")
    return corr(np.diff(a), np.diff(b))


# PURPOSE: Windowed correlation using the *project's own* metric, so the numbers are
#          directly comparable with previous runs. Reimplementing it was a mistake:
#          the project's version searches a per-window time shift (the two
#          instruments sample different air, so they lag) and a naive nearest-time
#          correlation understates agreement badly. This delegates to
#          clouds.metrics.alignment.windowed_correlation rather than approximating it.
# INPUTS: times, values, ceilometer times/values; window, shift search, tolerance.
# OUTPUTS: list of per-window correlations on the same definition as the runs.
# KEYWORDS: metrics, windowed_correlation, project_metric
def windowed_r(times, values, ct, ch, window_min=30, hop_min=15,
               shift_s=300, tol_s=60):
    ok = np.isfinite(values)
    rows = align.windowed_correlation([t for t, m in zip(times, ok) if m],
                                      [v for v, m in zip(values, ok) if m],
                                      ct, ch, window_min, shift_s, tol_s, hop_min)
    return [r["r"] for r in rows if np.isfinite(r["r"])]


def main():
    base_dir, corr_dir = sys.argv[1], sys.argv[2]
    B = load(base_dir)
    C = load(corr_dir)
    print(f"baseline  : {os.path.basename(base_dir)}")
    print(f"corrected : {os.path.basename(corr_dir)}")
    print(f"  frames: baseline {len(B['times'])}, corrected {len(C['times'])}, "
          f"ceilometer {len(B['ct'])}")

    # the focal correction, applied as an exact multiplicative scale
    fscale = F_MEASURED / F_ASSUMED
    C_f = {**C, "filt": C["filt"] * fscale, "raw": C["raw"] * fscale}
    print(f"  focal correction: D -> D * {F_MEASURED:.0f}/{F_ASSUMED:.1f} = {fscale:.4f}")

    variants = [("as shipped", B, BASE), ("roll corrected", C, CORR),
                ("roll + f corrected", C_f, FSCALE)]

    print(f"\n{'variant':>20} {'median h':>9} {'r(shift)':>9} {'r(fixed)':>9} "
          f"{'r(df/frame)':>12} {'r>0.5':>7} {'max r':>7} {'global r':>9} "
          f"{'jump m':>8} {'autocorr':>9}")
    table = {}
    for name, V, _col in variants:
        m = V["metrics"]
        wr = windowed_r(V["times"], V["filt"], V["ct"], V["ch"],
                        window_min=30, hop_min=15, shift_s=300)
        wr_fixed = windowed_r(V["times"], V["filt"], V["ct"], V["ch"],
                              window_min=30, hop_min=15, shift_s=0)
        dr = diff_r(V["times"], V["filt"], V["ct"], V["ch"])
        wr = [x for x in wr if np.isfinite(x)]
        a, b = pair(V["times"], V["filt"], V["ct"], V["ch"], tol_s=1800)
        med = float(np.nanmedian(V["filt"]))
        jd = np.abs(np.diff(V["filt"]))
        jd = jd[np.isfinite(jd)]
        ac = m["smoothing"]["filtered_autocorr"]
        table[name] = dict(median=med, mean_r=float(np.mean(wr)) if wr else float("nan"),
                           mean_r_fixed_shift=float(np.mean(wr_fixed)) if wr_fixed else float("nan"),
                           diff_r=dr,
                           frac_gt_half=float(np.mean([x > 0.5 for x in wr])) if wr else float("nan"),
                           max_r=float(np.max(wr)) if wr else float("nan"),
                           global_r=corr(a, b), jump=float(np.median(jd)) if len(jd) else float("nan"),
                           autocorr=ac, n_windows=len(wr))
        t = table[name]
        print(f"{name:>20} {t['median']:>9.0f} {t['mean_r']:>9.3f} "
              f"{t['mean_r_fixed_shift']:>9.3f} {t['diff_r']:>12.3f} "
              f"{t['frac_gt_half']:>7.2f} {t['max_r']:>7.3f} {t['global_r']:>9.3f} "
              f"{t['jump']:>8.1f} {t['autocorr']:>9.3f}")

    json.dump(table, open(os.path.join(OUT, "comparison.json"), "w"), indent=1)

    # ---------------- figure 1: the time series ----------------
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    ax = axes[0]
    ax.scatter(B["ct"], B["ch"], s=6, color=CEILO, alpha=0.65, zorder=2,
               label=f"ceilometer DOL-2  (n={len(B['ct'])})")
    ax.plot(B["times"], B["raw"], color=BASE, lw=0.7, alpha=0.30,
            label="as shipped — per-frame raw")
    ax.plot(B["times"], B["filt"], color=BASE, lw=1.5, zorder=3,
            label="as shipped — filtered")
    ax.plot(C["times"], C["filt"], color=CORR, lw=1.5, zorder=4,
            label="roll corrected — filtered")
    ax.plot(C_f["times"], C_f["filt"], color=FSCALE, lw=1.5, ls="--", zorder=4,
            label=f"roll + f corrected (×{fscale:.3f})")
    ax.set_ylabel("cloud base height (m)")
    ax.set_ylim(0, 4200)
    ax.set_title("Algorithm heights against the ceilometer — same window, same clock")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.92)

    ax2 = axes[1]
    ax2.plot(B["times"], B["filt"], color=BASE, lw=1.4, label="as shipped — filtered")
    ax2.plot(C["times"], C["filt"], color=CORR, lw=1.4, label="roll corrected — filtered")
    ax2.scatter(B["ct"], B["ch"], s=5, color=CEILO, alpha=0.55, label="ceilometer")
    ax2.set_ylabel("height (m)")
    ax2.set_xlabel("time (MSK, ceilometer clock)")
    ax2.set_ylim(900, 3200)
    ax2.set_title("Zoomed: the staggered step pattern the correction preserves")
    ax2.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.92)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    fig.savefig(os.path.join(OUT, "comparison_timeseries.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------------- figure 2: the metrics ----------------
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    names = [v[0] for v in variants]
    cols = [v[2] for v in variants]

    ax = axes[0]
    x = np.arange(len(names))
    vals_s = [table[n]["mean_r"] for n in names]
    vals_f = [table[n]["mean_r_fixed_shift"] for n in names]
    ax.bar(x - 0.2, vals_s, 0.4, color=cols, label="best shift per window")
    ax.bar(x + 0.2, vals_f, 0.4, color=cols, alpha=0.45, hatch="//",
           label="fixed shift, no search")
    for i, (vs, vf) in enumerate(zip(vals_s, vals_f)):
        ax.text(i - 0.2, vs + 0.008, f"{vs:.2f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, vf + 0.008, f"{vf:.2f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color=CEILO, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("mean windowed r")
    ax.set_ylim(0, max(max(vals_s) * 1.35, 0.6))
    ax.set_title("Dynamics: windowed correlation")
    ax.legend(fontsize=7, loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)

    ax = axes[1]
    vals = [table[n]["jump"] for n in names]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("median frame-to-frame jump (m)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Smoothness: frame-to-frame change")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)

    ax = axes[2]
    for name, V, col in variants:
        a, b = pair(V["times"], V["filt"], V["ct"], V["ch"], tol_s=1800)
        ax.scatter(b, a, s=5, alpha=0.35, color=col, label=f"{name} (r={corr(a,b):.2f})")
    lim = [0, 4200]
    ax.plot(lim, lim, color=INK, lw=1, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("ceilometer height (m)")
    ax.set_ylabel("algorithm height (m)")
    ax.set_title("Level: estimate vs ceilometer")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(os.path.join(OUT, "comparison_metrics.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {OUT}/comparison_timeseries.png and comparison_metrics.png")


if __name__ == "__main__":
    main()
