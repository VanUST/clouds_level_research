# PURPOSE: Characterise the ceilometer's LAYER structure over the validation window,
#          because a correlation between a single camera series and a ceilometer
#          series is only meaningful if the two track the same cloud. The DOL-2
#          reports up to two layers (H1, H2) plus a backscatter width L; if H1
#          swaps between two different clouds, no time shift can align the series
#          and any reported correlation is meaningless regardless of lag.
#
#          This is the complementary question to the temporal-lag research: the lag
#          only matters if both instruments are watching one moving cloud.
# INPUTS: the ceilometer log for 2025-05-23 and the algorithm's height series.
# OUTPUTS: layer statistics and a figure; printed table.
# KEYWORDS: ceilometer, layers, temporal_alignment, non_stationarity
from __future__ import annotations

import datetime as dt
import json
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
from clouds.sources import ceilometer as C                              # noqa: E402

OUT = os.path.join(REPO, "docs", "report")
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
RUN = "/home/iustimov/clouds_data/logs/20261013T112649Z_evaluate_clouds_2025-05-23_3565c5"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})


def main():
    samples = C.parse_ceilometer_log(LOG)
    lo = dt.time(12, 0, 0)
    hi = dt.time(17, 0, 0)
    w = [s for s in samples
         if s.time.date() == dt.date(2025, 5, 23) and lo <= s.time.time() <= hi
         and s.h1 is not None]
    print(f"ceilometer samples in the window: {len(w)}")
    t = [s.time for s in w]
    h1 = np.array([s.h1 for s in w], float)
    h2 = np.array([np.nan if s.h2 is None else s.h2 for s in w], float)
    q = np.array([np.nan if s.quality is None else s.quality for s in w], float)

    print(f"\nH1: median {np.median(h1):.0f} m, range {h1.min():.0f}-{h1.max():.0f} m")
    present2 = np.isfinite(h2)
    print(f"H2 present in {present2.sum()}/{len(w)} samples ({100*present2.mean():.0f} %)")
    if present2.sum():
        print(f"   H2 median {np.nanmedian(h2):.0f} m, range {np.nanmin(h2):.0f}-{np.nanmax(h2):.0f} m")

    # layer switching: a change of more than 500 m in one step cannot be a real
    # cloud-base motion, so it is the instrument changing which layer it reports
    dt_s = np.array([(t[i + 1] - t[i]).total_seconds() for i in range(len(t) - 1)])
    dh = np.diff(h1)
    big = np.abs(dh) > 500
    print(f"\nconsecutive H1 steps: {len(dh)}")
    print(f"  |change| > 500 m in one step: {big.sum()} ({100*big.mean():.1f} %)")
    print(f"  median |change|: {np.median(np.abs(dh)):.0f} m")
    print(f"  median sample interval: {np.median(dt_s):.0f} s")

    # does H1 sit in a stable band, or wander between two populations?
    hist, edges = np.histogram(h1, bins=np.arange(1000, 4500, 250))
    print("\nH1 histogram (250 m bins):")
    for c, e in zip(hist, edges):
        if c:
            print(f"  {e:5.0f}-{e+250:5.0f} m | {'#' * int(60 * c / hist.max()):<60} {c}")

    alg = json.load(open(os.path.join(RUN, "series.json")))
    at = [dt.datetime.fromisoformat(x) for x in alg["times_ceilometer_clock"]]
    ah = np.array([np.nan if v is None else v for v in alg["filtered_h"]], float)
    print(f"\nalgorithm (roll corrected) median {np.nanmedian(ah):.0f} m, "
          f"p10-p90 {np.nanpercentile(ah,10):.0f}-{np.nanpercentile(ah,90):.0f} m")
    print(f"H1 median {np.median(h1):.0f} m  ->  algorithm is "
          f"{100*(np.nanmedian(ah)-np.median(h1))/np.median(h1):+.0f} % relative to H1")

    # ---------------- regime split ----------------
    # If the disagreement is mostly layer IDENTITY rather than a time lag, then
    # restricting the comparison to periods when the ceilometer is on one layer
    # should change the correlation, while a pure lag problem would not care.
    from clouds.metrics import alignment as align
    at_arr = np.array([x.timestamp() for x in at], float)
    ah_arr = ah.copy()
    print("\n--- correlation split by which layer the ceilometer reports ---")
    print(f"{'regime':>28} {'n_algo':>7} {'n_ceil':>7} {'r shift=0':>10} {'r best +-300s':>14}")
    regimes = [("H1 below 2750 m (lower)", h1 < 2750),
               ("H1 above 3500 m (upper)", h1 > 3500),
               ("all samples", np.ones(len(h1), bool))]
    for name, mask in regimes:
        ct = [t[i] for i in range(len(t)) if mask[i]]
        ch = h1[mask]
        if len(ct) < 20:
            continue
        r0 = align.windowed_correlation(
            [x for x in at], [v if np.isfinite(v) else None for v in ah_arr],
            ct, list(ch), 30, 0, 60, 15)
        r0 = [x["r"] for x in r0 if np.isfinite(x["r"])]
        r3 = align.windowed_correlation(
            [x for x in at], [v if np.isfinite(v) else None for v in ah_arr],
            ct, list(ch), 30, 300, 60, 15)
        r3 = [x["r"] for x in r3 if np.isfinite(x["r"])]
        print(f"{name:>28} {len(at):>7} {len(ct):>7} "
              f"{(np.mean(r0) if r0 else float('nan')):>10.3f} "
              f"{(np.mean(r3) if r3 else float('nan')):>14.3f}")
    print("\n  A lag problem would leave these similar; a layer-identity problem")
    print("  changes them, because the shared signal only exists inside a regime.")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(t, h1, color="#5c6672", lw=1.1, label="ceilometer H1 (lowest layer)")
    ax.plot(t, h2, color="#8b949e", lw=1.1, ls=":", label="ceilometer H2 (second layer)")
    ax.plot(at, ah, color="#1a7f37", lw=1.4, label="algorithm (roll corrected)")
    ax.set_ylabel("height (m)")
    ax.set_ylim(0, 4400)
    ax.set_title("The ceilometer's reported base moves between populations; the algorithm does not")
    ax.legend(fontsize=8, loc="upper right", ncol=3)

    ax2 = axes[1]
    ax2.plot(t[1:], np.abs(dh), color="#c1440e", lw=0.9)
    ax2.axhline(500, color="#1b2430", ls="--", lw=1)
    ax2.annotate("500 m: too large to be cloud motion,\nso the instrument changed layer",
                 (t[len(t) // 6], 520), fontsize=7.5, color="#1b2430")
    ax2.set_ylabel("|H1 change| per step (m)")
    ax2.set_xlabel("time (MSK)")
    ax2.set_yscale("symlog", linthresh=100)
    ax2.set_title("Layer switching: the ground truth is not one continuous cloud")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "ceilometer_layers.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/ceilometer_layers.png")


if __name__ == "__main__":
    main()
