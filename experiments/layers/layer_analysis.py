# PURPOSE: Test the layer hypothesis directly. If the sky usually contains several
#          distinct cloud layers, the per-match height distribution should be
#          multi-modal and those modes should persist in time; if the "layers" are
#          an artefact of the clustering settings, the distribution should be a
#          single broad hump that merely drifts.
#
#          Two products:
#            * a height-versus-time density map of every match, which shows the
#              structure without any clustering choice at all;
#            * per-frame mode detection with a significance criterion, so "how many
#              layers" becomes a measured quantity rather than a parameter.
#
#          The ceilometer's H1/H2 are drawn for comparison only; they never enter
#          the mode detection.
# INPUTS: experiments/layers/heights.npz, and the ceilometer log.
# OUTPUTS: docs/report/layer_structure.png, layer_modes.json, a printed table.
# KEYWORDS: layers, multimodality, KDE, mode_detection, validation
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
BIN_M = 50.0
HMAX = 5000.0


# PURPOSE: Find significant local maxima in a smoothed height histogram.
#          A mode counts only if it stands a minimum prominence above the deeper of
#          the two valleys around it, which is what separates a real layer from
#          ripple on a single broad distribution. The prominence threshold is
#          expressed as a fraction of the frame's total mass, so it does not depend
#          on how many matches the frame happened to yield.
# INPUTS: heights (N,) in metres; bin width; maximum height; bin-mass thresholds.
# OUTPUTS: list of (height, mass_fraction) for each accepted mode, tallest-weighted
#          first; empty list for an empty or single-valued input.
# KEYWORDS: mode_detection, prominence, layer
def find_modes(heights, bin_m=BIN_M, hmax=HMAX, min_prom=0.02, min_mass=0.005):
    """
    Smooths a 50 m histogram with a 5-bin triangular kernel and delegates peak
    finding to scipy.signal.find_peaks, whose prominence is exactly the quantity
    wanted: the height of a peak above the higher of the two valleys separating it
    from a taller peak. Prominence and mass are then expressed as fractions of the
    frame's total match count, so the criteria do not depend on match yield.
    """
    from scipy.signal import find_peaks
    if heights.size < 20:
        return []
    edges = np.arange(0, hmax + bin_m, bin_m)
    hist, _ = np.histogram(heights, bins=edges)
    total = int(hist.sum())
    if total == 0:
        return []
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    sm = np.convolve(hist.astype(float), k, mode="same")
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx, props = find_peaks(sm, prominence=min_prom * total)
    modes = []
    for i, prom in zip(idx, props["prominences"]):
        if hist[i] / total < min_mass:
            continue
        modes.append((float(centres[i]), float(hist[i] / total), float(prom / total)))
    modes.sort(key=lambda m: -m[1])
    kept = []
    for h, mass, prom in modes:
        if all(abs(h - h2) > 2 * bin_m for h2, _, _ in kept):
            kept.append((h, mass, prom))
    return kept


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    stamps = [dt.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S") for s in d["stamps"]]
    H = list(d["heights"])
    print(f"frames {len(stamps)}, matches {sum(len(h) for h in H)}")
    print(f"heights: median {np.median(np.concatenate(H)):.0f} m, "
          f"p5 {np.percentile(np.concatenate(H),5):.0f}, "
          f"p95 {np.percentile(np.concatenate(H),95):.0f} m")

    # ---- density map ----
    edges = np.arange(0, HMAX + BIN_M, BIN_M)
    M = np.zeros((len(edges) - 1, len(stamps)))
    for k, h in enumerate(H):
        c, _ = np.histogram(h, bins=edges)
        M[:, k] = c
    Mn = M / np.maximum(M.sum(axis=0, keepdims=True), 1)

    # ---- modes per frame ----
    nmodes, modall = [], []
    for k, h in enumerate(H):
        m = find_modes(h)
        nmodes.append(len(m))
        modall.append(m)
    nmodes = np.array(nmodes)
    print(f"\nmodes per frame: mean {nmodes.mean():.2f}, median {np.median(nmodes):.0f}, "
          f"distribution {dict(zip(*np.unique(nmodes, return_counts=True)))}")
    print(f"  frames with >= 2 modes: {int((nmodes>=2).sum())}/{len(nmodes)} "
          f"({100*(nmodes>=2).mean():.0f} %)")

    # how persistent is the strongest mode? measure frame-to-frame motion of the
    # primary mode, which for a real layer should be small
    prim = np.array([m[0][0] if m else np.nan for m in modall])
    step_all = np.abs(np.diff(prim))
    step = step_all[np.isfinite(step_all)]
    print(f"  primary-mode height: median {np.nanmedian(prim):.0f} m, "
          f"p10-p90 {np.nanpercentile(prim,10):.0f}-{np.nanpercentile(prim,90):.0f} m")
    if step.size:
        print(f"  primary mode frame-to-frame change: median {np.median(step):.0f} m, "
              f"p90 {np.percentile(step,90):.0f} m")
    else:
        print("  primary mode frame-to-frame change: not available (no modes found)")

    json.dump({"n_frames": len(stamps),
               "modes_per_frame_hist": {int(k): int(v) for k, v in
                                        zip(*np.unique(nmodes, return_counts=True))},
               "primary_median_m": float(np.nanmedian(prim)),
               "primary_step_median_m": float(np.median(step)) if step.size else None},
              open(os.path.join(HERE, "layer_modes.json"), "w"), indent=1)

    # ---- figure ----
    samples = C.parse_ceilometer_log(LOG)
    lo, hi = dt.time(12, 0, 0), dt.time(17, 0, 0)
    ce = [s for s in samples if s.time.date() == dt.date(2025, 5, 23)
          and lo <= s.time.time() <= hi and s.h1 is not None]
    ct = [s.time for s in ce]
    ch1 = [s.h1 for s in ce]
    ch2 = [s.h2 if s.h2 is not None else np.nan for s in ce]

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    ax = axes[0]
    # matplotlib date numbers, not POSIX seconds: the axis formatters below expect
    # days since the matplotlib epoch, and feeding them seconds overflows.
    tc = np.asarray(mdates.date2num(stamps), float)
    centres_h = 0.5 * (edges[:-1] + edges[1:])
    vmax = float(np.percentile(Mn[Mn > 0], 99.0)) if (Mn > 0).any() else 1.0
    pcm = ax.pcolormesh(np.array(tc), centres_h, Mn, cmap="viridis",
                        shading="nearest", vmax=vmax)
    ct_num = np.asarray(mdates.date2num(ct), float)
    ax.plot(ct_num, ch1, color="#ffffff", lw=1.6, label="ceilometer H1")
    ax.plot(ct_num, ch2, color="#ff9f43", lw=1.2, ls=":", label="ceilometer H2")
    ax.plot(tc, prim, color="#ff4d4d", lw=1.0, label="strongest mode")
    ax.set_ylabel("height (m)")
    ax.set_ylim(0, HMAX)
    ax.set_title("Every match's height over time — layers appear as persistent bands, "
                 "clustering is not involved")
    ax.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.9)
    cb = fig.colorbar(pcm, ax=ax, pad=0.01)
    cb.set_label("fraction of the frame's matches", fontsize=8)

    ax2 = axes[1]
    ax2.plot(tc, nmodes, color="#1f6feb", lw=0.9, drawstyle="steps-mid")
    ax2.axhline(nmodes.mean(), color="#1b2430", ls="--", lw=1)
    ax2.set_ylabel("significant modes")
    ax2.set_ylim(0, max(6, nmodes.max() + 1))
    ax2.set_title(f"Detected layers per frame (mean {nmodes.mean():.2f})")

    ax3 = axes[2]
    ax3.plot(tc[1:], step[:len(tc) - 1], color="#c1440e", lw=0.9)
    if step.size:
        ax3.axhline(np.median(step), color="#1b2430", ls="--", lw=1)
    ax3.set_ylabel("strongest-mode change (m)")
    ax3.set_xlabel("time (MSK)")
    ax3.set_title("Frame-to-frame motion of the strongest mode")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "layer_structure.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/layer_structure.png")


if __name__ == "__main__":
    main()
