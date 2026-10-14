# PURPOSE: Decide whether the persistent 1500-1800 m band in the height distribution
#          is the cloud or the near-field scene. The 640 px SQUARE crop has corners
#          at 452 px from its centre, and the frame corners contain trees, a building
#          and a mast. A near-field object converts a large disparity into a low
#          apparent height, so an occluder population would sit at LOW height and at
#          LARGE radius.
#
#          The test is therefore: how does the height distribution depend on distance
#          from the principal point? A cloud layer occupies the whole field; an
#          occluder population lives only in the outer annulus.
#
#          Nothing here consults the ceilometer.
# INPUTS: experiments/layers/heights.npz (per-match heights and image coordinates).
# OUTPUTS: docs/report/height_by_radius.png, printed tables.
# KEYWORDS: occluders, field_mask, validation, artefacts, radial_profile
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    H = list(d["heights"])
    UV = list(d["uv"])
    cx, cy = d["centre"]
    allh, allr = [], []
    for h, uv in zip(H, UV):
        if len(h) == 0:
            continue
        allh.append(h)
        allr.append(np.hypot(uv[:, 0] - cx, uv[:, 1] - cy))
    h = np.concatenate(allh)
    r = np.concatenate(allr)
    print(f"matches: {len(h)}")
    print(f"radius from crop centre: median {np.median(r):.0f} px, "
          f"p90 {np.percentile(r,90):.0f}, max {r.max():.0f} px")
    print(f"height: median {np.median(h):.0f} m\n")

    print(f"{'radius band':>14} {'matches':>9} {'share':>7} {'median h':>9} "
          f"{'p10':>7} {'p90':>7}")
    bands = [(0, 100), (100, 200), (200, 280), (280, 360), (360, 452)]
    stats = []
    for lo, hi in bands:
        m = (r >= lo) & (r < hi)
        if m.sum() < 50:
            continue
        stats.append((lo, hi, int(m.sum()), float(np.median(h[m])),
                      float(np.percentile(h[m], 10)), float(np.percentile(h[m], 90))))
        print(f"{f'{lo}-{hi} px':>14} {m.sum():>9} {100*m.mean():>6.1f}% "
              f"{np.median(h[m]):>9.0f} {np.percentile(h[m],10):>7.0f} "
              f"{np.percentile(h[m],90):>7.0f}")

    # height as a function of radius, in fine bins
    print(f"\n{'radius':>8} {'median height':>14}")
    prof = []
    for lo in range(0, 452, 25):
        m = (r >= lo) & (r < lo + 25)
        if m.sum() < 100:
            continue
        prof.append((lo + 12, float(np.median(h[m])), int(m.sum())))
        print(f"{lo:>6}-{lo+25:<3} {np.median(h[m]):>14.0f}")

    # the decisive comparison: central field only versus the whole crop
    for rad in (150, 200, 250, 300):
        m = r < rad
        print(f"\ncentral r < {rad} px: {m.sum()} matches "
              f"({100*m.mean():.1f} % of all), median height {np.median(h[m]):.0f} m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    x = [p[0] for p in prof]
    y = [p[1] for p in prof]
    ax1.plot(x, y, "o-", color="#1a7f37", lw=1.8, ms=4)
    ax1.set_xlabel("radius from principal point (px)")
    ax1.set_ylabel("median match height (m)")
    ax1.set_title("Height versus radius: occluders would pull the edge down, not up")
    ax1.axhline(np.median(h), color="#5c6672", ls="--", lw=1)
    ax1.annotate(f"whole-crop median {np.median(h):.0f} m", (300, np.median(h)),
                 textcoords="offset points", xytext=(0, 8), fontsize=8, color="#5c6672")

    bins = np.linspace(0, 4000, 90)
    for lo, hi, col, lbl in ((0, 150, "#1f6feb", "inner r < 150 px"),
                             (300, 452, "#c1440e", "outer r > 300 px")):
        m = (r >= lo) & (r < hi)
        ax2.hist(h[m], bins=bins, density=True, histtype="step", lw=1.8,
                 color=col, label=f"{lbl}  (n={m.sum()}, med {np.median(h[m]):.0f} m)")
    ax2.set_xlabel("match height (m)")
    ax2.set_ylabel("density")
    ax2.set_title("Inner versus outer field")
    ax2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "height_by_radius.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/height_by_radius.png")


if __name__ == "__main__":
    main()
