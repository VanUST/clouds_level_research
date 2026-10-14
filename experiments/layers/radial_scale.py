# PURPOSE: Identify the systematic error behind the radial trend in apparent height.
#
#          The measured height falls from ~1850 m at the principal point to ~1300 m
#          at the edge of the crop, under every candidate lens law (equidistant,
#          equisolid, stereographic, orthographic), so the projection model is not
#          the cause. A flat cloud layer must give the SAME height everywhere, so
#          the trend is an instrument or calibration signature.
#
#          A disparity that grows linearly with image radius is the signature of a
#          SCALE difference between the two cameras: if one camera's effective
#          focal constant is larger by a fraction s, every correspondence acquires
#          an extra disparity (s-1)*r. This script measures that term directly from
#          the disparity field and then tests whether removing it flattens the
#          height profile - which is the falsifiable prediction.
#
#          No ground truth and no ceilometer are used.
# INPUTS: experiments/layers/heights.npz.
# OUTPUTS: docs/report/radial_scale.png, printed before/after profile.
# KEYWORDS: scale_error, calibration, radial_trend, disparity_field
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
    K1, K2 = list(d["uv"]), list(d["kp2"])
    f = float(d["f"])
    B = float(d["baseline"])
    cx, cy = d["centre"]

    # measure the disparity as a function of radius, which is the raw signature
    disp, rad = [], []
    for a, b in zip(K1, K2):
        if len(a) == 0:
            continue
        disp.append((a[:, 0] - b[:, 0]).astype(np.float64))
        rad.append(np.hypot(a[:, 0] - cx, a[:, 1] - cy))
    disp = np.concatenate(disp)
    rad = np.concatenate(rad)
    print(f"matches {len(disp)}, median disparity {np.median(disp):.2f} px\n")
    print(f"{'radius':>10} {'matches':>8} {'median disp':>12}")
    prof = []
    for lo in np.arange(0, 425, 25):
        m = (rad >= lo) & (rad < lo + 25)
        if m.sum() < 100:
            continue
        prof.append((lo + 12.5, float(np.median(disp[m])), int(m.sum())))
        print(f"{lo:>5}-{lo+25:<4} {m.sum():>8} {np.median(disp[m]):>12.2f}")

    x = np.array([p[0] for p in prof])
    y = np.array([p[1] for p in prof])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"\ndisparity = {slope:.5f} * r + {intercept:.2f}   "
          f"(so a {100*slope:.2f} % radial scale term, "
          f"{100*slope*300/max(intercept,1e-9):.1f} % of the disparity at r=300)")

    # Falsifiable prediction: removing slope*r must flatten the height profile.
    # Only valid where intercept - slope*r stays positive, which bounds the radius
    # the correction can be applied to.
    print(f"\n{'radius':>10} {'h as measured':>14} {'h with slope removed':>22}")
    h_before, h_after = [], []
    for lo in np.arange(0, 425, 25):
        m = (rad >= lo) & (rad < lo + 25)
        if m.sum() < 100:
            continue
        r_mid = lo + 12.5
        corr = disp[m] - slope * (rad[m] - r_mid)   # de-trend about this bin
        hb = f * B / np.median(disp[m])
        ha = f * B / np.median(corr)
        h_before.append((r_mid, hb))
        h_after.append((r_mid, ha))
        print(f"{r_mid:>10.0f} {hb:>14.0f} {ha:>22.0f}")

    hb = np.array([p[1] for p in h_before])
    ha = np.array([p[1] for p in h_after])
    print(f"\nspread of the profile (max-min)/mean over the crop:")
    print(f"  as measured        : {100*(hb.max()-hb.min())/hb.mean():.1f} %")
    print(f"  with the radial term removed: {100*(ha.max()-ha.min())/ha.mean():.1f} %")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.5))
    ax1.plot(x, y, "o-", color="#c1440e", lw=1.8, ms=4)
    ax1.plot(x, slope * x + intercept, "--", color="#1b2430", lw=1.2,
             label=f"slope {slope:.5f} px/px")
    ax1.set_xlabel("radius from principal point (px)")
    ax1.set_ylabel("median disparity (px)")
    ax1.set_title("Disparity grows with radius")
    ax1.legend(fontsize=8)

    ax2.plot([p[0] for p in h_before], hb, "o-", color="#c1440e", lw=1.8, ms=4,
             label="as measured")
    ax2.plot([p[0] for p in h_after], ha, "s-", color="#1a7f37", lw=1.8, ms=4,
             label="radial term removed")
    ax2.set_xlabel("radius from principal point (px)")
    ax2.set_ylabel("median height (m)")
    ax2.set_title("Does removing it flatten the height?")
    ax2.legend(fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "radial_scale.png"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT}/radial_scale.png")


if __name__ == "__main__":
    main()
