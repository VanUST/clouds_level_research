# PURPOSE: Test the falsifiable prediction of workplan R2 — that applying the measured
#          lens distortion makes the apparent cloud height independent of image radius.
#
#          The defect: with the shipped ideal-equidistant law the apparent height falls
#          from ~1849 m at the principal point to ~1254 m at the edge of the crop, a
#          36.7 % swing. A flat cloud layer must give the same height everywhere, so
#          this is an instrument signature. Fitting theta = (r/f)(1 + a(r/f)^2) to the
#          sky itself gives a = -0.30 and a 3.2 % residual spread
#          (experiments/layers/distortion_fit.py).
#
#          This checks the same thing through the ACTUAL PIPELINE: take the per-cluster
#          heights from a --dump-clusters run, bin them by the cluster's distance from
#          the principal point, and compare the radial profile between the shipped
#          configuration, exact triangulation alone, and exact triangulation with the
#          distortion applied. If the prediction holds, the third profile is flat.
#
#          Ground-truth-free: the ceilometer is not read.
# INPUTS: clusters.jsonl from two or more evaluation runs.
# OUTPUTS: printed radial profiles and a figure.
# KEYWORDS: distortion, R2, radial_profile, verification, falsifiable
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
CROP = 640
CENTRE = CROP / 2.0


# PURPOSE: Load every cluster from a run's dump, in the cropped frame's coordinates.
# INPUTS: path to clusters.jsonl.
# OUTPUTS: (heights, radii) arrays over all clusters of all frames.
#          Clusters without a finite positive height are dropped.
# KEYWORDS: load, clusters, dump
def load(path):
    h, r = [], []
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for c in rec.get("clusters", []):
                hv = c.get("h")
                if hv is None or not np.isfinite(hv) or hv <= 0:
                    continue
                u, v = c.get("u"), c.get("v")
                if u is None or v is None:
                    continue
                h.append(float(hv))
                r.append(float(np.hypot(u - CENTRE, v - CENTRE)))
    return np.array(h), np.array(r)


# PURPOSE: Median height per radius bin.
# INPUTS: heights, radii, bin width, maximum radius.
# OUTPUTS: (centres, medians, counts) for bins with enough clusters.
# KEYWORDS: profile
def profile(h, r, step=25.0, rmax=452.0, min_n=500):
    """A bin needs enough clusters to be a median rather than an anecdote: the
    outermost bin holds very few clusters and every one of them has a near-zero
    disparity, so a handful of them produced a 10,625 m 'median' that dominated the
    max-min spread and made the comparison meaningless. min_n removes that."""
    c, m, n = [], [], []
    for lo in np.arange(0, rmax, step):
        mask = (r >= lo) & (r < lo + step)
        if mask.sum() >= min_n:
            c.append(lo + step / 2)
            m.append(float(np.median(h[mask])))
            n.append(int(mask.sum()))
    return np.array(c), np.array(m), np.array(n)


def main():
    runs = {
        "as shipped (paraxial, ideal lens)": sys.argv[1],
        "exact triangulation, ideal lens": sys.argv[2] if len(sys.argv) > 2 else None,
        "exact + distortion a=-0.30": sys.argv[3] if len(sys.argv) > 3 else None,
    }
    print(f"{'configuration':>36} {'clusters':>9} {'inner':>8} {'outer':>8} "
          f"{'ratio':>7} {'IQR':>8} {'IQR%':>7}")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    summary = {}
    for label, path in runs.items():
        if not path or not os.path.exists(path):
            continue
        h, r = load(path)
        if len(h) < 100:
            print(f"{label:>36}  (too few clusters)")
            continue
        c, m, n = profile(h, r)
        if len(m) < 4:
            continue
        # Robust summaries. max-min is discarded: it is decided by the extreme bins,
        # and the profile is a trend, not an envelope. The interquartile range of the
        # profile measures how much the median height actually moves with radius,
        # and the inner/outer ratio states its direction.
        iqr = float(np.percentile(m, 75) - np.percentile(m, 25))
        inner = float(np.mean(m[:4]))
        outer = float(np.mean(m[-4:]))
        ratio = outer / inner
        summary[label] = {"clusters": int(len(h)), "inner_m": inner,
                          "outer_m": outer, "ratio": ratio,
                          "profile_iqr_m": iqr,
                          "flatness_pct": 100 * iqr / float(np.mean(m))}
        print(f"{label:>36} {len(h):>9} {inner:>8.0f} {outer:>8.0f} "
              f"{ratio:>7.2f} {iqr:>8.0f} {100*iqr/float(np.mean(m)):>7.1f}%")
        ax.plot(c, m, "o-", ms=4, lw=1.8,
                label=f"{label}  (outer/inner {ratio:.2f})")
    ax.set_xlabel("cluster distance from the principal point (px)")
    ax.set_ylabel("median cluster height (m)")
    ax.set_title("Does the correction remove the radial trend? A flat layer needs a flat line")
    ax.legend(fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "r2_radial_profile.png"), bbox_inches="tight",
                facecolor="white")
    json.dump(summary, open(os.path.join(HERE, "r2_radial_profile.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/r2_radial_profile.png")
    print("Prediction (workplan R2): the distortion profile's flatness should improve")
    print("by more than 20 %, with the median height moving less than 3 %.")


if __name__ == "__main__":
    main()
