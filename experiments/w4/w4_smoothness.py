# PURPOSE: W4 gate 2 — are the correspondences under each regime spatially coherent, or
#          is the disparity difference carried by scattered noise?
#
#          Gate 1 established the essential fact: the regimes differ in disparity by
#          +1.8 px POOLED and at EVERY certainty decile (+0.8 to +2.5 px), so certainty
#          cannot explain it. Gate 1 also corrected a premise: the match RATE is
#          identical between regimes (11,655 vs 11,656 per frame) because RoMa returns a
#          fixed dense grid. There is no larger population in one regime - the SAME grid
#          points carry different disparities. So the "extra population" of the directive
#          is better described as a different disparity ASSIGNMENT at the same pixels.
#
#          That reframes the question this gate answers: are those assignments coherent?
#          A cloud surface is smooth, so a match should agree with its immediate
#          neighbours. Scattered noise does not. If the high regime's field is as
#          coherent as the low regime's, the difference is a real displacement of the
#          matching solution and not an artefact of admitting noise.
#
#          Ground-truth-free and internal: no ceilometer enters the metric, only the
#          frame's own matches.
# INPUTS: experiments/w4/matches_roma.npz, the ceilometer log.
# OUTPUTS: docs/report/w4_smoothness.png, experiments/w4/w4_smoothness.json.
# KEYWORDS: W4, smoothness, coherence, noise, internal, gate2
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
ST = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")
K = 25                     # neighbours for the local field


def main():
    d = np.load(os.path.join(HERE, "matches_roma.npz"), allow_pickle=True)
    stamps = list(d["stamps"]); uv = list(d["uv"]); kp2 = list(d["kp2"])
    conf = list(d["conf"])
    series = [(x.time, x.h1) for x in C.parse_ceilometer_log(LOG) if x.h1 is not None]

    def ref_at(stamp):
        m = ST.search(str(stamp))
        if not m:
            return None
        t = dt.datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
                        int(m.group(2)), int(m.group(3)), int(m.group(4))) + dt.timedelta(hours=3)
        best, bd = None, None
        for tt, hh in series:
            dd = abs((tt - t).total_seconds())
            if bd is None or dd < bd:
                best, bd = hh, dd
        return best if (bd is not None and bd <= 120) else None

    lo_res, hi_res = [], []
    lo_med, hi_med = [], []
    for i in range(len(stamps)):
        r = ref_at(stamps[i])
        if r is None or len(uv[i]) < 500:
            continue
        pos = np.asarray(uv[i], float)
        du = pos[:, 0] - np.asarray(kp2[i], float)[:, 0]
        good = np.isfinite(du)
        pos, du = pos[good], du[good]
        if len(du) < 500:
            continue
        # the residual of each match against the local median of its neighbours
        tree = cKDTree(pos)
        _, idx = tree.query(pos, k=min(K, len(pos)))
        local = np.median(du[idx], axis=1)
        resid = float(np.median(np.abs(du - local)))
        med = float(np.median(du))
        if r < 2750:
            lo_res.append(resid); lo_med.append(med)
        elif r > 3500:
            hi_res.append(resid); hi_med.append(med)

    lo_res = np.array(lo_res); hi_res = np.array(hi_res)
    print(f"frames: low {len(lo_res)}, high {len(hi_res)}   "
          f"(neighbourhood k={K})")
    print(f"\nlocal smoothness residual |d - median(neighbours)|:")
    print(f"  low regime   median {np.median(lo_res):.3f} px   "
          f"p90 {np.percentile(lo_res,90):.3f} px")
    print(f"  high regime  median {np.median(hi_res):.3f} px   "
          f"p90 {np.percentile(hi_res,90):.3f} px")
    ratio = float(np.median(hi_res) / max(np.median(lo_res), 1e-9))
    print(f"  ratio high/low = {ratio:.3f}")

    # the disparity difference measured again here, as the reference for the comparison
    dd = float(np.median(hi_med) - np.median(lo_med))
    print(f"\n  disparity difference (high - low) = {dd:+.3f} px")
    print(f"  smoothness difference = {np.median(hi_res)-np.median(lo_res):+.3f} px "
          f"({100*(ratio-1):+.1f} %)")

    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(lo_res, hi_res, alternative="two-sided")
    print(f"  Mann-Whitney on the residuals: p = {p:.2e}")

    if ratio < 1.10:
        verdict = ("BOTH REGIMES ARE EQUALLY COHERENT — the disparity difference is a "
                   "real displacement of the matching solution, not admitted noise")
    elif ratio > 1.25:
        verdict = ("THE HIGH REGIME IS MATERIALLY LESS COHERENT — its difference is "
                   "carried by scattered correspondences")
    else:
        verdict = "MARGINAL — the high regime is slightly less coherent"
    print(f"\n  VERDICT: {verdict}")

    json.dump({"n_low": int(len(lo_res)), "n_high": int(len(hi_res)),
               "resid_low_px": float(np.median(lo_res)),
               "resid_high_px": float(np.median(hi_res)), "ratio": ratio,
               "disparity_diff_px": dd, "mw_p": float(p), "verdict": verdict},
              open(os.path.join(HERE, "w4_smoothness.json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    bins = np.linspace(0, max(np.percentile(np.concatenate([lo_res, hi_res]), 99), 0.1), 30)
    ax1.hist(lo_res, bins=bins, alpha=0.6, color="#1f6feb", label="low regime", density=True)
    ax1.hist(hi_res, bins=bins, alpha=0.6, color="#c1440e", label="high regime", density=True)
    ax1.set_xlabel("local smoothness residual (px)"); ax1.set_ylabel("density")
    ax1.set_title("Disagreement with the local disparity field", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.axvline(np.median(lo_res), color="#1f6feb", ls="--", lw=1)
    ax1.axvline(np.median(hi_res), color="#c1440e", ls="--", lw=1)
    ax2.scatter(lo_med, lo_res, s=8, color="#1f6feb", alpha=0.6, label="low")
    ax2.scatter(hi_med, hi_res, s=8, color="#c1440e", alpha=0.6, label="high")
    ax2.set_xlabel("median disparity of the frame (px)")
    ax2.set_ylabel("smoothness residual (px)")
    ax2.set_title(f"Coherence vs regime (ratio {ratio:.2f})", fontsize=9)
    ax2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w4_smoothness.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/w4_smoothness.png and {HERE}/w4_smoothness.json")


if __name__ == "__main__":
    main()
