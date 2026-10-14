# PURPOSE: The W7 synthetic control, run on a renderer that has passed its gate.
#
#          The renderer (w7_renderer.py) is validated first: du(zenith) must equal
#          f*atan(B/H) and fall as 1/H. Only if that holds is the matcher tested, because
#          a control whose ground truth is wrong cannot clear or convict anything - which
#          is exactly how the first W7 attempt failed.
#
#          Two questions are asked, in order:
#            1. Does RoMa + exact triangulation recover the known height? (the pipeline's
#               front end, per-pixel)
#            2. Does the production CLUSTERING + SELECTION preserve that? (the pipeline's
#               back end, which the first attempt never exercised)
# INPUTS: the validated renderer.
# OUTPUTS: experiments/w7/w7_control.json, docs/report/w7_control.png.
# KEYWORDS: W7, synthetic_control, matcher, selection, validated
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import w7_renderer as R                                                       # noqa: E402
from w7_synthetic import cloud_texture                                        # noqa: E402


def main():
    from clouds.config import load_config
    from clouds.clustering.disparity import cluster_shifts
    from clouds.geometry.triangulation import heights_from_matches
    from clouds.matching.matchers import create_matcher
    from clouds.pipeline.analyze import lowest_prominent_mode

    if not R.validate():
        print("RENDERER GATE FAILED - no result may be reported")
        return

    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    hcfg = {"min_cluster_size": cfg.clustering.min_cluster_size,
            "min_samples": cfg.clustering.min_samples,
            "allow_single_cluster": cfg.clustering.allow_single_cluster}
    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")
    tex = cloud_texture(1024, 90.0, seed=3)
    y0 = (R.ORIG - R.CROP) // 2
    heights = [1000.0, 1500.0, 2000.0, 3000.0, 4000.0]

    print(f"\n{'H_true':>7} {'gt du':>8} {'RoMa du':>9} {'bias px':>8} {'IQR':>7} "
          f"{'median H':>9} {'lowest H':>9} {'mode H':>8}")
    rows = []
    for H in heights:
        g1, g2 = R.render_pair(H, tex, tex_extent_m=4000.0)
        gtd, gtdv = R.gt_disparity(H)
        c1 = (g1[y0:y0 + R.CROP, y0:y0 + R.CROP] * 255).astype(np.uint8)
        c2 = (g2[y0:y0 + R.CROP, y0:y0 + R.CROP] * 255).astype(np.uint8)
        k0, k1 = matcher(c1, c2)
        k0 = np.asarray(k0, float).reshape(-1, 2)
        k1 = np.asarray(k1, float).reshape(-1, 2)
        rec = k0[:, 0] - k1[:, 0]
        ui = np.clip(k0[:, 0].astype(int), 0, R.CROP - 1)
        vi = np.clip(k0[:, 1].astype(int), 0, R.CROP - 1)
        truth = gtd[y0:y0 + R.CROP, y0:y0 + R.CROP][vi, ui]
        ok = np.isfinite(truth)
        bias = float(np.median(rec[ok] - truth[ok]))
        # --- the pipeline's back end ---
        hh = heights_from_matches(k0, k1, R.F_PX, R.B_M,
                                  centre=(R.CROP / 2.0, R.CROP / 2.0),
                                  distortion_a=R.A_DIST)["heights"]
        hf = hh[np.isfinite(hh)]
        shifts = k0 - k1
        _c, _s, labels, _col = cluster_shifts(shifts, hcfg)
        sizes = {int(l): int((labels == l).sum()) for l in np.unique(labels) if l >= 0}

        def cluster_h(l):
            v = hh[labels == l]; v = v[np.isfinite(v)]
            return float(np.median(v)) if len(v) else None

        keep = [(l, cluster_h(l)) for l, n in sizes.items() if n >= 8]
        keep = [(l, h) for l, h in keep if h is not None]
        lowest_h = float(min(h for _l, h in keep)) if keep else None
        hmode, _i = lowest_prominent_mode(hf, None)
        row = {"H_true": H, "gt_du": float(np.median(truth[ok])),
               "roma_du": float(np.median(rec[ok])), "bias_px": bias,
               "iqr": float(np.percentile(rec[ok], 75) - np.percentile(rec[ok], 25)),
               "n": int(ok.sum()), "median_h": float(np.median(hf)),
               "lowest_h": lowest_h, "mode_h": None if hmode is None else float(hmode),
               "n_clusters": len(sizes)}
        rows.append(row)

        def f6(v):
            return "    n/a" if v is None or not np.isfinite(v) else f"{v:>7.0f}"
        print(f"{H:>7.0f} {row['gt_du']:>8.2f} {row['roma_du']:>9.2f} {bias:>+8.3f} "
              f"{row['iqr']:>7.2f} {row['median_h']:>9.0f} {f6(row['lowest_h'])} "
              f"{f6(row['mode_h'])}")

    from scipy.stats import spearmanr
    H = np.array([r["H_true"] for r in rows])
    gtdu = np.array([r["gt_du"] for r in rows])
    rcdu = np.array([r["roma_du"] for r in rows])
    rho, p = spearmanr(gtdu, rcdu)
    print(f"\n1) MATCHER: rho(gt du, RoMa du) = {rho:+.3f} (p = {p:.3f})")
    print(f"   median |bias| = {np.median([abs(r['bias_px']) for r in rows]):.3f} px")
    for key, nm in (("median_h", "median"), ("lowest_h", "lowest"), ("mode_h", "mode")):
        v = np.array([r[key] if r[key] is not None else np.nan for r in rows], float)
        m = np.isfinite(v)
        if m.sum() >= 3:
            r2, p2 = spearmanr(H[m], v[m])
            er = v[m] - H[m]
            print(f"2) {nm:>7}: rho(true, reported) = {r2:+.3f} (p = {p2:.3f}), "
                  f"error {er.min():+.0f} to {er.max():+.0f} m")

    verdict = ("MATCHER RECOVERS THE HEIGHT" if rho > 0.9 else
               "MATCHER DOES NOT TRACK THE HEIGHT")
    print(f"\nVERDICT: {verdict}")
    json.dump({"rows": rows, "matcher_rho": float(rho), "matcher_p": float(p),
               "verdict": verdict}, open(os.path.join(HERE, "w7_control.json"), "w"),
              indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    a1.plot(H, gtdu, "o-", color="#1b2430", label="ground truth")
    a1.plot(H, rcdu, "s--", color="#c1440e", label="RoMa")
    a1.set_xlabel("rendered height (m)"); a1.set_ylabel("median disparity (px)")
    a1.set_title(f"Matcher vs known truth (rho = {rho:+.2f})", fontsize=9); a1.legend(fontsize=8)
    for key, c, lab in (("median_h", "#5c6672", "median"), ("lowest_h", "#c1440e", "lowest"),
                        ("mode_h", "#1a7f37", "lowest_mode")):
        v = np.array([r[key] if r[key] is not None else np.nan for r in rows], float)
        a2.plot(H, v, "s--", color=c, label=lab)
    a2.plot(H, H, "o-", color="#1b2430", label="truth")
    a2.set_xlabel("rendered height (m)"); a2.set_ylabel("reported height (m)")
    a2.set_title("Selection rules vs truth", fontsize=9); a2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w7_control.png"), bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}/w7_control.png and {HERE}/w7_control.json")


if __name__ == "__main__":
    main()
