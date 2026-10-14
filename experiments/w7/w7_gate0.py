# PURPOSE: W7 gate 0 — run the synthetic disparity fields through the EXACT production
#          clustering and selection rule before adding any atmospheric physics.
#
#          WHY THIS GATE EXISTS. W7 validated the matcher and the triangulation against a
#          per-pixel ground truth by comparing the MEDIAN of the raw RoMa disparity field.
#          But the pipeline's delivered product is not that median: it is the `lowest`
#          cluster among HDBSCAN fragments of size >= 8 (or, now, the lowest prominent
#          mode). W3.4 already showed that the selection is precisely where the real-world
#          signal disappears - the median disparity tracks the regime at rho = +0.55 while
#          every statistic the pipeline reports is flat. A control that bypasses the
#          selection stage therefore cannot clear it.
#
#          This is the cheapest test in the project: the synthetic fields already exist and
#          only the selection has to be re-run. It asks whether a rule that picks a tail of
#          a ~9 px spread can, by itself, produce a height-dependent bias - the inversion -
#          with no atmospheric physics added at all.
#
#          Three selectors are run, all through the production code path:
#            - `lowest`      : the shipped rule, minimum height among clusters of size >= 8
#            - `largest`     : the size-maximum rule, as a control
#            - `lowest_mode` : the prominence rule that is now the default
#          and each is compared against the exact per-pixel ground truth.
# INPUTS: the measured camera model; the production clusterer and selectors.
# OUTPUTS: experiments/w7/w7_gate0.json, docs/report/w7_gate0.png.
# KEYWORDS: gate0, selection_rule, clustering, HDBSCAN, order_statistic, control
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

from w7_synthetic import (B, CROP, F_PX, ORIG, cloud_texture, gt_disparity,      # noqa: E402
                          render)


def main():
    from clouds.clustering.disparity import cluster_shifts
    from clouds.matching.matchers import create_matcher

    # the shipped clustering defaults, read from the working config
    from clouds.config import load_config
    from clouds.pipeline.analyze import lowest_prominent_mode
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    hcfg = {"min_cluster_size": cfg.clustering.min_cluster_size,
            "min_samples": cfg.clustering.min_samples,
            "allow_single_cluster": cfg.clustering.allow_single_cluster}
    print(f"production clustering config: {hcfg}")
    from clouds.geometry.triangulation import heights_from_matches

    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")
    tex = cloud_texture(1024, 90.0, seed=3)
    heights = [1000.0, 1500.0, 2000.0, 3000.0, 4000.0]
    y0 = (ORIG - CROP) // 2

    rows = []
    for H in heights:
        o1 = np.array([0.0, 0.0, 0.0]); o2 = np.array([B, 0.0, 0.0])
        g1, g2, _ = render(H, o1, o2, tex, tex_extent_m=4000.0)
        c1 = (g1[y0:y0 + CROP, y0:y0 + CROP] * 255).astype(np.uint8)
        c2 = (g2[y0:y0 + CROP, y0:y0 + CROP] * 255).astype(np.uint8)
        k0, k1 = matcher(c1, c2)
        k0 = np.asarray(k0, float).reshape(-1, 2)
        k1 = np.asarray(k1, float).reshape(-1, 2)
        shifts = k0 - k1                       # the 2-D shift vectors, as the pipeline uses
        gtd = gt_disparity(H, o1, o2)[y0:y0 + CROP, y0:y0 + CROP]
        ui = np.clip(k0[:, 0].astype(int), 0, CROP - 1)
        vi = np.clip(k0[:, 1].astype(int), 0, CROP - 1)
        truth = gtd[vi, ui]

        # the exact pipeline: cluster the 2-D shifts, then select
        centers, std_devs, labels, _ = cluster_shifts(shifts, hcfg)
        hh = heights_from_matches(k0, k1, F_PX, B,
                                 centre=(CROP / 2.0, CROP / 2.0),
                                 distortion_a=-0.30)["heights"]
        sizes = {int(l): int((labels == l).sum()) for l in np.unique(labels) if l >= 0}
        out = {}
        # A cluster's height must be taken over its FINITE members: heights_from_matches
        # returns NaN for matches whose rays do not triangulate, and a plain median then
        # propagates NaN through min(), which silently reports "no cluster" for frames
        # that in fact have hundreds. That was a real bug in the first version of this
        # gate and it produced a false failure of the selection rule.
        def cluster_h(l):
            v = hh[labels == l]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if len(v) else None

        for name in ("lowest", "largest"):
            keep = [l for l, n in sizes.items() if n >= 8]
            hvals = [(l, cluster_h(l)) for l in keep]
            hvals = [(l, h) for l, h in hvals if h is not None]
            if not hvals:
                out[name] = None
                continue
            if name == "lowest":
                out[name] = float(min(h for _l, h in hvals))
            else:
                out[name] = float(max(hvals, key=lambda t: sizes[t[0]])[1])
        hmode, _info = lowest_prominent_mode(hh[np.isfinite(hh)], None)
        # the ground truth as a HEIGHT, for comparability
        gt_h = float(np.median(F_PX * B / truth[np.isfinite(truth)]))
        row = {"H_true": H, "n_matches": int(len(k0)), "n_clusters": int(len(sizes)),
               "median_h": float(np.median(hh[np.isfinite(hh)])),
               "gt_h": gt_h,
               "lowest_h": out["lowest"], "largest_h": out["largest"],
               "mode_h": None if hmode is None else float(hmode)}
        rows.append(row)
        def f6(v):
            return "   n/a" if v is None or not np.isfinite(v) else f"{v:>6.0f}"
        print(f"H={H:>6.0f}  truth {gt_h:>6.0f} m | median {row['median_h']:>6.0f} "
              f"| lowest {f6(row['lowest_h'])} | largest {f6(row['largest_h'])} "
              f"| mode {f6(row['mode_h'])}  ({row['n_clusters']} clusters)")

    def series(key):
        return [(r[key] if r[key] is not None else np.nan) for r in rows]

    print("\nGATE 0 — does the SELECTION RULE introduce a height-dependent bias?")
    print(f"{'selector':>22} {'monotonic rho':>14} {'bias spread':>13} {'sign':>8}")
    summary = {}
    H = np.array([r["H_true"] for r in rows])
    gt = np.array([r["gt_h"] for r in rows])
    from scipy.stats import spearmanr
    for name, key in (("median (W7 original)", "median_h"),
                      ("lowest (shipped)", "lowest_h"),
                      ("largest", "largest_h"), ("lowest_mode", "mode_h")):
        v = np.array(series(key), float)
        ok = np.isfinite(v)
        if ok.sum() < 3:
            print(f"{name:>22} {'insufficient':>14}")
            continue
        rho, p = spearmanr(gt[ok], v[ok])
        bias = v[ok] - gt[ok]
        # the SIGN of the height-vs-truth relation is what an inversion means
        sign = "CORRECT" if rho > 0 else "INVERTED"
        summary[name] = {"rho": float(rho), "p": float(p),
                         "bias_min": float(bias.min()), "bias_max": float(bias.max()),
                         "spread": float(bias.max() - bias.min()), "sign": sign}
        print(f"{name:>22} {rho:>+14.3f} {bias.max()-bias.min():>12.1f} m {sign:>8}")

    inv = [k for k, v in summary.items() if v["sign"] == "INVERTED"]
    if inv:
        verdict = ("THE SELECTION RULE INVERTS THE SYNTHETIC HEIGHT — the defect is "
                   "statistical and needs no atmospheric physics: " + ", ".join(inv))
    else:
        # ordering preserved, but a large height-dependent bias is still a failure of
        # the product even when the sign is right
        worst = max(summary.items(), key=lambda kv: kv[1]["spread"])
        verdict = (f"ORDERING PRESERVED but the reported height carries a "
                   f"height-dependent bias up to {worst[1]['spread']:.0f} m "
                   f"({worst[0]}); Gate 0 passes on sign and fails on magnitude")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"rows": rows, "summary": summary, "verdict": verdict},
              open(os.path.join(HERE, "w7_gate0.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(H, gt, "o-", color="#1b2430", label="ground truth")
    for name, key, c in (("median", "median_h", "#5c6672"),
                         ("lowest (shipped)", "lowest_h", "#c1440e"),
                         ("lowest_mode", "mode_h", "#1a7f37")):
        v = np.array(series(key), float)
        ax.plot(H, v, "s--", color=c, label=name)
    ax.set_xlabel("rendered cloud height (m)"); ax.set_ylabel("reported height (m)")
    ax.set_title("Gate 0: does the selection rule preserve the ordering?", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w7_gate0.png"), bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}/w7_gate0.png and {HERE}/w7_gate0.json")


if __name__ == "__main__":
    main()
