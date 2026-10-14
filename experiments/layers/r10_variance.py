# PURPOSE: Answer the owner's R10 — why is the reported height almost constant when the
#          cloud base changes by ~1450 m and the within-frame field is 1000-1400 m wide?
#
#          The paradox in numbers: the field inside one frame is broad and noisy, yet a
#          statistic extracted from it is stable to ~130 m across five hours. A broad,
#          CHANGING field should give a less stable statistic, not a more stable one.
#          So either the field's shape barely changes, or the reported statistic is
#          insensitive to the part of the field the cloud controls.
#
#          Two candidate mechanisms are tested, each with a discriminator that can
#          refute it:
#
#            M1  a persistent artefact dominates the field, so the low tail the
#                `lowest` rule selects is fixed structure rather than cloud.
#                DISCRIMINATOR: does the selected cluster's image POSITION stay in the
#                same region all session? A moving cloud must move.
#
#            M2  the reported value is an order statistic whose level is set by the
#                field's SPREAD and the MATCH COUNT, both of which are stable.
#                DISCRIMINATOR: regress the reported height on those two. If they
#                explain it, the retrieval is reporting a property of the estimator.
#
#          Ground-truth-free: the ceilometer is not read.
# INPUTS: a clusters.jsonl dump and the run's series.json.
# OUTPUTS: experiments/layers/r10_variance.json, figures, a printed verdict.
# KEYWORDS: R10, variance, order_statistic, target_identity, diagnosis
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

DUMP = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/iustimov/clouds_data/logs/20261013T143051Z_evaluate_clouds_2025-05-23_d95aa8/clusters.jsonl"
RUN = os.path.dirname(DUMP)


def main():
    frames = []
    with open(DUMP) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cl = rec.get("clusters", [])
            if not cl:
                continue
            hs = np.array([c["h"] for c in cl], float)
            sz = np.array([c["size"] for c in cl], int)
            u = np.array([c["u"] for c in cl], float)
            v = np.array([c["v"] for c in cl], float)
            # the shipped selection: size >= 8, then the lowest height
            big = sz >= 8
            if not big.any():
                continue
            idx = np.where(big)[0]
            sel = idx[np.argmin(hs[idx])]
            frames.append(dict(
                h_sel=float(hs[sel]), size_sel=int(sz[sel]),
                u_sel=float(u[sel]), v_sel=float(v[sel]),
                n_clusters=int(len(cl)), n_big=int(big.sum()),
                h_all=hs, u_all=u, v_all=v, sz_all=sz,
                # field statistics over all clusters and over the size-filtered set
                field_median=float(np.median(hs)),
                field_iqr=float(np.percentile(hs, 75) - np.percentile(hs, 25)),
                field_p10=float(np.percentile(hs, 10)),
                field_min=float(np.min(hs)),
                big_median=float(np.median(hs[big])),
                big_iqr=float(np.percentile(hs[big], 75) - np.percentile(hs[big], 25)),
                n_matches=int(sum(sz)),
            ))
    n = len(frames)
    print(f"frames {n}   (selection rule: size >= 8, then lowest)")
    sel = np.array([f["h_sel"] for f in frames])
    field = np.array([f["field_median"] for f in frames])
    fiqr = np.array([f["field_iqr"] for f in frames])
    matches = np.array([f["n_matches"] for f in frames])
    usize = np.array([f["size_sel"] for f in frames])
    print(f"\nreported height   median {np.median(sel):7.0f} m   "
          f"std {np.std(sel):6.0f} m   range {sel.min():.0f}-{sel.max():.0f}")
    print(f"field median      median {np.median(field):7.0f} m   "
          f"std {np.std(field):6.0f} m")
    print(f"field IQR         median {np.median(fiqr):7.0f} m   "
          f"std {np.std(fiqr):6.0f} m")
    print(f"size of selection median {np.median(usize):7.0f}")
    print(f"matches per frame median {np.median(matches):7.0f}")
    ratio_field = np.std(field) / max(np.std(sel), 1e-9)
    print(f"\n  the FIELD varies {ratio_field:.1f}x more than the reported value")
    print(f"  the reference moved ~1450 m over the window; the report moved "
          f"{np.percentile(sel,95)-np.percentile(sel,5):.0f} m")

    # --- M2: is the reported height explained by field spread and match count? ---
    X = np.vstack([fiqr, matches, np.ones(n)]).T
    coef, _res, _rank, _sv = np.linalg.lstsq(X, sel, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((sel - pred) ** 2) / max(np.sum((sel - sel.mean()) ** 2), 1e-9)
    print(f"\nM2: reported height ~ field IQR + match count")
    print(f"  coefficients: IQR {coef[0]:+.4f} m per m, matches {coef[1]:+.5f} m per match,"
          f" intercept {coef[2]:.0f}")
    print(f"  R^2 = {r2:.3f}  ->  "
          f"{'field statistics EXPLAIN the reported level' if r2 > 0.5 else 'they do not explain it'}")

    # --- M1: does the selected cluster stay in one place? ---
    us = np.array([f["u_sel"] for f in frames])
    vs = np.array([f["v_sel"] for f in frames])
    r_sel = np.hypot(us - CENTRE, vs - CENTRE)
    H, xe, ye = np.histogram2d(us, vs, bins=[np.arange(0, CROP + 1, 60),
                                             np.arange(0, CROP + 1, 60)])
    top = H.max() / H.sum()
    # how concentrated is the selection in image space? compare with the whole field
    uall = np.concatenate([f["u_all"] for f in frames])
    vall = np.concatenate([f["v_all"] for f in frames])
    Hall, _, _ = np.histogram2d(uall, vall, bins=[np.arange(0, CROP + 1, 60),
                                                  np.arange(0, CROP + 1, 60)])
    print(f"\nM1: does the selection stay in one image region?")
    print(f"  most-populated 60x60 bin holds {100*top:.1f} % of all frame selections")
    print(f"  selected-cluster radius: median {np.median(r_sel):.0f} px, "
          f"p10-p90 {np.percentile(r_sel,10):.0f}-{np.percentile(r_sel,90):.0f}")
    print(f"  spread of the selection in u: std {np.std(us):.0f} px, in v: {np.std(vs):.0f} px")
    conc = top / max(Hall.max() / Hall.sum(), 1e-9)
    print(f"  concentration relative to the whole field: {conc:.1f}x")

    json.dump({"n_frames": n, "reported_median": float(np.median(sel)),
               "reported_std": float(np.std(sel)),
               "reported_p5_p95": float(np.percentile(sel, 95) - np.percentile(sel, 5)),
               "field_median_std": float(np.std(field)),
               "field_iqr_median": float(np.median(fiqr)),
               "m2_r2": float(r2), "m2_coef": [float(c) for c in coef],
               "m1_top_bin_share": float(top),
               "m1_selected_radius_median": float(np.median(r_sel)),
               "m1_position_std": [float(np.std(us)), float(np.std(vs))],
               "m1_concentration_vs_field": float(conc)},
              open(os.path.join(HERE, "r10_variance.json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    ax1.hist(sel, bins=40, color="#1a7f37", alpha=0.85, label="reported (selected) height")
    ax1.hist(field, bins=40, color="#5c6672", alpha=0.55, label="field median height")
    ax1.set_xlabel("height (m)")
    ax1.set_ylabel("frames")
    ax1.set_title(f"Reported std {np.std(sel):.0f} m vs field std {np.std(field):.0f} m")
    ax1.legend(fontsize=8)
    ax2.scatter(fiqr, sel, s=7, color="#c1440e", alpha=0.6)
    ax2.set_xlabel("within-frame field IQR (m)")
    ax2.set_ylabel("reported height (m)")
    ax2.set_title(f"M2: reported height vs field spread (R$^2$={r2:.2f})")
    fig.savefig(os.path.join(OUT, "r10_variance.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/r10_variance.png and {HERE}/r10_variance.json")


if __name__ == "__main__":
    main()
