# PURPOSE: Answer the one outstanding question before RoMa can be adopted —
#          are its LOW-certainty cloud-interior predictions accurate, or does the
#          low confidence correctly flag them as unreliable?
#
#          RoMa is dense: it emits a prediction at every grid point. On real stereo
#          pairs its certainty in cloud interiors is ~200x lower than at cloud edges,
#          and the pipeline's threshold removes them. Whether that matters depends
#          entirely on whether those discarded predictions were right. If they were,
#          the low certainty is a usable WEIGHT and the threshold was simply too
#          aggressive; if they were wrong, RoMa's confidence is a good filter and the
#          coverage problem is real.
#
#          Test: the disparity field of a cloud layer is spatially smooth. So a
#          low-certainty prediction should agree with the disparity field implied by
#          the HIGH-certainty predictions near it. The null is the same comparison with
#          the low-certainty disparities spatially shuffled, which destroys any real
#          agreement while preserving their distribution.
#
#          Ground-truth-free: no ceilometer, no synthetic data.
# INPUTS: real stereo pairs from 2025-05-23.
# OUTPUTS: experiments/matchers/roma_interior_check.json and a printed verdict.
# KEYWORDS: roma, certainty, interiors, spatial_consistency, null_test, adoption
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
import harness                                                             # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
OUT = os.path.join(HERE, "roma_interior_check.json")
N_FRAMES = 12
K_NEIGHBOURS = 25


def main():
    from clouds.config import load_config
    from clouds.matching.matchers import create_matcher
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    names = sorted(x for x in os.listdir(IMAGES)
                   if x.endswith("devID1.jpg") and x.startswith("img-2025-05-23")
                   and "10-" <= x[15:18] <= "14-")[:N_FRAMES]

    # threshold 0: keep everything, so we can judge the low-certainty set ourselves
    roma = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0}, "cuda")
    loftr = create_matcher("kornia_loftr", {}, "cuda")

    rng = np.random.default_rng(11)
    rows = []
    for nm in names:
        base = nm.replace("devID1.jpg", "")
        i1 = cv2.imread(os.path.join(IMAGES, nm))
        i2 = cv2.imread(os.path.join(IMAGES, base + "devID2.jpg"))
        if i1 is None or i2 is None:
            continue
        w1 = cv2.warpAffine(i1, M, dsize=cfg.camera.original_size)
        y0 = (w1.shape[0] - crop) // 2
        g1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY)[y0:y0 + crop, y0:y0 + crop]
        g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)[y0:y0 + crop, y0:y0 + crop]
        k0, k1, conf = roma(g1, g2, return_conf=True)
        if len(k0) < 500:
            continue
        d = k0[:, 0] - k1[:, 0]                      # horizontal disparity
        _, _, grad = harness.gradient_masks(g1)
        ui = np.clip(k0[:, 0].astype(int), 0, g1.shape[1] - 1)
        vi = np.clip(k0[:, 1].astype(int), 0, g1.shape[0] - 1)
        pt_grad = grad[vi, ui]
        interior = pt_grad <= np.percentile(grad, 33)

        hi = (conf >= 0.5)
        lo = interior & (conf < 0.5)
        if hi.sum() < 100 or lo.sum() < 100:
            continue
        tree = cKDTree(k0[hi])
        # for each low-certainty interior point, the local disparity of the
        # high-certainty predictions around it
        dist, idx = tree.query(k0[lo], k=K_NEIGHBOURS)
        local = np.median(d[hi][idx], axis=1)
        resid = np.abs(d[lo] - local)

        # null: same comparison, low-certainty disparities shuffled in space
        perm = rng.permutation(len(local))
        resid_null = np.abs(d[lo][perm] - local)

        # independent cross-check: where LoFTR also has interior matches
        lk0, lk1 = loftr(g1, g2)
        lk0 = np.asarray(lk0, float).reshape(-1, 2)
        lk1 = np.asarray(lk1, float).reshape(-1, 2)
        xmatch = float("nan")
        n_x = 0
        if len(lk0):
            lui = np.clip(lk0[:, 0].astype(int), 0, g1.shape[1] - 1)
            lvi = np.clip(lk0[:, 1].astype(int), 0, g1.shape[0] - 1)
            l_int = grad[lvi, lui] <= np.percentile(grad, 33)
            if l_int.sum() >= 20:
                d_l = lk0[:, 0] - lk1[:, 0]
                # for each LoFTR interior match, RoMa's nearest interior prediction
                lo_tree = cKDTree(k0[lo])
                dd, ii = lo_tree.query(lk0[l_int], k=1)
                near = dd < 8.0
                if near.sum() >= 20:
                    xmatch = float(np.median(np.abs(d[lo][ii[near]] - d_l[l_int][near])))
                    n_x = int(near.sum())
        rows.append({"frame": base,
                     "n_high": int(hi.sum()), "n_low_interior": int(lo.sum()),
                     "median_certainty_high": float(np.median(conf[hi])),
                     "median_certainty_low": float(np.median(conf[lo])),
                     "median_resid_px": float(np.median(resid)),
                     "median_resid_null_px": float(np.median(resid_null)),
                     "cross_matcher_resid_px": xmatch, "cross_matcher_n": n_x})
        print(f"  {base}  high={hi.sum():5d} low_int={lo.sum():5d}  "
              f"resid={np.median(resid):6.2f} px  null={np.median(resid_null):6.2f} px"
              + (f"  vs LoFTR={xmatch:5.2f} px (n={n_x})" if n_x else ""))

    if not rows:
        print("no usable frames")
        return
    r = np.array([x["median_resid_px"] for x in rows])
    rn = np.array([x["median_resid_null_px"] for x in rows])
    xr = np.array([x["cross_matcher_resid_px"] for x in rows if np.isfinite(x["cross_matcher_resid_px"])])
    print(f"\nframes {len(rows)}")
    print(f"  low-certainty interior predictions vs the local high-certainty field:")
    print(f"    median |residual|      {np.median(r):6.2f} px")
    print(f"    same, spatially shuffled (null) {np.median(rn):6.2f} px")
    if len(xr):
        print(f"  vs LoFTR's own interior matches: {np.median(xr):.2f} px "
              f"({len(xr)} frames)")
    base_disp = float(np.median([x["n_high"] for x in rows]))
    print(f"\n  for scale: the working disparity is ~15-27 px, and 1 px is 5.5 % of")
    print(f"  height at 1.5 km.")
    verdict = ("USABLE - they agree with the reliable field"
               if np.median(r) < 1.0 and np.median(r) < 0.6 * np.median(rn)
               else "NOT usable - low certainty correctly flags them")
    print(f"\n  VERDICT: {verdict}")
    json.dump({"rows": rows, "median_resid_px": float(np.median(r)),
               "median_resid_null_px": float(np.median(rn)),
               "cross_matcher_resid_px": (float(np.median(xr)) if len(xr) else None),
               "verdict": verdict}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
