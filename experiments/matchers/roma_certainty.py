# PURPOSE: Answer precisely what "0 % interior coverage" means for RoMa.
#
#          The question matters because the two possible diagnoses lead to opposite
#          decisions. If RoMa is structurally sparse — it simply cannot produce a
#          prediction in a textureless region — it is disqualified. If RoMa IS dense
#          and its own confidence measure merely reports low certainty in those
#          regions, then the predictions exist, the filter removed them, and the
#          right response is to ask whether they are usable rather than to reject the
#          matcher.
#
#          Reading the wrapper settles the structure: `model.match()` returns a
#          (warp, certainty) field on the upsampled grid, the wrapper subsamples it
#          on a regular `grid_stride` lattice, and then keeps only points with
#          `certainty >= certainty_threshold`. Every grid point HAS a prediction.
#          So the coverage number is a property of the threshold, not of the matcher.
#
#          This script measures the certainty distribution on real stereo pairs,
#          without thresholding, stratified by image gradient, so the claim can be
#          stated as a number rather than an interpretation.
# INPUTS: real stereo pairs from 2025-05-23.
# OUTPUTS: printed distributions; experiments/matchers/roma_certainty.json.
# KEYWORDS: roma, certainty, confidence, coverage, diagnosis
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
import harness                                                             # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
OUT = os.path.join(HERE, "roma_certainty.json")


def main():
    from clouds.config import load_config
    from clouds.matching.matchers import create_matcher
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    names = sorted(x for x in os.listdir(IMAGES)
                   if x.endswith("devID1.jpg") and x.startswith("img-2025-05-23")
                   and "10-" <= x[15:18] <= "14-")[:6]

    # threshold 0 so that NOTHING is filtered: we want the raw predictions
    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")
    allc, allg = [], []
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
        k0, k1, conf = matcher(g1, g2, return_conf=True)
        if len(k0) == 0:
            continue
        _, _, grad = harness.gradient_masks(g1)
        ui = np.clip(k0[:, 0].astype(int), 0, g1.shape[1] - 1)
        vi = np.clip(k0[:, 1].astype(int), 0, g1.shape[0] - 1)
        allc.append(conf)
        allg.append(grad[vi, ui])
    conf = np.concatenate(allc)
    grad = np.concatenate(allg)
    print(f"real stereo pairs: {len(allc)} frames, {len(conf)} dense predictions")
    print("(certainty_threshold = 0, so NOTHING was filtered)\n")

    lo = grad <= np.percentile(grad, 33)
    hi = grad >= np.percentile(grad, 67)
    print(f"{'group':>22} {'n':>8} {'median':>8} {'p25':>7} {'p75':>7} {'p95':>7}")
    for name, m in (("lowest-gradient tercile", lo),
                    ("middle tercile", ~lo & ~hi),
                    ("highest-gradient tercile", hi),
                    ("all", np.ones(len(conf), bool))):
        c = conf[m]
        print(f"{name:>22} {m.sum():>8} {np.median(c):>8.4f} {np.percentile(c,25):>7.4f} "
              f"{np.percentile(c,75):>7.4f} {np.percentile(c,95):>7.4f}")

    print(f"\nfraction of predictions surviving each certainty threshold:")
    print(f"{'threshold':>10} {'interior':>10} {'edge':>8} {'all':>8}")
    table = {}
    for t in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
        fi = float((conf[lo] >= t).mean())
        fe = float((conf[hi] >= t).mean())
        fa = float((conf >= t).mean())
        table[str(t)] = {"interior": fi, "edge": fe, "all": fa}
        print(f"{t:>10.3f} {100*fi:>9.2f}% {100*fe:>7.2f}% {100*fa:>7.2f}%")

    print("\nthe pipeline's measured '0 % interior coverage' used threshold 0.5.")
    print(f"at 0.5 the interior survival rate is {100*table['0.5']['interior']:.2f} %.")
    lo_med = float(np.median(conf[lo]))
    hi_med = float(np.median(conf[hi]))
    print(f"\ninterior median certainty {lo_med:.5f} vs edge {hi_med:.5f} "
          f"(ratio {lo_med/max(hi_med,1e-9):.1f}x)")
    print("So the predictions EXIST at every grid point; RoMa is dense. What is low "
          "in cloud interiors is its own confidence, and the filter removes them.")
    json.dump({"n_predictions": int(len(conf)),
               "interior_median_certainty": lo_med,
               "edge_median_certainty": hi_med,
               "survival_by_threshold": table},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
