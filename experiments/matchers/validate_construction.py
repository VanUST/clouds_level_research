# PURPOSE: Validate the ground-truth construction used by subpixel_accuracy.py before
#          any matcher ranking is believed.
#
#          That test translates a real cloud image by a known sub-pixel amount with
#          cv2.warpAffine and asks how well a matcher recovers it. Two things were
#          wrong with the first results and neither was a matcher: the recovered
#          displacement was ~0.9x the applied one at large shifts, and MASt3R
#          returned exactly 0.000 for every shift including 2.45 px. If the
#          TRANSLATION ITSELF cannot be recovered by an exact method, the test is
#          measuring its own construction and the matcher ranking from it is void.
#
#          This checks the construction with phase correlation, which is a direct
#          sub-pixel estimator with no learned prior, and additionally verifies that
#          the interpolation does not move the image's intensity centroid.
# INPUTS: real cloud crops from 2025-05-23.
# OUTPUTS: printed recovery table; experiments/matchers/construction_check.json.
# KEYWORDS: validation, phase_correlation, sub_pixel, ground_truth, methodology
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
IMAGES = "/home/iustimov/clouds_data/images"
SHIFTS = [0.15, 0.35, 0.5, 0.65, 0.85, 1.35, 2.45, 5.0]


def load(n=8):
    from clouds.config import load_config
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    names = sorted(x for x in os.listdir(IMAGES)
                   if x.endswith("devID1.jpg") and x.startswith("img-2025-05-23")
                   and "10-" <= x[15:18] <= "14-")
    out = []
    for nm in names[::max(1, len(names) // (n * 6))]:
        i1 = cv2.imread(os.path.join(IMAGES, nm))
        if i1 is None:
            continue
        w = cv2.warpAffine(i1, M, dsize=cfg.camera.original_size)
        g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        y0 = (g.shape[0] - crop) // 2
        c = g[y0:y0 + crop, y0:y0 + crop]
        if c.std() < 8:
            continue
        out.append(c)
        if len(out) >= n:
            break
    return out


def translate(g, dx):
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], float)
    return cv2.warpAffine(g, M, (g.shape[1], g.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def main():
    frames = load()
    print(f"frames: {len(frames)}\n")
    print("Phase correlation (an exact sub-pixel estimator) recovering a known shift.")
    print("If it cannot, the translation construction is the problem, not the matcher.\n")
    print(f"{'applied':>9} {'recovered':>11} {'error':>8} {'ratio':>7} "
          f"{'centroid':>10}")
    rows = []
    for s in SHIFTS:
        rec, errs, cen = [], [], []
        for g in frames:
            g2 = translate(g, s)
            # phase correlation returns the shift that maps g2 onto g
            (dx, dy), _resp = cv2.phaseCorrelate(np.float32(g), np.float32(g2))
            rec.append(dx)
            # where did the intensity centroid move? a pure translation must move it
            # by exactly the applied amount
            m1 = float(np.sum(g * np.arange(g.shape[1])[None, :]) / np.sum(g))
            m2 = float(np.sum(g2 * np.arange(g2.shape[1])[None, :]) / np.sum(g2))
            cen.append(m2 - m1)
        rec = float(np.median(rec))
        cen = float(np.median(cen))
        rows.append({"applied": s, "phase_recovered": rec, "centroid_shift": cen})
        print(f"{s:>9.2f} {rec:>11.3f} {rec + s:>8.3f} {(-rec / s if s else 0):>7.3f} "
              f"{cen:>10.3f}")
    print("\n(phase correlation returns the shift that maps image2 back onto image1,")
    print(" so a translation of +s in image2 should recover -s.)")
    json.dump(rows, open(os.path.join(HERE, "construction_check.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'construction_check.json')}")


if __name__ == "__main__":
    main()
