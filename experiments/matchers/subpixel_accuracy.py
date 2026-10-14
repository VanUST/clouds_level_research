# PURPOSE: Measure each candidate matcher's TRUE sub-pixel localisation accuracy on
#          real cloud imagery, by ground-truth construction.
#
#          The previous evaluation judged matchers by geometric residuals — how well
#          their correspondences satisfy a plane model — which conflates localisation
#          error with un-modelled geometry. The radial-slope result (every candidate
#          gave -2.3 to -3.4 m/px) showed that most of the residual is geometry, not
#          matching, so residuals cannot rank matchers by pixel accuracy.
#
#          This measures it directly. Take a real cloud image, translate it by a
#          KNOWN sub-pixel amount with high-quality interpolation, run the matcher
#          between the original and the translated copy, and compare the recovered
#          displacement with the applied one. The reference is exact and known, the
#          texture is real cloud, and the answer is in pixels.
#
#          It is also stratified by image gradient, because that is where the earlier
#          evaluation found the decisive difference: RoMa and DKM are most accurate
#          on the matches they return, but return none in cloud interiors.
#
#          Limits, stated: interpolation blurs the translated copy, which is a fair
#          proxy for a second lens but is not identical to it; and a global
#          translation is not a stereo disparity field. This measures LOCALISATION
#          accuracy, which is what it claims to measure.
# INPUTS: real 640x640 cloud crops from 2025-05-23.
# OUTPUTS: experiments/matchers/subpixel_accuracy.json and a printed ranking.
# KEYWORDS: sub_pixel, accuracy, localisation, ground_truth, matcher_comparison
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from clouds.matching.matchers import create_matcher                        # noqa: E402
import harness                                                             # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
OUT = os.path.join(HERE, "subpixel_accuracy.json")

# Shifts chosen to probe sub-pixel behaviour: the magnitudes where a discrete
# matcher either resolves the fraction or snaps to an integer.
SHIFTS = [0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.35, 2.45]
N_FRAMES = 14

# The backend configurations the earlier evaluation used, so this is a like-for-like
# comparison rather than a new set of tuned parameters.
CANDIDATES = [
    ("kornia_loftr", "kornia LoFTR (default)", {}),
    ("mast3r", "MASt3R conf>=0.5", {"min_conf": 0.5}),
    ("roma", "RoMa", {"grid_stride": 8, "certainty_threshold": 0.5}),
    ("dkm", "DKM", {"grid_stride": 8, "certainty_threshold": 0.5}),
    ("efficient_loftr", "EfficientLoFTR", {}),
]


# PURPOSE: Collect real cloud crops with enough texture to match.
# INPUTS: number of frames wanted.
# OUTPUTS: list of (name, gray_uint8 640x640).
# KEYWORDS: data, frames
def load_frames(n):
    from clouds.config import load_config
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    names = sorted(x for x in os.listdir(IMAGES)
                   if x.endswith("devID1.jpg") and x.startswith("img-2025-05-23")
                   and "10-" <= x[15:18] <= "14-")
    out = []
    for nm in names[::max(1, len(names) // (n * 4))]:
        base = nm.replace("devID1.jpg", "")
        i1 = cv2.imread(os.path.join(IMAGES, nm))
        if i1 is None:
            continue
        w = cv2.warpAffine(i1, M, dsize=cfg.camera.original_size)
        g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        y0 = (g.shape[0] - crop) // 2
        c = g[y0:y0 + crop, y0:y0 + crop]
        if c.std() < 8:          # skip featureless frames; they cannot test accuracy
            continue
        out.append((base, c))
        if len(out) >= n:
            break
    return out


# PURPOSE: Translate an image by a known sub-pixel amount.
# INPUTS: gray (2-D uint8), dx, dy (floats).
# OUTPUTS: translated image, same size, no wraparound (border replicated).
# KEYWORDS: shift, ground_truth, interpolation
def translate(gray, dx, dy):
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], float)
    return cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def main():
    frames = load_frames(N_FRAMES)
    print(f"frames: {len(frames)}  (real 640x640 cloud crops, 2025-05-23)\n")
    if not frames:
        return
    # gradient mask from the first frame, used to stratify
    lo_mask, hi_mask, grad = harness.gradient_masks(frames[0][1])
    print(f"gradient terciles: lowest tertile covers {100*lo_mask.mean():.0f} % of the field\n")

    results = {}
    for backend, label, cfg in CANDIDATES:
        try:
            matcher = create_matcher(backend, cfg, "cuda")
        except Exception as exc:
            print(f"  {label}: could not load ({exc})")
            continue
        rows = []
        t0 = time.time()
        for name, g in frames:
            for s in SHIFTS:
                g2 = translate(g, s, 0.0)
                try:
                    res = matcher(g, g2, return_conf=True)
                except TypeError:
                    res = matcher(g, g2)
                kp1 = np.asarray(res[0], float).reshape(-1, 2)
                kp2 = np.asarray(res[1], float).reshape(-1, 2)
                if len(kp1) < 8:
                    continue
                # recovered displacement: an image translated by +s has its features
                # at u - s, so (u1 - u2) should equal s
                d = kp1[:, 0] - kp2[:, 0]
                ui = np.clip(kp1[:, 0].astype(int), 0, g.shape[1] - 1)
                vi = np.clip(kp1[:, 1].astype(int), 0, g.shape[0] - 1)
                in_lo = lo_mask[vi, ui]
                # robust centre and spread; the median is the localisation estimate
                med = float(np.median(d))
                rows.append({"frame": name, "shift": s, "n": int(len(d)),
                             "median": med, "err": med - s,
                             "iqr": float(np.percentile(d, 75) - np.percentile(d, 25)),
                             "mad": float(np.median(np.abs(d - med))),
                             "n_lo": int(in_lo.sum()),
                             "err_lo": (float(np.median(d[in_lo]) - s)
                                        if in_lo.sum() >= 5 else None),
                             "iqr_lo": (float(np.percentile(d[in_lo], 75)
                                              - np.percentile(d[in_lo], 25))
                                        if in_lo.sum() >= 5 else None)})
        dt = time.time() - t0
        if not rows:
            continue
        # a matcher that snaps to integers shows a periodic error: |err| is small at
        # integer shifts and large at half-integer shifts. Compare the two.
        int_err = [abs(r["err"]) for r in rows if abs(r["shift"] - round(r["shift"])) < 0.01]
        half_err = [abs(r["err"]) for r in rows
                    if 0.25 <= abs(r["shift"] - round(r["shift"])) <= 0.75]
        sub = [r for r in rows if r["shift"] > 0]
        lo_errors = [abs(r["err_lo"]) for r in rows if r["err_lo"] is not None]
        lo_iqr = [r["iqr_lo"] for r in rows if r["iqr_lo"] is not None]
        results[label] = {
            "backend": backend, "cfg": cfg,
            "overall_bias_abs": float(np.median([abs(r["err"]) for r in rows])),
            "subpixel_bias_abs": float(np.median([abs(r["err"]) for r in sub])),
            "interior_bias_abs": float(np.median(lo_errors)) if lo_errors else None,
            "overall_iqr": float(np.median([r["iqr"] for r in rows])),
            "interior_iqr": float(np.median(lo_iqr)) if lo_iqr else None,
            "interior_share": float(np.mean([r["n_lo"] / max(r["n"], 1) for r in rows])),
            "integer_shift_err": float(np.median(int_err)) if int_err else None,
            "half_integer_shift_err": float(np.median(half_err)) if half_err else None,
            "median_matches": float(np.median([r["n"] for r in rows])),
            "seconds": dt, "rows": rows}

    print(f"{'matcher':>24} {'bias px':>8} {'subpix':>7} {'interior':>9} "
          f"{'IQR px':>7} {'int IQR':>8} {'int %':>6} {'matches':>8}")
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1]["subpixel_bias_abs"]
                                    + (kv[1]["interior_bias_abs"] or 9)))
    for label, r in ranked:
        print(f"{label:>24} {r['overall_bias_abs']:>8.3f} {r['subpixel_bias_abs']:>7.3f} "
              f"{(r['interior_bias_abs'] if r['interior_bias_abs'] is not None else float('nan')):>9.3f} "
              f"{r['overall_iqr']:>7.3f} "
              f"{(r['interior_iqr'] if r['interior_iqr'] is not None else float('nan')):>8.3f} "
              f"{100*r['interior_share']:>5.1f}% {r['median_matches']:>8.0f}")
    print("\n'bias px' is the median |recovered - applied| shift over all matches;")
    print("'subpix' keeps only non-integer shifts, where a matcher that snaps to the")
    print("pixel grid gives its error away; 'interior' restricts to the lowest-gradient")
    print("tercile, which is where cloud-base retrieval actually has to work.")
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
