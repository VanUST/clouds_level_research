# PURPOSE: Decide the matcher question with the measurement that actually matches the
#          task, rather than with absolute small shifts.
#
#          The height comes from the disparity, and at 1.5 km the disparity is 18.3 px.
#          One pixel of disparity error is 5.5 % of height there. So what the pipeline
#          needs is not the ability to recover a 0.5 px ABSOLUTE displacement — it is
#          the ability to resolve a 0.5 px CHANGE riding on top of a ~20 px disparity.
#          Those are different properties, and every ranking so far has measured the
#          first one while the pipeline consumes the second.
#
#          Method: build image B by translating image A by a large base shift plus a
#          small offset. Sweep the offset at fixed base, and measure how the recovered
#          displacement responds. The slope of that response is the differential
#          sensitivity (1.0 is perfect); its scatter about the fitted line is the
#          precision. The construction is the same one already validated against phase
#          correlation in validate_construction.py.
# INPUTS: real cloud crops from 2025-05-23.
# OUTPUTS: experiments/matchers/differential_sensitivity.json and a printed table.
# KEYWORDS: differential_accuracy, disparity, sub_pixel, matcher_selection, decision
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
OUT = os.path.join(HERE, "differential_sensitivity.json")

# 20 px is a representative working disparity (about 1370 m at f = 611, B = 45).
# The offsets are the height-relevant increments: 0.5 px is the whole accuracy
# budget at 1.5 km, and 1 px is 5.5 % of height.
BASE = 20.0
OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
N_FRAMES = 12

CANDIDATES = [
    ("kornia_loftr", "kornia LoFTR (default)", {}),
    ("mast3r", "MASt3R conf>=0.5", {"min_conf": 0.5}),
    ("roma", "RoMa (cert 0.5)", {"grid_stride": 8, "certainty_threshold": 0.5}),
    ("roma", "RoMa (cert 0.05, interiors kept)", {"grid_stride": 8,
                                                  "certainty_threshold": 0.05}),
    ("dkm", "DKM", {"grid_stride": 8, "certainty_threshold": 0.5}),
    ("efficient_loftr", "EfficientLoFTR", {}),
]


def load_frames(n):
    from clouds.config import load_config
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    names = sorted(x for x in os.listdir(IMAGES)
                   if x.endswith("devID1.jpg") and x.startswith("img-2025-05-23")
                   and "10-" <= x[15:18] <= "14-")
    out = []
    for nm in names[::max(1, len(names) // (n * 5))]:
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
    frames = load_frames(N_FRAMES)
    lo_mask, _, _ = harness.gradient_masks(frames[0])
    print(f"frames {len(frames)}; base disparity {BASE} px; offsets {OFFSETS} px")
    print("A perfect matcher has slope 1.00 and scatter 0.00.\n")
    print(f"{'matcher':>34} {'slope':>7} {'scatter':>8} {'interior slope':>15} "
          f"{'int share':>10} {'n':>7} {'ms':>6}")
    results = {}
    for backend, label, cfg in CANDIDATES:
        try:
            matcher = create_matcher(backend, cfg, "cuda")
        except Exception as exc:
            print(f"{label:>34}  could not load: {exc}")
            continue
        t0 = time.time()
        # per frame: the median displacement at each offset
        curves, curves_lo, shares = [], [], []
        for g in frames:
            ds, ds_lo, share = [], [], []
            for off in OFFSETS:
                g2 = translate(g, BASE + off)
                try:
                    res = matcher(g, g2, return_conf=True)
                except TypeError:
                    res = matcher(g, g2)
                k1 = np.asarray(res[0], float).reshape(-1, 2)
                k2 = np.asarray(res[1], float).reshape(-1, 2)
                if len(k1) < 8:
                    ds.append(np.nan); ds_lo.append(np.nan); share.append(np.nan)
                    continue
                d = k1[:, 0] - k2[:, 0]
                ui = np.clip(k1[:, 0].astype(int), 0, g.shape[1] - 1)
                vi = np.clip(k1[:, 1].astype(int), 0, g.shape[0] - 1)
                inlo = lo_mask[vi, ui]
                ds.append(float(np.median(d)))
                ds_lo.append(float(np.median(d[inlo])) if inlo.sum() >= 5 else np.nan)
                share.append(float(inlo.mean()))
            if np.all(np.isfinite(ds)):
                curves.append(ds)
            if np.all(np.isfinite(ds_lo)):
                curves_lo.append(ds_lo)
            shares.append(float(np.nanmean(share)))
        dt = time.time() - t0
        curves = np.array(curves)
        if len(curves) < 3:
            print(f"{label:>34}  insufficient matches on real frames")
            continue
        offs = np.array(OFFSETS)
        # differential response: slope of recovered displacement vs applied offset
        slopes = []
        for c in curves:
            sl = np.polyfit(offs, c, 1)[0]
            slopes.append(sl)
        slope = float(np.median(slopes))
        # scatter about the per-frame fitted line, in px
        scat = float(np.median([np.std(c - np.polyval(np.polyfit(offs, c, 1), offs))
                                for c in curves]))
        slope_lo = float("nan")
        if len(curves_lo) >= 3:
            slope_lo = float(np.median([np.polyfit(offs, c, 1)[0] for c in curves_lo]))
        results[label] = {"backend": backend, "cfg": cfg, "slope": slope,
                          "scatter_px": scat, "interior_slope": slope_lo,
                          "interior_share": float(np.nanmean(shares)),
                          "n_curves": len(curves), "ms": dt / max(len(frames), 1)}
        print(f"{label:>34} {slope:>7.3f} {scat:>8.3f} {slope_lo:>15.3f} "
              f"{100*np.nanmean(shares):>9.1f}% {len(curves):>7} "
              f"{dt/max(len(frames),1):>6.0f}")
    json.dump(results, open(OUT, "w"), indent=1, default=float)
    print(f"\n'interslope' uses only the lowest-gradient tercile.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
