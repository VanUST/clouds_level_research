# PURPOSE: W8 injection 4/5 — photometric superposition. Does alpha-blending a synthetic
#          cloud over a REAL clear-sky frame disperse the matched disparity field the way
#          the real high regime does?
#
#          WHY THIS INJECTION, AND WHY NOW. Testing the sky-hallucination route on real data
#          produced two facts. A DISCRETE sky mode is refuted: the sky-like population is
#          2.6-3.6 % of matches, against lowest_mode's 10 % prominence floor, and cannot
#          move a median by the observed 1.80 px. But the real high regime's disparity
#          field is far more dispersed than the low regime's - its 1st-to-99th percentile
#          span goes from 37.0 to 81.7 px, a factor of 2.2, while its interquartile range
#          moves only 4.11 to 4.56 px.
#
#          That combination - explosive tails, static core - is what a CONTINUUM of mixed
#          pixels produces, as distinct from a second population. Alpha blending is the
#          mechanism that creates such a continuum: partial cloud cover within a receptive
#          field yields intermediate appearances, and a matcher can place those at
#          intermediate disparities, filling the range between the cloud and the sky behind
#          it.
#
#          The sky layer is a REAL 2025-05-22 frame, processed exactly as the pipeline
#          processes its input. The hallucination is a property of that specific imagery,
#          so a procedural gradient would be a guess at the thing under test.
#
#          THE PRE-REGISTERED TRIGGER is tail growth at a fixed core: the 1-99 span must
#          grow by a factor of at least 1.8 while the IQR grows by less than 40 %. That is
#          the real signature; an injection that merely widens the IQR reproduces a
#          different phenomenon and does not count.
# INPUTS: the validated renderer (w7_renderer.py) and a real cloudless pair.
# OUTPUTS: experiments/w8/w8_superposition.json, docs/report/w8_superposition.png.
# KEYWORDS: W8, superposition, alpha_blend, mixed_pixel, tails, dispersion, injection
from __future__ import annotations

import json
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
IMAGES = "/home/iustimov/clouds_data/images"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "w7"))

import w7_renderer as R                                                       # noqa: E402
from w7_synthetic import cloud_texture                                        # noqa: E402


# PURPOSE: A real cloudless pair, processed exactly as the pipeline processes its input.
# INPUTS: a frame index into the 2025-05-22 archive.
# OUTPUTS: (img1_warped, img2, g1, g2) grayscale crops of the pipeline's shape.
#          These are the sky layers: using a real frame keeps the clear-sky response
#          identical to the one measured in the cloudless null (11,653 matches at 44.91 px)
#          rather than approximating it.
# KEYWORDS: sky_layer, empirical, clear_sky
def sky_layers(k=0, roll=1.5):
    from clouds.config import load_config
    from clouds.geometry.imaging import center_crop, rotation_about
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = R._affine()
    orig = cfg.camera.original_size
    cxy = (orig[0] / 2.0, orig[1] / 2.0)
    names = sorted(f for f in os.listdir(IMAGES)
                   if f.startswith("img-2025-05-22T12-") and f.endswith("devID1.jpg"))
    nm = names[k % len(names)]
    base = nm.replace("devID1.jpg", "")
    i1 = cv2.imread(os.path.join(IMAGES, nm), cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(os.path.join(IMAGES, base + "devID2.jpg"), cv2.IMREAD_GRAYSCALE)
    i2 = cv2.warpAffine(i2, rotation_about(cxy, roll), dsize=orig)
    w1 = cv2.warpAffine(i1, M, dsize=orig)
    g1 = center_crop(w1, R.CROP, R.CROP).astype(np.float32) / 255.0
    g2 = center_crop(i2, R.CROP, R.CROP).astype(np.float32) / 255.0
    return g1, g2, base


def cloud_layers(H, tex, extent_m, roll=0.0):
    """The synthetic cloud pair in the pipeline's convention, cropped.

    roll MUST be 0.0 here. The synthetic rig carries no residual roll, so rendering it
    with one and never correcting it models a different instrument than the control does -
    and that single default cost 551 m and one debugging cycle. The REAL sky layer, by
    contrast, does carry the measured residual and is corrected with the pipeline's own
    +1.5 deg, so both layers end in the same frame. w7_control calls render_pair without
    roll_deg and therefore gets 0.0; this must match it.
    """
    g1w, g2 = R.render_pair(H, tex, tex_extent_m=extent_m, roll_deg=roll)
    y0 = (R.ORIG - R.CROP) // 2
    c1 = g1w[y0:y0 + R.CROP, y0:y0 + R.CROP]
    c2 = g2[y0:y0 + R.CROP, y0:y0 + R.CROP]
    gt, _ = R.gt_disparity(H, roll_deg=roll)
    return c1, c2, gt[y0:y0 + R.CROP, y0:y0 + R.CROP]


def main():
    from clouds.geometry.triangulation import heights_from_matches
    from clouds.matching.matchers import create_matcher
    from clouds.pipeline.analyze import lowest_prominent_mode

    if not R.validate():
        print("RENDERER GATE FAILED")
        return
    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")
    tex = cloud_texture(1024, 90.0, seed=3)
    s1, s2, sky_name = sky_layers()
    print(f"sky layer: {sky_name} (a real cloudless frame, pipeline-processed)")
    print(f"  sky layer alone: mean {s1.mean():.3f}")

    rows = []
    print(f"\n{'H':>6} {'alpha':>6} {'median du':>10} {'IQR':>7} {'1-99 span':>10} "
          f"{'mode H':>8} {'n':>7}")
    for H in (2000.0, 4000.0):
        c1, c2, gt = cloud_layers(H, tex, 4000.0)
        # the sky layers are at a different exposure; match their level to the cloud's so
        # the blend varies only the mixing fraction and not the overall brightness
        s1n = s1 - s1.mean() + c1.mean()
        s2n = s2 - s2.mean() + c2.mean()
        for alpha in (1.0, 0.8, 0.6, 0.4, 0.2):
            m1 = alpha * c1 + (1 - alpha) * s1n
            m2 = alpha * c2 + (1 - alpha) * s2n
            i1 = (np.clip(m1, 0, 1) * 255).astype(np.uint8)
            i2 = (np.clip(m2, 0, 1) * 255).astype(np.uint8)
            k0, k1 = matcher(i1, i2)
            k0 = np.asarray(k0, float).reshape(-1, 2)
            k1 = np.asarray(k1, float).reshape(-1, 2)
            if len(k0) < 100:
                print(f"{H:>6.0f} {alpha:>6.2f}  too few matches")
                continue
            du = k0[:, 0] - k1[:, 0]
            hh = heights_from_matches(k0, k1, R.F_PX, R.B_M,
                                      centre=(R.CROP / 2.0, R.CROP / 2.0),
                                      distortion_a=R.A_DIST)["heights"]
            hf = hh[np.isfinite(hh)]
            hmode, _ = lowest_prominent_mode(hf, None)
            iqr = float(np.percentile(du, 75) - np.percentile(du, 25))
            span = float(np.percentile(du, 99) - np.percentile(du, 1))
            rows.append({"H": H, "alpha": alpha, "median_du": float(np.median(du)),
                         "iqr": iqr, "span_1_99": span,
                         "mode_h": None if hmode is None else float(hmode),
                         "n": int(len(du))})
            print(f"{H:>6.0f} {alpha:>6.2f} {np.median(du):>10.2f} {iqr:>7.2f} "
                  f"{span:>10.2f} "
                  f"{(hmode if hmode is not None else float('nan')):>8.0f} {len(du):>7}")

    print("\nPRE-REGISTERED TRIGGER: 1-99 span x>=1.8 while IQR grows <40%")
    print(f"{'H':>6} {'alpha':>6} {'span x':>8} {'IQR x':>7} {'triggered':>10}")
    hit = []
    for H0 in (2000.0, 4000.0):
        base = next((r for r in rows if r["H"] == H0 and r["alpha"] == 1.0), None)
        if not base:
            continue
        for r in rows:
            if r["H"] != H0 or r["alpha"] == 1.0:
                continue
            sx = r["span_1_99"] / base["span_1_99"]
            ix = r["iqr"] / base["iqr"]
            trig = sx >= 1.8 and ix < 1.4
            if trig:
                hit.append((H0, r["alpha"]))
            print(f"{H0:>6.0f} {r['alpha']:>6.2f} {sx:>8.2f} {ix:>7.2f} "
                  f"{'YES' if trig else 'no':>10}")

    verdict = ("SUPERPOSITION PRODUCES THE REAL SIGNATURE at "
               + ", ".join(f"H={int(h)}, alpha={a}" for h, a in hit)
               if hit else
               "SUPERPOSITION DOES NOT REPRODUCE THE TAIL GROWTH - the real signature "
               "(span x>=1.8 at a fixed core) is not produced at any mixing fraction")
    print(f"\nVERDICT: {verdict}")
    json.dump({"rows": rows, "sky_frame": sky_name, "verdict": verdict},
              open(os.path.join(HERE, "w8_superposition.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for H0, c in ((2000.0, "#1f6feb"), (4000.0, "#c1440e")):
        rs = [r for r in rows if r["H"] == H0]
        a1.plot([r["alpha"] for r in rs], [r["span_1_99"] for r in rs], "o-", color=c,
                label=f"H={H0:.0f} m  1-99 span")
        a1.plot([r["alpha"] for r in rs], [r["iqr"] for r in rs], "s--", color=c,
                alpha=0.6, label=f"H={H0:.0f} m  IQR")
    a1.set_xlabel("cloud weight alpha"); a1.set_ylabel("disparity spread (px)")
    a1.set_title("Tails vs core under photometric mixing", fontsize=9)
    a1.legend(fontsize=7)
    a2.axhline(37.0, color="#5c6672", ls=":", label="real low regime span")
    a2.axhline(81.7, color="#c1440e", ls=":", label="real high regime span")
    for H0, c in ((2000.0, "#1f6feb"), (4000.0, "#c1440e")):
        rs = [r for r in rows if r["H"] == H0]
        a2.plot([r["alpha"] for r in rs], [r["span_1_99"] for r in rs], "o-", color=c,
                label=f"H={H0:.0f} m")
    a2.set_xlabel("cloud weight alpha"); a2.set_ylabel("1-99 span (px)")
    a2.set_title("Against the real regimes", fontsize=9); a2.legend(fontsize=7)
    fig.savefig(os.path.join(OUT, "w8_superposition.png"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT}/w8_superposition.png and {HERE}/w8_superposition.json")


if __name__ == "__main__":
    main()
