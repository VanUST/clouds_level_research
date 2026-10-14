# PURPOSE: Determine, by construction, whether the pipeline's triangulation uses the
#          correct ray origin — the coordinate-frame claim under review.
#
#          The geometry, derived rather than assumed. The calibration affine M is a pure
#          rotation (|det| = 1.000000) plus a translation. Warping camera 1 by M brings
#          it into camera 2's frame, so the COMMON frame is camera 2's sensor frame. For
#          an ideal rotationally-symmetric fisheye, a rotation about the optical axis
#          commutes with the projection, so the warped image 1 is itself a valid fisheye
#          image — centred on camera 2's principal point c2.
#
#          Writing M(p) = R(p - w) + w + delta, with w the image centre, the alignment
#          condition for a distant object is
#              c2 = R(c1 - w) + w + delta
#          which fixes c2 RELATIVE to c1 but NOT absolutely. The calibration therefore
#          cannot say where either principal point is; it fixes only their relation.
#
#          The pipeline assumes c1 = c2 = crop centre. This test asks how much height
#          error that assumption causes, by projecting a known plane through the model
#          with the true centre and triangulating with the assumed one.
#
#          Both rays are taken about c2 in the rectified frame, and the baseline is along
#          +x in that frame, which the observed disparity direction (du ~ 15 px,
#          dv ~ 2 px) independently confirms.
# INPUTS: configs/calib/2025-05-22_affine_old.json
# OUTPUTS: experiments/layers/frame_bug_test.json and a printed verdict.
# KEYWORDS: coordinate_frame, principal_point, verification, synthetic, R8
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.geometry.triangulation import heights_from_matches                 # noqa: E402

CAL = os.path.join(REPO, "configs", "calib", "2025-05-22_affine_old.json")
F = 654.0            # the measured focal constant, px/rad
B = 45.0
CROP = 640
ORIG = 1920
CENTRE = (ORIG / 2.0, ORIG / 2.0)


def project(dirv, centre, f=F, a=0.0):
    """Fisheye projection of direction vectors `dirv` about `centre`."""
    d = np.atleast_2d(np.asarray(dirv, float))
    n = np.linalg.norm(d, axis=1)
    th = np.arccos(np.clip(d[:, 2] / n, -1, 1))
    ph = np.arctan2(d[:, 1], d[:, 0])
    if a:
        t = th.copy()
        for _ in range(60):
            g = t * (1 + a * t * t) - th
            t = t - g / (1 + 3 * a * t * t)
        r = f * t
    else:
        r = f * th
    return np.stack([centre[0] + r * np.cos(ph), centre[1] + r * np.sin(ph)], axis=1)


def main():
    cal = json.load(open(CAL))
    M = np.array(cal["matrix_2x3"], float)
    R, t = M[:, :2], M[:, 2]
    w = np.array(CENTRE)
    delta = t - (w - R @ w)
    print(f"affine rotation {math.degrees(math.atan2(R[1,0], R[0,0])):.3f} deg, "
          f"|det| = {abs(np.linalg.det(R)):.6f}")
    print(f"residual beyond a rotation about the image centre: "
          f"({delta[0]:+.2f}, {delta[1]:+.2f}) px, |delta| = {np.hypot(*delta):.2f} px")
    print(f"  = {math.degrees(np.hypot(*delta)/F):.2f} deg of angle at f={F:.0f}\n")

    # c2 = R(c1 - w) + w + delta. Assuming camera 1's principal point is at the image
    # centre, camera 2's lands at w + delta. This is the case the pipeline ignores.
    c1_assumed_centre = w
    c2_true = R @ (c1_assumed_centre - w) + w + delta
    print(f"if camera 1's principal point is the image centre {w},")
    print(f"the alignment condition puts camera 2's principal point at "
          f"({c2_true[0]:.2f}, {c2_true[1]:.2f})")
    print(f"  -> {np.hypot(*(c2_true - w)):.2f} px from the image centre, "
          f"{math.degrees(np.hypot(*(c2_true - w))/F):.2f} deg of angle\n")

    # a plane at H, spanning the field in the rectified frame
    out = {}
    print(f"{'H (m)':>8} {'centre used':>26} {'median h':>10} {'std':>8} "
          f"{'bias':>9} {'u-gradient':>12}")
    for H in (1000.0, 1500.0, 1800.0, 3000.0):
        # world points; camera 1 at origin, camera 2 at +x, both in the rectified frame
        rng = np.linspace(-700, 700, 25)
        XX, YY = np.meshgrid(rng, rng)
        P = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, H)], axis=1)
        O1 = np.zeros(3)
        O2 = np.array([B, 0.0, 0.0])
        d1 = P - O1
        d2 = P - O2
        for label, centre in (("crop centre (what the code uses)", np.array([ORIG/2, ORIG/2.0])),
                              ("true c2 (alignment condition)", c2_true)):
            p1 = project(d1, centre)          # warped frame 1 = camera 2's frame
            p2 = project(d2, centre)
            off = (ORIG - CROP) / 2.0
            k1 = p1 - off
            k2 = p2 - off
            # the pipeline takes BOTH about the crop centre, whatever `centre` was used
            # to render; render with `centre`, triangulate about the crop centre
            h = heights_from_matches(k1, k2, F, B,
                                     centre=(CROP/2.0, CROP/2.0),
                                     distortion_a=0.0)["heights"]
            ok = np.isfinite(h)
            hh = h[ok]; uu = k1[ok, 0] - CROP/2.0
            A = np.vstack([uu, np.ones_like(uu)]).T
            coef, *_ = np.linalg.lstsq(A, hh, rcond=None)
            print(f"{H:>8.0f} {label:>26} {np.median(hh):>10.0f} {np.std(hh):>8.0f} "
                  f"{100*(np.median(hh)-H)/H:>+8.1f}% {coef[0]*CROP:>10.0f} m")
            out[f"H={H:.0f}|{label}"] = {"median_h": float(np.median(hh)),
                                         "bias_pct": float(100*(np.median(hh)-H)/H),
                                         "u_gradient_m": float(coef[0]*CROP)}
        print()

    # the decisive question: does rendering about c2 but triangulating about the crop
    # centre bias the height, and does it produce a u-gradient?
    print("Interpretation")
    print("  Row A (crop centre)  : both the rendering and the triangulation use the")
    print("                         crop centre, so the model is self-consistent and the")
    print("                         ideal-fisheye round trip is exact — this is a control.")
    print("  Row B (true c2)      : the rays are rendered about camera 2's real principal")
    print("                         point but triangulated about the crop centre, which is")
    print("                         what the pipeline does whenever c2 != crop centre.")
    a_bias = [v["bias_pct"] for k, v in out.items()
              if "crop centre" in k and "H=1800" in k]
    b_bias = [v["bias_pct"] for k, v in out.items()
              if "true c2" in k and "H=1800" in k]
    print(f"\n  at H=1800 m: row A bias {a_bias[0]:+.2f} %, row B bias {b_bias[0]:+.2f} %")
    verdict = ("the wrong principal point biases the height by "
               f"{abs(b_bias[0]):.2f} % at 1800 m"
               if abs(b_bias[0]) > 1.0 else
               "the wrong principal point does NOT materially bias the height")
    print(f"\nVERDICT: {verdict}")
    out["verdict"] = verdict
    out["c2_true"] = c2_true.tolist()
    out["residual_px"] = delta.tolist()
    json.dump(out, open(os.path.join(HERE, "frame_bug_test.json"), "w"), indent=1)
    print(f"wrote {os.path.join(HERE, 'frame_bug_test.json')}")


if __name__ == "__main__":
    main()
