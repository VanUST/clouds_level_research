# PURPOSE: Check the relative-orientation fit against an INDEPENDENT observable, and
#          report honestly if it fails.
#
#          The self-calibrating fit in relative_orientation.py flattens the
#          height-versus-radius profile from 37 % to 2.5 % by fitting a relative
#          rotation of pitch +0.99 deg, yaw +0.82 deg. But a rotation must also
#          PREDICT the vertical disparity: a pitch of theta produces roughly f*theta
#          pixels of vertical offset, so 0.99 deg predicts ~10.6 px, while the
#          measured vertical disparity is ~2.2 px. If the fit is real those must
#          agree; if they do not, the fit is exploiting the height-rotation
#          degeneracy rather than recovering the orientation.
#
#          This script evaluates the profile on a grid of rotations whose implied
#          vertical disparity is consistent with what is observed, and reports
#          whether any of them flattens the profile.
# INPUTS: experiments/layers/heights.npz.
# OUTPUTS: a printed grid; relative_orientation_check.json.
# KEYWORDS: degeneracy, tilt, falsification, self_calibration, vertical_disparity
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.geometry.triangulation import pixels_to_rays                       # noqa: E402
from relative_orientation import cost, profile, rot                            # noqa: E402


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    f = float(d["f"]); B = float(d["baseline"]); cx, cy = d["centre"]
    rays1, rays2, radii = [], [], []
    dv_all = []
    for a, b in zip(K1, K2):
        if len(a) == 0:
            continue
        rays1.append(pixels_to_rays(a, f, (cx, cy)))
        rays2.append(pixels_to_rays(b, f, (cx, cy)))
        radii.append(np.hypot(a[:, 0] - cx, a[:, 1] - cy))
        dv_all.append(np.abs(a[:, 1] - b[:, 1]))
    dv = np.concatenate(dv_all)
    print(f"observed |vertical disparity|: median {np.median(dv):.2f} px "
          f"(this is an INDEPENDENT constraint on any fitted rotation)\n")

    w_fit = np.radians([0.993, 0.818, 0.188])
    pred_fit = f * math.hypot(w_fit[0], w_fit[1])
    print(f"the fit from relative_orientation.py: pitch {0.993:+.3f} deg, "
          f"yaw {0.818:+.3f} deg")
    print(f"  it predicts ~{pred_fit:.1f} px of vertical disparity from a tilt alone; "
          f"measured {np.median(dv):.2f} px")
    print(f"  -> {'CONSISTENT' if abs(pred_fit-np.median(dv)) < 1.0 else 'INCONSISTENT: the fit is not the true orientation'}\n")

    # a rotation whose implied vertical disparity matches what is observed
    theta_obs = float(np.median(dv)) / f
    print(f"a tilt consistent with the data is at most ~{math.degrees(theta_obs):.3f} deg\n")

    print(f"{'pitch(deg)':>11} {'yaw(deg)':>9} {'implied dv(px)':>15} "
          f"{'profile spread':>15} {'median H(m)':>12}")
    rows = []
    for pitch_deg in (-0.30, -0.21, -0.10, 0.0, 0.10, 0.21, 0.30):
        for yaw_deg in (0.0, 0.21, -0.21):
            w = np.radians([pitch_deg, yaw_deg, 0.0])
            c = cost(w, rays1, rays2, radii, B)
            _, med = profile(rays1, rays2, radii, B, w)
            implied = f * math.hypot(w[0], w[1])
            print(f"{pitch_deg:>11.2f} {yaw_deg:>9.2f} {implied:>15.2f} "
                  f"{100*c:>14.1f}% {np.median(med):>12.0f}")
            rows.append(dict(pitch_deg=pitch_deg, yaw_deg=yaw_deg,
                             implied_dv_px=implied, spread_pct=100 * c,
                             median_h=float(np.median(med))))
    best = min(rows, key=lambda r: r["spread_pct"])
    print(f"\nbest within the physically consistent range: spread "
          f"{best['spread_pct']:.1f} % at pitch {best['pitch_deg']:+.2f} deg, "
          f"median height {best['median_h']:.0f} m")
    print("If that spread is still large, the radial trend is NOT explained by the "
          "relative orientation, and the 37 %->2.5 % fit is a degeneracy artefact.")
    json.dump({"observed_dv_median_px": float(np.median(dv)),
               "fit_implied_dv_px": pred_fit, "grid": rows},
              open(os.path.join(HERE, "relative_orientation_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
