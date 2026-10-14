# PURPOSE: Recover the relative orientation between the two cameras, which the fixed
#          calibration cannot represent, and show what it is worth.
#
#          The evidence that it is missing: the vertical disparity is a nearly
#          constant ~2.2 px across the whole crop, independent of radius. A relative
#          ROLL about the optical axis has already been measured (1.57 deg) and a
#          roll produces a tangential displacement that changes sign across the
#          centre. A displacement that is constant in magnitude and direction across
#          the field is instead a relative PITCH or YAW - a tilt between the optical
#          axes. The calibration is a 2-D affine warp, and a 2-D warp can express
#          rotation-about-the-axis and translation but NOT tilt. After such a warp
#          the pair is not genuinely rectified away from the principal point, so the
#          disparity has a vertical component that neither the paraxial formula nor
#          a parallel-axis triangulation accounts for.
#
#          The rotation is fitted here WITHOUT ground truth, by the physical
#          requirement that a cloud layer's apparent height must not depend
#          systematically on where in the image the match lies. That is a
#          self-calibration from the sky, and its falsifiable prediction is that the
#          radial trend in height collapses once the rotation is applied.
# INPUTS: experiments/layers/heights.npz.
# OUTPUTS: the fitted rotation, before/after radial profiles, a figure.
# KEYWORDS: relative_orientation, tilt, self_calibration, rectification, triangulation
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.geometry.triangulation import pixels_to_rays                       # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})


# PURPOSE: Build a rotation matrix from a small rotation vector (Rodrigues).
# INPUTS: w (3,) axis-angle in radians.
# OUTPUTS: (3,3) rotation matrix.
# KEYWORDS: rotation, rodrigues
def rot(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


# PURPOSE: Height profile (median per radius bin) after applying a relative rotation
#          to camera 2's rays.
# INPUTS: rays1, rays2 (list of (N,3) arrays), radii (list of (N,) arrays),
#         baseline, rotation vector (3,).
# OUTPUTS: (bin_centres, median_heights) for bins with enough matches.
# KEYWORDS: height_profile, rotation, triangulation
def profile(rays1, rays2, radii, baseline, w):
    R = rot(w)
    o2 = np.array([baseline, 0.0, 0.0])
    hs, rs = [], []
    for d1, d2, r in zip(rays1, rays2, radii):
        d2r = d2 @ R.T
        wv = -o2
        a = np.einsum("ij,ij->i", d1, d1)
        b = np.einsum("ij,ij->i", d1, d2r)
        c = np.einsum("ij,ij->i", d2r, d2r)
        dd = d1 @ wv
        e = d2r @ wv
        den = a * c - b * b
        ok = np.abs(den) > 1e-12
        t1 = np.where(ok, (b * e - c * dd) / np.where(ok, den, 1.0), np.nan)
        t2 = np.where(ok, (a * e - b * dd) / np.where(ok, den, 1.0), np.nan)
        p1 = t1[:, None] * d1
        p2 = o2[None, :] + t2[:, None] * d2r
        z = 0.5 * (p1[:, 2] + p2[:, 2])
        keep = ok & np.isfinite(z) & (z > 50) & (z < 20000)
        hs.append(z[keep]); rs.append(r[keep])
    h = np.concatenate(hs); r = np.concatenate(rs)
    centres, med = [], []
    for lo in np.arange(0, 425, 25):
        m = (r >= lo) & (r < lo + 25)
        if m.sum() >= 200:
            centres.append(lo + 12.5); med.append(float(np.median(h[m])))
    return np.array(centres), np.array(med)


# PURPOSE: Spread of the height profile relative to its mean - the objective that
#          self-calibration minimises - with a guard against the degenerate
#          solution.
#
#          A pure relative-spread objective is minimised by driving every height to
#          the SAME value, and a large enough rotation does exactly that: it makes
#          the rays nearly parallel, the implied range enormous, and the heights
#          collapse to a near-constant ~70 m. That is flat but false. The spread is
#          therefore penalised unless the profile also stays in a physically
#          possible band, and the caller additionally bounds the rotation to angles
#          a working rig could actually have.
# INPUTS: w (3,) rotation vector; ray lists; baseline; plausible height band.
# OUTPUTS: scalar cost; large when the solution is degenerate or not computable.
# KEYWORDS: objective, self_calibration, degeneracy_guard
def cost(w, rays1, rays2, radii, baseline, band=(400.0, 6000.0)):
    _, med = profile(rays1, rays2, radii, baseline, w)
    if len(med) < 5 or not np.all(np.isfinite(med)):
        return 1e3
    spread = float((med.max() - med.min()) / med.mean())
    level = float(np.median(med))
    lo, hi = band
    if not (lo <= level <= hi):
        # push the optimiser back into the physical band instead of letting it
        # collapse the profile to a constant
        return 1e2 + abs(math.log(max(level, 1e-6) / lo if level < lo
                                  else level / hi)) * 10.0
    return spread


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    f = float(d["f"]); B = float(d["baseline"]); cx, cy = d["centre"]
    rays1, rays2, radii = [], [], []
    for a, b in zip(K1, K2):
        if len(a) == 0:
            continue
        rays1.append(pixels_to_rays(a, f, (cx, cy)))
        rays2.append(pixels_to_rays(b, f, (cx, cy)))
        radii.append(np.hypot(a[:, 0] - cx, a[:, 1] - cy))
    print(f"{len(rays1)} frames, {sum(len(x) for x in radii)} matches")

    c0 = cost(np.zeros(3), rays1, rays2, radii, B)
    print(f"\nbefore (parallel axes): profile spread {100*c0:.1f} %")
    x0, m0 = profile(rays1, rays2, radii, B, np.zeros(3))

    from scipy.optimize import minimize
    # The rig is rigid and the calibration was derived from the Sun, so the residual
    # relative orientation must be small. Bounding it is a physical prior, not a
    # convenience: an unconstrained fit collapses the profile (see cost()).
    LIM = 0.035          # rad, about 2.0 deg
    bounds = [(-LIM, LIM)] * 3
    best = None
    for guess in (0.0, 0.004, -0.004, 0.01, -0.01):
        res = minimize(cost, np.array([guess, 0.0, 0.0]),
                       args=(rays1, rays2, radii, B), method="L-BFGS-B",
                       bounds=bounds,
                       options={"ftol": 1e-12, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    w = best.x
    print(f"fitted relative rotation: pitch {math.degrees(w[0]):+.3f} deg, "
          f"yaw {math.degrees(w[1]):+.3f} deg, roll {math.degrees(w[2]):+.3f} deg")
    print(f"after: profile spread {100*best.fun:.1f} %")
    x1, m1 = profile(rays1, rays2, radii, B, w)

    print(f"\n{'radius':>8} {'before (m)':>11} {'after (m)':>10}")
    for xi, a_, b_ in zip(x0, m0, m1):
        print(f"{xi:>8.0f} {a_:>11.0f} {b_:>10.0f}")
    print(f"\nimplied pitch of {math.degrees(w[0]):+.3f} deg corresponds to "
          f"{abs(math.radians(w[0]))*f:.2f} px of vertical disparity at f = {f:.0f} px/rad")
    json.dump({"pitch_deg": math.degrees(w[0]), "yaw_deg": math.degrees(w[1]),
               "roll_deg": math.degrees(w[2]),
               "spread_before_pct": 100 * c0, "spread_after_pct": 100 * best.fun},
              open(os.path.join(HERE, "relative_orientation.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(x0, m0, "o-", color="#c1440e", lw=1.9, ms=4.5, label="parallel axes (shipped)")
    ax.plot(x1, m1, "s-", color="#1a7f37", lw=1.9, ms=4.5,
            label=f"with fitted relative rotation ({math.degrees(w[0]):+.2f}° pitch)")
    ax.set_xlabel("radius from principal point (px)")
    ax.set_ylabel("median height (m)")
    ax.set_title("A rotation the affine cannot express removes most of the radial trend")
    ax.legend(fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "relative_orientation.png"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT}/relative_orientation.png")


if __name__ == "__main__":
    main()
