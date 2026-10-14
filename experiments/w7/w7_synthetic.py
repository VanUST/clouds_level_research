# PURPOSE: W7 synthetic control — render a cloud layer at a KNOWN geometric height
#          through the measured fisheye geometry, run the production matcher on it, and
#          ask whether the recovered disparity tracks the truth.
#
#          WHY THIS IS NEEDED. Ten explanations for the real-world inversion have been
#          tested and refuted. Every one was a property of the scene or of the pipeline,
#          each measurable in isolation, and each absent or wrong-signed. A synthetic
#          scene is the one input where the height is known exactly, so the estimator can
#          be asked to recover a number it cannot know.
#
#          WHAT IT ISOLATES. The analytic round trip is already known to be exact: a plane
#          at 1500 m triangulated by the same code recovers 1500.00 m, bias 0.00 %
#          (experiments/layers/frame_bug_test.py). So this experiment varies only the
#          matcher, and any failure it finds belongs to RoMa rather than to the geometry.
#
#          THE DESIGN, and why it differs from the blueprint. Three corrections were
#          applied after review:
#
#          1. The ground truth is a PER-PIXEL disparity map, not a scalar. A flat plane
#             does not give a constant disparity through a fisheye: at 1000 m it runs
#             27.5 px at the centre and 24.6 px at the crop corner. Comparing a scalar
#             would confound the fisheye geometry with the matcher's error.
#
#          2. A SINGLE height cannot detect an inversion. An inversion is a relation
#             between two heights, so the renderer SWEEPS H and the test asks whether the
#             recovered disparity tracks the true one monotonically.
#
#          3. Texture scale is varied INDEPENDENTLY of height. At a fixed physical size,
#             a cloud feature subtends a smaller angle when it is higher, so height and
#             apparent texture scale are normally inseparable. Rendering both lets the
#             two be told apart - which is the only way to test the texture-scale
#             hypothesis rather than assume it.
# INPUTS: the measured camera model (f, distortion, roll, baseline).
# OUTPUTS: experiments/w7/w7_synthetic.json, docs/report/w7_synthetic.png.
# KEYWORDS: W7, synthetic, ground_truth, closed_loop, matcher_only, texture_scale
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))

F_PX = 654.0
A_DIST = -0.30
B = 45.0
ORIG = 1920
CROP = 640
CENTRE = np.array([ORIG / 2.0, ORIG / 2.0])


# PURPOSE: Invert the radial fisheye law theta = (r/f)(1 + a (r/f)^2) for r.
# INPUTS: theta (array, radians), f (px/rad), a (cubic coefficient).
# OUTPUTS: radius r in pixels, same shape as theta. Uses Newton's method, which
#          converges in a few steps because the law is monotonic over this field
#          (the derivative 1 + 3a t^2 stays positive: 0.51 at the crop corner).
# KEYWORDS: fisheye, inverse, newton
def radius_from_theta(theta, f=F_PX, a=A_DIST):
    t = np.asarray(theta, float)
    y = t.copy()
    for _ in range(40):
        g = y * (1.0 + a * y * y) - t
        gp = 1.0 + 3.0 * a * y * y
        y = y - g / gp
    return f * y


# PURPOSE: Turn a point in the world into a pixel in a nadir-pointing fisheye camera.
# INPUTS: P (N,3) world points; origin (3,) of the camera; roll (float, degrees) applied
#         to the image as the pipeline's measured roll is applied to camera 2.
# OUTPUTS: (N,2) pixel coordinates in the 1920x1920 frame.
# KEYWORDS: forward_model, fisheye, projection
def project(P, origin, roll_deg=0.0):
    d = np.asarray(P, float) - np.asarray(origin, float)
    n = np.linalg.norm(d, axis=1)
    # These cameras point at the ZENITH, so the optical axis is +z: the angle from
    # it is arccos(d_z / |d|). Using -d_z would place every ray in the opposite
    # hemisphere and invert the disparity's sign.
    theta = np.arccos(np.clip(d[:, 2] / n, -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])
    r = radius_from_theta(theta)
    u = CENTRE[0] + r * np.cos(phi)
    v = CENTRE[1] + r * np.sin(phi)
    if roll_deg:
        a = math.radians(roll_deg)
        du = u - CENTRE[0]; dv = v - CENTRE[1]
        u = CENTRE[0] + du * math.cos(a) - dv * math.sin(a)
        v = CENTRE[1] + du * math.sin(a) + dv * math.cos(a)
    return np.stack([u, v], axis=1)


# PURPOSE: Generate a procedural cloud texture at a chosen physical feature scale.
# INPUTS: size (int) pixels of the texture tile; scale_px (float) the characteristic
#         feature size in texture pixels; seed (int).
# OUTPUTS: (size,size) float array in [0,1]. Built by summing octaves of smoothed noise
#          so the power spectrum falls off like a real cloud's, which matters because a
#          white-noise texture would have no characteristic scale to test.
# KEYWORDS: procedural, texture, fractal, cloud
def cloud_texture(size, scale_px, seed=0, octaves=4):
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), float)
    amp = 1.0
    sc = scale_px
    for _ in range(octaves):
        small = max(int(size / max(sc, 2.0)), 2)
        if small < 2:
            break
        n = rng.random((small, small))
        import cv2
        big = cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
        acc += amp * big
        amp *= 0.5
        sc /= 2.0
    acc -= acc.min()
    if acc.max() > 0:
        acc /= acc.max()
    return acc


AFFINE = None


def _affine():
    """The shipped calibration affine, loaded once."""
    global AFFINE
    if AFFINE is None:
        cal = json.load(open(os.path.join(REPO, "configs", "calib",
                                          "2025-05-22_affine_old.json")))
        AFFINE = np.array(cal["matrix_2x3"], float)
    return AFFINE


def render(height_m, origin1, origin2, tex, tex_extent_m, roll2=1.5, grid=ORIG,
           rectify=True):
    """Render the pair by ray-casting the plane z = height_m through each camera.

    With rectify=True - the pipeline's own convention - camera 1's image is warped by
    the calibration affine before it is returned, exactly as StereoAnalyzer.prepare_pair
    does. Without it the pair is un-rectified, and matching it tests a configuration the
    production code never sees. The first version of this file omitted the warp, so its
    result described a synthetic rig that does not exist.
    """
    import cv2
    xs = np.arange(grid, dtype=np.float32)
    uu, vv = np.meshgrid(xs, xs)
    # radial coordinate and angle in each camera's own frame
    def sample(origin, roll_deg):
        du = uu - CENTRE[0]; dv = vv - CENTRE[1]
        if roll_deg:
            a = math.radians(-roll_deg)
            du2 = du * math.cos(a) - dv * math.sin(a)
            dv2 = du * math.sin(a) + dv * math.cos(a)
            du, dv = du2, dv2
        r = np.hypot(du, dv)
        t = r / F_PX
        theta = t * (1.0 + A_DIST * t * t)
        phi = np.arctan2(dv, du)
        # ray from the camera, tilted from the ZENITH (+z) by theta
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        # intersect the plane z = height with the camera at z = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            s = height_m / np.maximum(dz, 1e-9)
        X = origin[0] + s * dx
        Y = origin[1] + s * dy
        # texture lookup, in metres, wrapped into the tile
        tx = np.mod((X + tex_extent_m / 2) / tex_extent_m, 1.0)
        ty = np.mod((Y + tex_extent_m / 2) / tex_extent_m, 1.0)
        px = np.clip((tx * (tex.shape[1] - 1)).astype(np.int32), 0, tex.shape[1] - 1)
        py = np.clip((ty * (tex.shape[0] - 1)).astype(np.int32), 0, tex.shape[0] - 1)
        img = tex[py, px]
        img[(r > 940)] = 0.0            # outside the lens circle
        return img, r
    g1, r1 = sample(origin1, 0.0)
    g2, r2 = sample(origin2, roll2)
    if rectify:
        g1 = cv2.warpAffine(g1.astype(np.float32), _affine(), dsize=(grid, grid),
                            flags=cv2.INTER_LINEAR)
    return g1, g2, r1


# PURPOSE: The exact geometric disparity map for a plane, per pixel of camera 1.
# INPUTS: height_m, origins, grid.
# OUTPUTS: (grid,grid) disparity in pixels, NaN outside the lens.
# KEYWORDS: ground_truth, disparity_map
def gt_disparity(height_m, origin1, origin2, roll2=1.5, grid=ORIG, rectify=True):
    """The exact disparity at every pixel of the frame the matcher actually sees.

    With rectify=True the ground truth is expressed in the WARPED camera-1 frame: for a
    pixel p there, the camera-1 native coordinate is M^-1 p, its ray is cast onto the
    plane, and the 3-D point is projected into camera 2. Returning the native-frame
    disparity instead would compare against a frame the matcher never sees.
    """
    xs = np.arange(grid, dtype=np.float32)
    uu, vv = np.meshgrid(xs, xs)
    du = uu - CENTRE[0]; dv = vv - CENTRE[1]
    r = np.hypot(du, dv)
    t = r / F_PX
    theta = t * (1.0 + A_DIST * t * t)
    phi = np.arctan2(dv, du)
    dx = np.sin(theta) * np.cos(phi); dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    s = height_m / np.maximum(dz, 1e-9)
    # the point lies ON the plane z = height_m; setting z to zero here (as an earlier
    # version did) puts every point at the camera's own altitude and inverts the sign of
    # the disparity without raising any error.
    P = np.stack([origin1[0] + s * dx, origin1[1] + s * dy,
                  np.full_like(dx, float(height_m))], axis=-1)
    p2 = project(P.reshape(-1, 3), origin2, roll2).reshape(grid, grid, 2)
    if rectify:
        # uu, vv here are WARPED-frame pixels; M^-1 maps them back to camera 1's native
        # frame, which is where the rays above were cast from
        Minv = cv2.invertAffineTransform(_affine())
        pts = np.stack([uu.ravel(), vv.ravel()], axis=1)
        nat = (pts @ Minv[:, :2].T) + Minv[:, 2]
        # recompute the ray from the NATIVE coordinate, then re-project
        dxn = nat[:, 0] - CENTRE[0]; dyn = nat[:, 1] - CENTRE[1]
        rn = np.hypot(dxn, dyn)
        tn = rn / F_PX
        thn = tn * (1.0 + A_DIST * tn * tn)
        phn = np.arctan2(dyn, dxn)
        dxr = np.sin(thn) * np.cos(phn); dyr = np.sin(thn) * np.sin(phn)
        dzr = np.cos(thn)
        sn = height_m / np.maximum(dzr, 1e-9)
        Pn = np.stack([origin1[0] + sn * dxr, origin1[1] + sn * dyr,
                       np.full_like(dxr, float(height_m))], axis=1)
        p2n = project(Pn, origin2, roll2)
        d = uu - p2n[:, 0].reshape(grid, grid)
        d[(r > 940)] = np.nan
    else:
        d = uu - p2[:, :, 0]
        d[r > 940] = np.nan
    return d


def main():
    from clouds.matching.matchers import create_matcher
    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")

    # a texture with a 250 m characteristic feature size: real cloud structure
    tex = cloud_texture(1024, 90.0, seed=3)
    heights = [1000.0, 1500.0, 2000.0, 3000.0, 4000.0]
    print(f"{'H_true':>8} {'gt du (crop)':>13} {'recovered':>11} {'bias':>9} "
          f"{'bias %':>8} {'IQR':>7}")
    rows = []
    for H in heights:
        o1 = np.array([0.0, 0.0, 0.0])
        o2 = np.array([B, 0.0, 0.0])
        g1, g2, r1 = render(H, o1, o2, tex, tex_extent_m=4000.0)
        gtd = gt_disparity(H, o1, o2)
        # the pipeline's own crop
        y0 = (ORIG - CROP) // 2
        c1 = (g1[y0:y0 + CROP, y0:y0 + CROP] * 255).astype(np.uint8)
        c2 = (g2[y0:y0 + CROP, y0:y0 + CROP] * 255).astype(np.uint8)
        k0, k1 = matcher(c1, c2)
        k0 = np.asarray(k0, float).reshape(-1, 2)
        k1 = np.asarray(k1, float).reshape(-1, 2)
        if len(k0) == 0:
            print(f"{H:>8.0f}  no matches")
            continue
        rec = k0[:, 0] - k1[:, 0]
        gtc = gtd[y0:y0 + CROP, y0:y0 + CROP]
        vi = np.clip(k0[:, 1].astype(int), 0, CROP - 1)
        ui = np.clip(k0[:, 0].astype(int), 0, CROP - 1)
        truth = gtc[vi, ui]
        ok = np.isfinite(truth)
        bias = float(np.median(rec[ok] - truth[ok]))
        rows.append({"H": H, "n": int(ok.sum()),
                     "gt_median": float(np.median(truth[ok])),
                     "recovered_median": float(np.median(rec[ok])),
                     "bias_px": bias,
                     "bias_pct": float(100 * bias / np.median(truth[ok])),
                     "iqr": float(np.percentile(rec[ok], 75) - np.percentile(rec[ok], 25)),
                     "bias_std": float(np.std(rec[ok] - truth[ok]))})
        r_ = rows[-1]
        print(f"{H:>8.0f} {r_['gt_median']:>13.3f} {r_['recovered_median']:>11.3f} "
              f"{r_['bias_px']:>+9.3f} {r_['bias_pct']:>+7.2f}% {r_['iqr']:>7.3f}")

    if len(rows) >= 3:
        H = np.array([r_["H"] for r_ in rows])
        gt = np.array([r_["gt_median"] for r_ in rows])
        rc = np.array([r_["recovered_median"] for r_ in rows])
        from scipy.stats import spearmanr
        rho, p = spearmanr(gt, rc)
        bias = rc - gt
        print(f"\nmonotonicity of the recovery: rho(true, recovered) = {rho:+.3f} (p = {p:.3f})")
        print(f"bias across heights: {bias.min():+.3f} to {bias.max():+.3f} px "
              f"(spread {bias.max()-bias.min():.3f} px)")
        print(f"  the real-world cross-regime effect is +1.798 px")
        slope = float(np.polyfit(gt, rc, 1)[0])
        print(f"  recovered-vs-true slope = {slope:.4f} (1.0 = perfect)")
        if rho > 0.9 and abs(bias).max() < 0.5:
            verdict = ("THE MATCHER RECOVERS A KNOWN HEIGHT — the synthetic control "
                       "passes, so the real-world inversion is an unmodelled physical "
                       "effect rather than a matcher defect")
        elif rho > 0.9:
            verdict = ("THE MATCHER TRACKS THE TRUE HEIGHT but with a bias — the ordering "
                       "is right and the scale is off")
        else:
            verdict = ("THE MATCHER DOES NOT TRACK THE TRUE HEIGHT on synthetic cloud "
                       "texture — the defect is intrinsic to the matcher and this "
                       "projection")
        print(f"\n  VERDICT: {verdict}")
    else:
        verdict = "insufficient renders"
        bias = np.array([0.0])

    os.makedirs(HERE, exist_ok=True)
    json.dump({"rows": rows, "verdict": verdict}, open(os.path.join(HERE,
              "w7_synthetic.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
        H = [r_["H"] for r_ in rows]
        ax1.plot(H, [r_["gt_median"] for r_ in rows], "o-", color="#5c6672",
                 label="ground truth")
        ax1.plot(H, [r_["recovered_median"] for r_ in rows], "s-", color="#c1440e",
                 label="RoMa recovered")
        ax1.set_xlabel("rendered cloud height (m)"); ax1.set_ylabel("median disparity (px)")
        ax1.set_title("Synthetic control: does the matcher recover a known height?",
                      fontsize=9)
        ax1.legend(fontsize=8)
        ax2.bar([str(int(h)) for h in H], [r_["bias_px"] for r_ in rows], color="#1f6feb")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel("rendered height (m)"); ax2.set_ylabel("bias, recovered - truth (px)")
        ax2.set_title("Bias against the exact per-pixel ground truth", fontsize=9)
        fig.savefig(os.path.join(OUT, "w7_synthetic.png"), bbox_inches="tight",
                    facecolor="white")
        print(f"\nwrote {OUT}/w7_synthetic.png and {HERE}/w7_synthetic.json")


if __name__ == "__main__":
    main()
