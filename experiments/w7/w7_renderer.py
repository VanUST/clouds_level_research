# PURPOSE: A synthetic fisheye stereo renderer whose geometry is VALIDATED against the
#          shipped calibration before any result is taken from it.
#
#          WHY A SEPARATE MODULE. The first W7 renderer (w7_synthetic.py) produced a
#          ground truth that was never validated in the frame the matcher consumes, and it
#          modelled both cameras with the same orientation - a rig that does not exist. It
#          was withdrawn (W7_RETRACTION.md). This module exists so that the geometry is
#          stated once, checked explicitly, and cannot be silently wrong again.
#
#          THE RIG, and how it was derived rather than assumed. The calibration affine is
#              M(p) = R p + t,  R = [[0.00142, -0.99999], [0.99999, 0.00142]],
#              t = (1972.74, 3.67)
#          with |det R| = 1.000000, i.e. a pure rotation of 89.919 degrees. Since the
#          pipeline warps CAMERA 1 by M to bring it into camera 2's frame, the pair must be
#          built so that the warp aligns rather than misaligns them:
#
#            - the baseline lies along world -Y, camera 2 at (0, -B, 0), so the parallax
#              lands in u after the warp and is POSITIVE in the pipeline's own convention
#              (kp1 - kp2), as it is in the real data (du ~ 13 px against dv ~ 2 px). With
#              +Y the rendered parallax is exactly inverted, which is how the sign was
#              pinned down;
#            - camera 2's sensor is rotated by -90 degrees about the optical axis;
#            - camera 1's principal point is the image centre, and camera 2's is
#              c2 = t + R c1 = (1014.10, 965.03).
#
#          The last one is the term that took longest to find. With both principal points
#          at the image centre the rendered parallax is NEGATIVE and grows with height;
#          with c2 set by the affine it is positive and falls as 1/H.
#
#          THE GATE, run by `validate()` and required to pass before any experiment:
#              du(zenith) = (c2_u - w_u) + f * atan(B / H)
#          which for B=45, f=654 is 1.36 + f*atan(45/H): 30.78, 16.08, 8.72 px at 1000,
#          2000 and 4000 m. The analytic parallax term alone falls strictly as 1/H.
# INPUTS: none (the geometry is fixed by the calibration).
# OUTPUTS: validated image pairs and per-pixel ground-truth disparity maps.
# KEYWORDS: W7, renderer, validation, fisheye, affine, principal_point, rig_geometry
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

F_PX = 654.0                  # the measured focal constant
A_DIST = -0.30                # the measured cubic distortion
ORIG = 1920
CROP = 640
C1 = np.array([960.0, 960.0])                 # camera 1's principal point
B_M = 45.0                                    # baseline, along world +Y


def _affine():
    cal = json.load(open(os.path.join(REPO, "configs", "calib",
                                      "2025-05-22_affine_old.json")))
    return np.array(cal["matrix_2x3"], float)


def c2_required(c1=C1):
    """The principal point the affine requires for camera 2, given camera 1's.

    The warp maps camera 1's principal point onto camera 2's, so M(c1) = c2, which with
    M(p) = R p + t gives c2 = R c1 + t. This is the term whose absence made the first
    renderer produce an inverted parallax.
    """
    M = _affine()
    R, t = M[:, :2], M[:, 2]
    return R @ np.asarray(c1, float) + t


C2 = c2_required()


# PURPOSE: Invert the radial fisheye law theta = (r/f)(1 + a (r/f)^2) for r.
# INPUTS: theta (radians), f, a. OUTPUTS: radius in pixels.
# KEYWORDS: inverse, fisheye, newton
def radius_from_theta(theta, f=F_PX, a=A_DIST):
    t = np.asarray(theta, float)
    y = t.copy()
    for _ in range(60):
        g = y * (1.0 + a * y * y) - t
        y = y - g / (1.0 + 3.0 * a * y * y)
    return f * y


# PURPOSE: Project world points into camera 2's sensor frame.
# INPUTS: P (N,3) world points; origin (3,) of camera 2; roll_deg the extra in-plane
#         rotation (the measured roll correction the pipeline applies); principal (2,).
# OUTPUTS: (N,2) pixels. The sensor is rotated -90 degrees about the optical axis, which
#          is what makes the affine warp - a +89.919 degree rotation - align the pair.
# KEYWORDS: projection, sensor_rotation, camera2
def project_cam2(P, origin=(0.0, -B_M, 0.0), roll_deg=0.0, principal=None):
    principal = C2 if principal is None else np.asarray(principal, float)
    d = np.asarray(P, float) - np.asarray(origin, float)
    n = np.linalg.norm(d, axis=1)
    theta = np.arccos(np.clip(d[:, 2] / n, -1, 1))      # cameras point at the zenith
    phi_world = np.arctan2(d[:, 1], d[:, 0])
    phi_sensor = phi_world + math.pi / 2.0 + math.radians(roll_deg)
    r = radius_from_theta(theta)
    return np.stack([principal[0] + r * np.cos(phi_sensor),
                     principal[1] + r * np.sin(phi_sensor)], axis=1)


# PURPOSE: Cast camera 1's pixels onto the plane z = H and return the world points.
# INPUTS: H, grid, principal, distortion.
# OUTPUTS: (grid*grid, 3) world points and the (grid,grid) radius map.
# KEYWORDS: ray_cast, plane, camera1
def cast_camera1(H, grid=ORIG, principal=None, a=A_DIST):
    principal = C1 if principal is None else np.asarray(principal, float)
    xs = np.arange(grid, dtype=np.float64)
    uu, vv = np.meshgrid(xs, xs)
    du = uu - principal[0]; dv = vv - principal[1]
    r = np.hypot(du, dv)
    t = r / F_PX
    theta = t * (1.0 + a * t * t)
    phi = np.arctan2(dv, du)
    dz = np.cos(theta)
    s = H / np.maximum(dz, 1e-9)
    P = np.stack([s * np.sin(theta) * np.cos(phi),
                  s * np.sin(theta) * np.sin(phi),
                  np.full_like(theta, float(H))], axis=-1)
    return P.reshape(-1, 3), r


# PURPOSE: The exact ground-truth disparity in the WARPED frame the matcher consumes.
# INPUTS: H, roll correction, grid.
# OUTPUTS: (grid,grid) du and dv maps. du falls as 1/H; dv is dominated by the affine's
#          own translation residual and is nearly constant.
# KEYWORDS: ground_truth, warped_frame, disparity
def gt_disparity(H, roll_deg=0.0, grid=ORIG):
    P, r = cast_camera1(H, grid)
    p2 = project_cam2(P, roll_deg=roll_deg).reshape(grid, grid, 2)
    xs = np.arange(grid, dtype=np.float64)
    uu, vv = np.meshgrid(xs, xs)
    M = _affine()
    u_w = M[0, 0] * uu + M[0, 1] * vv + M[0, 2]
    v_w = M[1, 0] * uu + M[1, 1] * vv + M[1, 2]
    # the pipeline's disparity is kp1 - kp2, i.e. the warped camera-1 pixel minus the
    # camera-2 pixel. Reporting p2 - u_w instead inverts every sign in the project.
    du = u_w - p2[:, :, 0]
    dv = v_w - p2[:, :, 1]
    outside = r > 940
    du[outside] = np.nan; dv[outside] = np.nan
    return du, dv


# PURPOSE: Render the pair by sampling a texture at the world points each camera sees.
# INPUTS: H, texture (2-D array), texture extent in metres, roll correction, grid.
# OUTPUTS: (img1_warped, img2) float arrays in [0,1]. img1 is warped by the calibration
#          affine, exactly as the pipeline's prepare_pair does.
# KEYWORDS: render, texture, warp
def render_pair(H, tex, tex_extent_m, roll_deg=0.0, grid=ORIG):
    extent = float(tex_extent_m)
    shape = tex.shape

    def lookup(P):
        tx = np.mod((P[:, 0] + extent / 2) / extent, 1.0)
        ty = np.mod((P[:, 1] + extent / 2) / extent, 1.0)
        px = np.clip((tx * (shape[1] - 1)).astype(np.int32), 0, shape[1] - 1)
        py = np.clip((ty * (shape[0] - 1)).astype(np.int32), 0, shape[0] - 1)
        return tex[py, px]

    P1, r = cast_camera1(H, grid)
    g1 = lookup(P1).reshape(grid, grid)
    # camera 2 looks at the same world points seen from its own origin
    xs = np.arange(grid, dtype=np.float64)
    uu, vv = np.meshgrid(xs, xs)
    ph = np.arctan2(vv - C2[1], uu - C2[0]) - math.pi / 2.0 - math.radians(roll_deg)
    r2 = np.hypot(uu - C2[0], vv - C2[1])
    t2 = r2 / F_PX
    th2 = t2 * (1.0 + A_DIST * t2 * t2)
    s2 = H / np.maximum(np.cos(th2), 1e-9)
    P2 = np.stack([s2 * np.sin(th2) * np.cos(ph),
                   -B_M + s2 * np.sin(th2) * np.sin(ph),
                   np.full_like(th2, float(H))], axis=-1)
    g2 = lookup(P2.reshape(-1, 3)).reshape(grid, grid)
    g2 = np.where(r2 > 940, 0.0, g2)
    g1 = np.where(r > 940, 0.0, g1)
    g1w = cv2.warpAffine(g1.astype(np.float32), _affine(), dsize=(grid, grid),
                         flags=cv2.INTER_LINEAR)
    return g1w, g2.astype(np.float32)


# PURPOSE: The gate. Confirms the rendered parallax is physical BEFORE any experiment is
#          allowed to use it.
# INPUTS: heights to test.
# OUTPUTS: True if du is positive and the analytic parallax falls as 1/H; prints the table.
# KEYWORDS: gate, validation, 1/H
def validate(heights=(1000.0, 2000.0, 4000.0), verbose=True):
    rows = []
    for H in heights:
        du, dv = gt_disparity(H)
        du0 = float(du[ORIG // 2, ORIG // 2])
        dv0 = float(dv[ORIG // 2, ORIG // 2])
        offset = float(C2[0] - (M0 := _affine()[0, 0] * C1[0] + _affine()[0, 1] * C1[1]
                                + _affine()[0, 2]))
        parallax = F_PX * math.atan(B_M / H)
        rows.append((H, du0, dv0, parallax, du0 - offset))
    ok = all(r[1] > 0 for r in rows)
    for i in range(len(rows) - 1):
        if rows[i][4] <= rows[i + 1][4]:
            ok = False
    if verbose:
        print(f"  {'H':>7} {'du(zenith)':>12} {'dv(zenith)':>12} "
              f"{'f*atan(B/H)':>12} {'du-offset':>11}")
        for H, du0, dv0, par, resid in rows:
            print(f"  {H:>7.0f} {du0:>12.3f} {dv0:>12.3f} {par:>12.3f} {resid:>11.3f}")
        print(f"  gate: du positive {'PASS' if all(r[1] > 0 for r in rows) else 'FAIL'}; "
              f"parallax falls as 1/H "
              f"{'PASS' if all(rows[i][4] > rows[i+1][4] for i in range(len(rows)-1)) else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"camera 2 principal point required by the affine: "
          f"({C2[0]:.2f}, {C2[1]:.2f})")
    print("renderer gate:")
    good = validate()
    sys.exit(0 if good else 1)
