# PURPOSE: Convert matched image points to cloud heights using the exact
#          projection geometry instead of the paraxial formula
#          D = xi0*B/(pi*|du|) that the pipeline currently uses.
#
#          Two things are wrong with the paraxial formula and both are fixed here:
#            * it is the small-angle limit, so it is biased away from the optical
#              axis. Measured on this rig, the bias reaches +11.7 % at the corner
#              of the 640 px crop for a 1 km cloud (docs/plan/stage-audit.md 2.1);
#            * it uses only the horizontal component of the disparity and throws
#              the vertical component away, although that component is physically
#              non-zero off-axis and carries information.
#
#          The correct treatment is to turn each image point into a unit ray in
#          the camera frame (exactly, using the equidistant fisheye mapping), and
#          intersect the two rays. The intersection residual is returned because
#          it is the one ground-truth-free quality signal the geometry itself
#          provides: a correspondence that does not triangulate is a bad match or
#          a point not on the assumed surface.
# INPUTS: image coordinates in both rectified frames, the focal constant f in
#         px/rad, the baseline in metres, the principal point.
# OUTPUTS: per-match heights, ranges, and intersection residuals.
# KEYWORDS: triangulation, fisheye, equidistant, unit_sphere, height, uncertainty
from __future__ import annotations

import numpy as np


# PURPOSE: Turn image points into unit viewing rays for an equidistant fisheye.
#          For this projection the image radius is proportional to the angle from
#          the optical axis, r = f*theta, which is what makes the mapping exactly
#          invertible with a single constant.
# INPUTS: uv (np.ndarray Nx2) image coordinates; f (float) px/rad; centre (2,);
#         distortion_a (float) cubic radial distortion coefficient, 0 = ideal
#         equidistant. A large negative value can in principle make theta
#         non-monotonic; the derivative is 1 + 3a t^2, so a > -1/(3 t_max^2) keeps
#         the mapping monotonic over the field.
# OUTPUTS: (N,3) array of unit vectors in the camera frame, +z along the axis.
#          A point exactly at the principal point yields (0,0,1). Zero radius
#          does not divide by zero.
# KEYWORDS: projection, fisheye, equidistant, ray
def pixels_to_rays(uv: np.ndarray, f: float, centre=(960.0, 960.0),
                   distortion_a: float = 0.0) -> np.ndarray:
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    d = uv - np.asarray(centre, dtype=float)
    r = np.hypot(d[:, 0], d[:, 1])
    t = r / f
    # A real fisheye deviates from the ideal equidistant law by a smooth radial
    # term. The leading deviation is cubic:
    #     theta = (r/f) * (1 + a * (r/f)^2)
    # The coefficient was measured from the sky itself
    # (experiments/layers/distortion_fit.py): with a = -0.30 the apparent height
    # becomes independent of image radius (radial spread 36.7 % -> 3.2 %), which a
    # flat cloud layer requires. a = 0 reproduces the ideal law exactly.
    theta = t * (1.0 + float(distortion_a) * t * t)
    s = np.sin(theta)
    c = np.cos(theta)
    # At r = 0 the azimuth is undefined but sin(theta) = 0, so any unit vector in
    # the plane works; use +x to avoid a division by zero.
    safe = np.where(r > 0, r, 1.0)
    return np.stack([s * d[:, 0] / safe, s * d[:, 1] / safe, c], axis=1)


# PURPOSE: Intersect two rays and return the range along each, the 3-D point and
#          the miss distance between the rays.
# INPUTS: d1, d2 (N,3) unit rays; origin2 (3,) origin of the second ray in the
#         first camera's frame (the baseline); origin1 is the first camera.
# OUTPUTS: dict with t1, t2 (range along each ray), points (N,3) the midpoint, and
#          residual (N,) the distance between the two closest points. When the
#          rays are parallel (a scene point at infinity, or a false match with no
#          disparity) the denominator vanishes; those entries get NaN ranges
#          rather than an exception or a silent infinity.
# KEYWORDS: triangulation, closest_approach, residual, depth
def triangulate(d1: np.ndarray, d2: np.ndarray, origin2=(45.0, 0.0, 0.0)) -> dict:
    d1 = np.asarray(d1, dtype=float).reshape(-1, 3)
    d2 = np.asarray(d2, dtype=float).reshape(-1, 3)
    w = np.asarray([0.0, 0.0, 0.0]) - np.asarray(origin2, dtype=float)   # O1 - O2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = d1 @ w
    e = d2 @ w
    denom = a * c - b * b
    degenerate = np.abs(denom) < 1e-12
    safe = np.where(degenerate, 1.0, denom)
    t1 = (b * e - c * d) / safe
    t2 = (a * e - b * d) / safe
    t1 = np.where(degenerate, np.nan, t1)
    t2 = np.where(degenerate, np.nan, t2)
    p1 = t1[:, None] * d1
    p2 = np.asarray(origin2, dtype=float)[None, :] + t2[:, None] * d2
    mid = 0.5 * (p1 + p2)
    residual = np.linalg.norm(p1 - p2, axis=1)
    return {"t1": t1, "t2": t2, "points": mid, "residual": residual}


# PURPOSE: Triangulate a set of matches and return heights in metres.
# INPUTS: kp1, kp2 (N,2) matched image coordinates in the two rectified frames;
#         f (float) px/rad; baseline (float) metres; centre (2,) of the *common*
#         rectified frame the coordinates are expressed in; distortion_a (float)
#         the cubic radial term, 0 for the ideal equidistant law.
# OUTPUTS: dict with heights (N,), residual (N,), valid (N,) boolean mask. Ray
#          pairs that are parallel or behind either camera are marked invalid and
#          their height is NaN, so a caller cannot accidentally average a
#          non-measurement.
# KEYWORDS: triangulation, cloud_base_height, validity
def heights_from_matches(kp1: np.ndarray, kp2: np.ndarray, f: float,
                         baseline: float, centre=(960.0, 960.0),
                         distortion_a: float = 0.0) -> dict:
    r1 = pixels_to_rays(kp1, f, centre, distortion_a)
    r2 = pixels_to_rays(kp2, f, centre, distortion_a)
    tri = triangulate(r1, r2, origin2=(baseline, 0.0, 0.0))
    z = tri["points"][:, 2]
    valid = (np.isfinite(z) & (z > 0) & (tri["t1"] > 0) & (tri["t2"] > 0)
             & np.isfinite(tri["residual"]))
    heights = np.where(valid, z, np.nan)
    return {"heights": heights, "residual": tri["residual"], "valid": valid,
            "ranges": tri["t1"]}


# PURPOSE: Reproduce what the current pipeline computes, for comparison.
# INPUTS: kp1, kp2 (N,2); image width in pixels; baseline in metres.
# OUTPUTS: (N,) heights from the paraxial formula D = xi0*B/(pi*|du|). Matches
#          with zero horizontal disparity give inf rather than raising.
# KEYWORDS: paraxial, baseline_formula, comparison
def paraxial_heights(kp1: np.ndarray, kp2: np.ndarray, image_width: float,
                     baseline: float) -> np.ndarray:
    du = np.asarray(kp1, dtype=float)[:, 0] - np.asarray(kp2, dtype=float)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = image_width * baseline / (np.pi * np.abs(du))
    return np.where(np.abs(du) < 1e-9, np.inf, h)
