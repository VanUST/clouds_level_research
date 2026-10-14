# PURPOSE: Shared machinery for the W3 matcher comparison: pick real cloud frames
#          from a session, rectify them into the exact crops the pipeline feeds the
#          matcher, and compute the ground-truth-free criteria the comparison is
#          pre-registered on (triangulation residual, epipolar structure, radial
#          flatness, perturbation stability). Every criterion here is a function of
#          image pairs and the fixed calibration alone — the ceilometer is never
#          read, so guardrail G1 holds by construction.
# INPUTS: a validated Config, image paths, (kp1, kp2) correspondence arrays, the
#         focal constant f in px/rad and the baseline in metres.
# OUTPUTS: plain dicts of scalars / arrays that are JSON-serialisable apart from
#          the per-bin radial profiles, which the caller converts.
# KEYWORDS: matcher_evaluation, criteria, triangulation, epipolar, radial_flatness
from __future__ import annotations

import datetime as dt
import math
import os
import sys

import cv2
import numpy as np

REPO = "/home/iustimov/clouds"
sys.path.insert(0, os.path.join(REPO, "src"))

from clouds.config import load_config                       # noqa: E402
from clouds.geometry.imaging import center_crop, rotation_about  # noqa: E402
from clouds.geometry.triangulation import heights_from_matches   # noqa: E402

CONFIG_PATH = os.path.join(REPO, "configs", "paper_dlcp2026.yaml")
IMAGES_DIR = "/home/iustimov/clouds_data/images"

# Crop offset between the 640x640 crop frame and the full 1920x1920 frame, which
# is the frame the triangulation's principal point (960, 960) lives in.
CROP = 640
FULL = 1920
OFF = (FULL - CROP) / 2.0
CENTRE = (FULL / 2.0, FULL / 2.0)


# PURPOSE: Load the project config once for every experiment in this directory.
# INPUTS: an optional path; defaults to configs/paper_dlcp2026.yaml.
# OUTPUTS: a frozen clouds.config.Config.
# KEYWORDS: config, load
def load_cfg(path=None):
    return load_config(path or CONFIG_PATH)


# PURPOSE: Convert an image filename stamp (img-YYYY-MM-DDTHH-MM-SSdevIDn.jpg)
#          into seconds since midnight, so frames can be selected by time.
# INPUTS: filename (str).
# OUTPUTS: int seconds, or None when the name does not carry a stamp.
# KEYWORDS: naming, timestamp
def stamp_seconds(name):
    s = name[4:23]
    try:
        return (int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19]))
    except (ValueError, IndexError):
        return None


# PURPOSE: Choose a fixed, evenly spaced set of real stereo pairs from one
#          session. The same list is used for every matcher, so the comparison is
#          paired and no matcher sees an easier frame set than another.
# INPUTS: cfg; date (str, YYYY-MM-DD); t0, t1 (HH:MM:SS, image/filename clock);
#         step_s (int) spacing between chosen frames; limit (int) cap.
# OUTPUTS: list of (path_devID1, path_devID2, stamp) sorted by time, one entry per
#          chosen grid time, empty when no frame is close enough.
# KEYWORDS: frame_selection, dataset, paired
def select_frames(cfg, date, t0="08:30:00", t1="16:30:00", step_s=900, limit=40):
    d = cfg.dataset.images_dir or IMAGES_DIR
    names = sorted(n for n in os.listdir(d)
                   if n.startswith("img-" + date) and n.endswith("devID1.jpg"))
    have = {}
    for n in names:
        sec = stamp_seconds(n)
        have[sec] = n
    a = _hm(t0)
    b = _hm(t1)
    want = list(range(a, b + 1, step_s))
    picked = []
    for w in want:
        sec = min(have, key=lambda s: abs(s - w))
        if abs(sec - w) > step_s / 2.0:
            continue
        n = have[sec]
        p2 = os.path.join(d, n.replace("devID1.jpg", "devID2.jpg"))
        if os.path.exists(p2):
            picked.append((os.path.join(d, n), p2, n[4:23]))
    picked = picked[:limit]
    return picked


def _hm(t):
    hh, mm, ss = (int(x) for x in t.split(":"))
    return hh * 3600 + mm * 60 + ss


# PURPOSE: Reproduce the pipeline's exact preparation of one stereo pair: rectify
#          camera 1 into camera 2's frame with the fixed affine (optionally the
#          measured residual roll on camera 2), then centre-crop both. The matcher
#          must see identical arrays to the ones the pipeline feeds it.
# INPUTS: cfg; img1_path, img2_path (str); affine (2x3 or None = config's);
#         roll_deg (float or None = config's roll_correction_deg).
# OUTPUTS: (gray1, gray2, bgr1, bgr2), all 640x640 uint8. (None,...) when a file
#          cannot be read.
# KEYWORDS: rectification, crop, preprocessing
def prepare_crop(cfg, img1_path, img2_path, affine=None, roll_deg=None):
    i1 = cv2.imread(img1_path)
    i2 = cv2.imread(img2_path)
    if i1 is None or i2 is None:
        return None
    aff = np.asarray(cfg.calibration_record.matrix_2x3, dtype=float) if affine is None else affine
    roll = float(getattr(cfg.camera, "roll_correction_deg", 0.0) or 0.0) if roll_deg is None else roll_deg
    if roll:
        i2 = cv2.warpAffine(i2, rotation_about(CENTRE, roll), dsize=(FULL, FULL))
    w1 = cv2.warpAffine(i1, aff, dsize=(FULL, FULL))
    c1 = center_crop(w1, CROP, CROP)
    c2 = center_crop(i2, CROP, CROP)
    return (cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY),
            c1, c2)


# PURPOSE: Shift an image by a known sub-pixel amount with bilinear interpolation,
#          used to measure how repeatable a matcher's localisation is under a small
#          perturbation of one image of the pair (the real pair is ~1 s apart, so
#          the scene is not identical between the two cameras either).
# INPUTS: gray (H,W) uint8; dx, dy (float) pixels.
# OUTPUTS: (H,W) uint8 shifted image, same size (border replicated by the affine
#          warp's default constant fill).
# KEYWORDS: perturbation, subpixel, repeatability
def subpixel_shift(gray, dx, dy):
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(gray, m, dsize=(gray.shape[1], gray.shape[0]),
                          flags=cv2.INTER_LINEAR)


# PURPOSE: Predict where a match seen in camera 1 appears in camera 2 for a
#          horizontal plane at height D, with an additional roll of camera 2 about
#          the principal point. This is the exact (non-paraxial) forward model of
#          the same geometry the project triangulates with.
# INPUTS: uv1 (N,2) full-frame coordinates; f (float) px/rad; baseline (m);
#         D (float) plane height (m); roll_deg (float); centre (2,).
# OUTPUTS: (N,2) predicted full-frame positions in camera 2.
# KEYWORDS: forward_model, plane, roll, fisheye
def predict_uv2(uv1, f, baseline, D, roll_deg, centre=CENTRE):
    d = uv1 - np.asarray(centre, float)
    r = np.hypot(d[:, 0], d[:, 1])
    th = r / f
    safe = np.where(r > 0, r, 1.0)
    z = np.cos(th)
    x = np.sin(th) * d[:, 0] / safe
    y = np.sin(th) * d[:, 1] / safe
    P = np.stack([D * x / z, D * y / z, np.full_like(z, D)], axis=1)
    P2 = P.copy()
    P2[:, 0] -= baseline
    n = np.linalg.norm(P2, axis=1)
    th2 = np.arccos(np.clip(P2[:, 2] / n, -1, 1))
    ph2 = np.arctan2(P2[:, 1], P2[:, 0])
    u2 = centre[0] + f * th2 * np.cos(ph2)
    v2 = centre[1] + f * th2 * np.sin(ph2)
    a = math.radians(roll_deg)
    du, dv = u2 - centre[0], v2 - centre[1]
    return np.stack([centre[0] + math.cos(a) * du - math.sin(a) * dv,
                     centre[1] + math.sin(a) * du + math.cos(a) * dv], axis=1)


# PURPOSE: Fit the plane-shift model (D, roll, tx, ty) to one frame's matches and
#          report the residual that the exact geometry cannot explain. This is the
#          sub-pixel consistency criterion: a matcher that localises better leaves
#          less unexplained shift. The translation is solved for analytically at
#          each (D, roll) as the median residual, which keeps the searched grid 2-D.
# INPUTS: kp1, kp2 (N,2) crop-frame matches; f; baseline.
# OUTPUTS: dict with D, roll_deg, tx, ty and rms residual (px) for the full model
#          and for reduced models (no roll, translation only, plane only), plus the
#          median 3-D ray miss distance in metres from clouds.geometry.triangulation.
# KEYWORDS: fit, plane, roll, residual, subpixel
def plane_fit(kp1, kp2, f, baseline):
    uv1 = np.asarray(kp1, float) + OFF
    uv2 = np.asarray(kp2, float) + OFF
    if len(uv1) < 20:
        return {"n": int(len(uv1)), "D": float("nan"), "roll_deg": float("nan"),
                "tx": float("nan"), "ty": float("nan"),
                "rms_full_px": float("nan"), "med_abs_res_px": float("nan"),
                "rms_trimmed_px": float("nan"), "trim_fraction": float("nan"),
                "rms_no_roll_px": float("nan"),
                "rms_plane_only_px": float("nan"), "ray_miss_median_m": float("nan")}
    f = float(f)
    baseline = float(baseline)

    def solve(D, roll):
        res = uv2 - predict_uv2(uv1, f, baseline, D, roll)
        tx = float(np.median(res[:, 0]))
        ty = float(np.median(res[:, 1]))
        d = res - np.array([tx, ty])
        return float(np.sqrt(np.mean(np.sum(d * d, axis=1)))), tx, ty

    best = (None, None, None, None, np.inf)
    for D in range(400, 6001, 200):
        for roll in np.arange(-6.0, 6.01, 1.0):
            rms, tx, ty = solve(D, roll)
            if rms < best[4]:
                best = (D, roll, tx, ty, rms)
    D0, r0 = best[0], best[1]
    for D in np.arange(max(100, D0 - 200), D0 + 201, 50):
        for roll in np.arange(r0 - 1.0, r0 + 1.01, 0.25):
            rms, tx, ty = solve(D, roll)
            if rms < best[4]:
                best = (D, roll, tx, ty, rms)
    D0, r0 = best[0], best[1]
    for D in np.arange(max(100, D0 - 50), D0 + 51, 10):
        for roll in np.arange(r0 - 0.25, r0 + 0.251, 0.05):
            rms, tx, ty = solve(D, roll)
            if rms < best[4]:
                best = (D, roll, tx, ty, rms)
    D0, r0, tx, ty, rms = best

    def rms_at(D, roll, txy):
        pred = predict_uv2(uv1, f, baseline, D, roll) + np.asarray(txy, float)
        return float(np.sqrt(np.mean(np.sum((uv2 - pred) ** 2, axis=1))))

    t_only = np.array([np.median(uv2[:, 0] - uv1[:, 0]), np.median(uv2[:, 1] - uv1[:, 1])])
    tri = heights_from_matches(uv1, uv2, f, baseline, CENTRE)
    miss = tri["residual"][np.isfinite(tri["residual"])]
    # Robust companions to the rms: different backends threshold confidence on
    # different scales, so a matcher may return more low-confidence outliers for a
    # reason unrelated to localisation. The median absolute residual and the
    # MAD-trimmed rms locate the bulk of the field instead of its tail.
    pred = predict_uv2(uv1, f, baseline, D0, r0) + np.array([tx, ty])
    res = np.linalg.norm(uv2 - pred, axis=1)
    med_abs = float(np.median(res))
    mad = float(np.median(np.abs(res - med_abs)))
    trim = res <= med_abs + 3.0 * 1.4826 * mad if mad > 0 else np.ones(len(res), bool)
    return {"n": int(len(uv1)), "D": float(D0), "roll_deg": float(r0),
            "tx": float(tx), "ty": float(ty),
            "rms_full_px": float(rms),
            "med_abs_res_px": med_abs,
            "rms_trimmed_px": float(np.sqrt(np.mean(res[trim] ** 2))) if trim.any() else float("nan"),
            "trim_fraction": float(trim.mean()),
            "rms_no_roll_px": rms_at(D0, 0.0, (tx, ty)),
            "rms_plane_only_px": rms_at(D0, 0.0, (0.0, 0.0)),
            "rms_translation_only_px": rms_at(D0, 0.0, t_only),
            "ray_miss_median_m": float(np.median(miss)) if len(miss) else float("nan")}


# PURPOSE: Split the vertical disparity after rectification into a smooth
#          (structured) part — an un-modelled geometry term — and a random part —
#          localisation noise — by least-squares fitting a degree-3 bivariate
#          polynomial in the image coordinates. The random part is the quantity a
#          better matcher should shrink; the structured part is shared geometry.
# INPUTS: kp1, kp2 (N,2) crop-frame matches.
# OUTPUTS: dict with total rms of dv, rms of the smooth component, rms of the
#          residual, and the median dv (bias). Returns NaNs for fewer than 20 pts.
# KEYWORDS: epipolar, vertical_disparity, structured, random, polynomial
def epipolar_split(kp1, kp2):
    uv = np.asarray(kp1, float)
    dv = np.asarray(kp1, float)[:, 1] - np.asarray(kp2, float)[:, 1]
    if len(uv) < 20:
        return {"dv_rms_px": float("nan"), "dv_struct_rms_px": float("nan"),
                "dv_rand_rms_px": float("nan"), "dv_median_px": float("nan")}
    u = (uv[:, 0] - CROP / 2.0) / (CROP / 2.0)
    v = (uv[:, 1] - CROP / 2.0) / (CROP / 2.0)
    cols = []
    for i in range(4):
        for j in range(4 - i):
            cols.append((u ** i) * (v ** j))
    X = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(X, dv, rcond=None)
    smooth = X @ beta
    rand = dv - smooth
    return {"dv_rms_px": float(np.sqrt(np.mean(dv ** 2))),
            "dv_struct_rms_px": float(np.sqrt(np.mean(smooth ** 2))),
            "dv_rand_rms_px": float(np.sqrt(np.mean(rand ** 2))),
            "dv_median_px": float(np.median(dv))}


# PURPOSE: Measure the radial flatness of the height field: a flat cloud layer of
#          constant height must give the same height at every radius from the
#          principal point, so the flattest measured profile (after differences in
#          the cloud seen cancel) is the preferable matcher. Robust statistics are
#          used because heights from near-zero disparity are heavy-tailed.
# INPUTS: kp1, kp2 (N,2) crop-frame matches; f; baseline; nbins (int); rmax (px).
# OUTPUTS: dict with median/IQR of height, the slope of binned median height against
#          radius (m/px; smaller absolute value = flatter), the spread of the bin
#          medians, and the per-bin arrays for the figure.
# KEYWORDS: radial_flatness, height_profile, robust, cloud_layer
def radial_profile(kp1, kp2, f, baseline, nbins=12, rmax=452.0):
    uv1 = np.asarray(kp1, float) + OFF
    uv2 = np.asarray(kp2, float) + OFF
    out = {"n": int(len(uv1)), "height_median_m": float("nan"),
           "height_iqr_m": float("nan"), "height_std_m": float("nan"),
           "slope_m_per_px": float("nan"), "bin_r": [], "bin_h": [], "bin_n": []}
    if len(uv1) < 20:
        return out
    tri = heights_from_matches(uv1, uv2, f, baseline, CENTRE)
    h = tri["heights"]
    keep = tri["valid"] & np.isfinite(h) & (h > 100.0) & (h < 50000.0)
    if keep.sum() < 20:
        return out
    h = h[keep]
    r = np.hypot(uv1[keep, 0] - CENTRE[0], uv1[keep, 1] - CENTRE[1])
    out["height_median_m"] = float(np.median(h))
    out["height_iqr_m"] = float(np.percentile(h, 75) - np.percentile(h, 25))
    out["height_std_m"] = float(np.std(h))
    edges = np.linspace(0.0, rmax, nbins + 1)
    centres, med = [], []
    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() >= 5:
            centres.append(float(0.5 * (edges[i] + edges[i + 1])))
            med.append(float(np.median(h[m])))
        else:
            centres.append(float(0.5 * (edges[i] + edges[i + 1])))
            med.append(float("nan"))
        out["bin_n"].append(int(m.sum()))
    out["bin_r"] = centres
    out["bin_h"] = med
    good = np.isfinite(med)
    if good.sum() >= 3:
        slope = float(np.polyfit(np.asarray(centres)[good], np.asarray(med)[good], 1)[0])
        out["slope_m_per_px"] = slope
        out["bin_spread_m"] = float(np.nanmax(med) - np.nanmin(med))
    return out


# PURPOSE: Build the low- and high-gradient masks of an image with the thresholds
#          fixed by the pre-registration, so a match can be attributed to a cloud
#          interior (smooth) or a cloud edge. Split out from gradient_strata so the
#          plane fit can be recomputed on each stratum separately — the metric that
#          answers whether a matcher is accurate where there is no texture.
# INPUTS: gray (H,W) uint8; lo_pct, hi_pct percentile thresholds.
# OUTPUTS: (lo_mask, hi_mask, mag) boolean masks and the smoothed gradient.
# KEYWORDS: gradient, masks, cloud_interior, strata
def gradient_masks(gray, lo_pct=33.0, hi_pct=67.0):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.boxFilter(np.hypot(gx, gy), -1, (9, 9))
    return mag <= float(np.percentile(mag, lo_pct)), mag >= float(np.percentile(mag, hi_pct)), mag


# PURPOSE: Attribute matches to a low- or high-gradient stratum of the image they
#          were found in, so coverage can be reported separately for cloud
#          interiors (smooth, no local texture) and cloud edges. A matcher that
#          only survives on edges is not usable here: the height is wanted across
#          the whole field, not on the rims of the clouds.
# INPUTS: gray (H,W) uint8 image 1; kp1 (N,2) crop-frame match positions.
# OUTPUTS: dict with the low-gradient match fraction, the low- and high-gradient
#          match densities per pixel, and the two stratum pixel counts. The
#          thresholds are fixed by the pre-registration: 33rd and 67th percentile
#          of the smoothed Sobel magnitude.
# KEYWORDS: gradient, coverage, cloud_interior, strata
def gradient_strata(gray, kp1, lo_pct=33.0, hi_pct=67.0):
    lo_mask, hi_mask, mag = gradient_masks(gray, lo_pct, hi_pct)
    lo, hi = float(np.percentile(mag, lo_pct)), float(np.percentile(mag, hi_pct))
    n_lo = int(lo_mask.sum())
    n_hi = int(hi_mask.sum())
    out = {"grad_lo_thresh": lo, "grad_hi_thresh": hi,
           "low_grad_pixels": n_lo, "high_grad_pixels": n_hi,
           "low_grad_frac": float("nan"), "low_grad_density": float("nan"),
           "high_grad_density": float("nan")}
    if len(kp1) == 0:
        return out
    kp = np.asarray(kp1, float)
    ui = np.clip(kp[:, 0].astype(int), 0, gray.shape[1] - 1)
    vi = np.clip(kp[:, 1].astype(int), 0, gray.shape[0] - 1)
    in_lo = lo_mask[vi, ui]
    in_hi = hi_mask[vi, ui]
    out["low_grad_frac"] = float(in_lo.mean())
    if n_lo:
        out["low_grad_density"] = float(in_lo.sum()) / n_lo
    if n_hi:
        out["high_grad_density"] = float(in_hi.sum()) / n_hi
    return out


# PURPOSE: Count how many of the matches in a perturbed run land on a nearly
#          identical point in image 1 of the unperturbed run, which is the subset
#          over which the perturbation-induced disparity change is meaningful.
# INPUTS: kp1_ref (N,2); kp1_pert (M,2); tol (px).
# OUTPUTS: (idx_ref, idx_pert) integer index arrays of the paired subset.
# KEYWORDS: pairing, nearest_neighbour, perturbation
def pair_by_image1(kp1_ref, kp1_pert, tol=1.5):
    from scipy.spatial import cKDTree
    if len(kp1_ref) == 0 or len(kp1_pert) == 0:
        return np.empty(0, int), np.empty(0, int)
    tree = cKDTree(kp1_pert)
    d, idx = tree.query(kp1_ref, distance_upper_bound=tol)
    ok = np.isfinite(d)
    return np.nonzero(ok)[0], idx[ok]
