# PURPOSE: Measure the residual rectification roll separately for each session, to
#          answer two questions at once:
#            * is the roll a stable property of the rig, or does it drift between
#              sessions? (W1's session-drift deliverable)
#            * can one correction be applied to both sessions, or does each need
#              its own?
#
#          Two independent estimators are computed per session and must agree:
#            * MODEL fit  — the exact plane + roll + translation model of
#              epipolar_check.py, fitted per frame and summarised by the median;
#            * SPREAD fit — the rotation of camera 2 that minimises the interquartile
#              range of the horizontal disparity over the whole session, i.e. the
#              angle that most nearly flattens the cloud field. This never uses a
#              plane model; it only assumes a cloud layer has small intrinsic spread.
#
#          Neither estimator consults the ceilometer or any ground truth.
# INPUTS: images under ~/clouds_data/images, the fixed calibration matrix, a date.
# OUTPUTS: per-session roll estimates; experiments/w1/roll_by_session.json.
# KEYWORDS: calibration, roll, drift, session, stability, self_calibration
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

from clouds.config import load_config                                   # noqa: E402
from clouds.geometry.imaging import center_crop                         # noqa: E402
from clouds.matching.filters import get_corresponding_feature_points    # noqa: E402
from epipolar_check import fit_frame, CACHE as CACHE_23                 # noqa: E402

CACHE_DIR = HERE


# PURPOSE: Collect rectified correspondences for one session and cache them.
# INPUTS: cfg, date, step (frame stride), limit.
# OUTPUTS: path to the written .npz cache.
# KEYWORDS: cache, correspondences, session
def build_cache(cfg, date, step=12, limit=40):
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    full = cfg.camera.original_size[0]
    off = (full - crop) / 2.0
    shift = cfg.dataset.clock_offset_h
    t0 = (dt.datetime.strptime(cfg.dataset.window_start, "%H:%M:%S")
          - dt.timedelta(hours=shift)).strftime("%H-%M-%S")
    t1 = (dt.datetime.strptime(cfg.dataset.window_end, "%H:%M:%S")
          - dt.timedelta(hours=shift)).strftime("%H-%M-%S")
    names = sorted(n for n in os.listdir(cfg.dataset.images_dir)
                   if n.endswith("devID1.jpg") and n.startswith("img-" + date))
    picked = [n for n in names if t0 <= n[15:23] <= t1][::step][:limit]
    kp1s, kp2s, stamps = [], [], []
    for n in picked:
        base = n.replace("devID1.jpg", "")
        p2 = os.path.join(cfg.dataset.images_dir, base + "devID2.jpg")
        if not os.path.exists(p2):
            continue
        i1 = cv2.imread(os.path.join(cfg.dataset.images_dir, n))
        i2 = cv2.imread(p2)
        if i1 is None or i2 is None:
            continue
        w = cv2.warpAffine(i1, M, dsize=cfg.camera.original_size)
        c1 = center_crop(w, crop, crop)
        c2 = center_crop(i2, crop, crop)
        k1, k2 = get_corresponding_feature_points(
            c1, c2, visualize=False, match_thresh=cfg.matching.match_threshold,
            backend=cfg.matching.backend, filter_config=cfg.matching.filter)
        if len(k1) < 100:
            continue
        kp1s.append(k1 + off)
        kp2s.append(k2 + off)
        stamps.append(base.replace("img-", ""))
        print(f"  {date} cached {len(kp1s):3d}  {base}  matches={len(k1)}")
    path = os.path.join(CACHE_DIR, f"matches_cache_{date}.npz")
    np.savez_compressed(path, kp1=np.array(kp1s, dtype=object),
                        kp2=np.array(kp2s, dtype=object),
                        stamps=np.array(stamps))
    return path


# PURPOSE: Rotate image points about a centre.
# INPUTS: uv (N,2), angle in degrees, centre (2,).
# OUTPUTS: (N,2) rotated points.
# KEYWORDS: rotation, roll_correction
def rotate(uv, deg, centre=(960.0, 960.0)):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (uv - np.asarray(centre, float)) @ R.T + np.asarray(centre, float)


# PURPOSE: Find the rotation of camera 2 that minimises the interquartile range of
#          the session's horizontal disparity.
# INPUTS: kp1s, kp2s, search range and step.
# OUTPUTS: dict with the optimal angle, the resulting IQR, and the curve.
# KEYWORDS: self_calibration, spread_minimisation, roll
def spread_optimum(kp1s, kp2s, lo=-4.0, hi=4.0, step=0.25):
    curve = []
    for ang in np.arange(lo, hi + 1e-9, step):
        du = np.concatenate([k1[:, 0] - rotate(k2, ang)[:, 0]
                             for k1, k2 in zip(kp1s, kp2s)])
        q1, q3 = np.percentile(du, [25, 75])
        curve.append((float(ang), float(q3 - q1), float(np.median(du))))
    best = min(curve, key=lambda t: t[1])
    # quadratic refinement around the minimum
    i = curve.index(best)
    if 0 < i < len(curve) - 1:
        x = np.array([curve[i - 1][0], curve[i][0], curve[i + 1][0]])
        y = np.array([curve[i - 1][1], curve[i][1], curve[i + 1][1]])
        try:
            p = np.polyfit(x, y, 2)
            if p[0] > 0:
                refined = -p[1] / (2 * p[0])
                if lo <= refined <= hi:
                    best = (float(refined), best[1], best[2])
        except np.linalg.LinAlgError:
            pass
    return {"angle_deg": best[0], "iqr_px": best[1], "median_du_px": best[2],
            "curve": curve}


# PURPOSE: Run both estimators for every session and report drift.
# INPUTS: none (config path fixed).
# OUTPUTS: table on stdout, experiments/w1/roll_by_session.json.
# KEYWORDS: drift, report, session
def main():
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026.yaml"))
    f = cfg.camera.image_width_px / math.pi
    B = cfg.camera.stereo_base_m
    out = {}
    for date in ("2025-05-22", "2025-05-23"):
        cache = os.path.join(CACHE_DIR, f"matches_cache_{date}.npz")
        if not os.path.exists(cache):
            print(f"building cache for {date}...")
            build_cache(cfg, date)
        d = np.load(cache, allow_pickle=True)
        kp1s, kp2s = d["kp1"], d["kp2"]
        n_match = sum(len(k) for k in kp1s)
        print(f"\n=== {date} ===  frames={len(kp1s)}  matches={n_match}")
        if len(kp1s) < 5 or n_match < 2000:
            # A clear sky gives LoFTR nothing to match, so the session carries no
            # stereo signal at all. Reported as such rather than crashing or
            # silently producing a meaningless roll.
            print(f"  UNUSABLE for stereo: {n_match} correspondences over {len(kp1s)} frames.")
            print("  A cloud-free sky has no texture to match; no calibration quantity can be")
            print("  estimated from this session.")
            out[date] = {"unusable": True, "n_frames": len(kp1s), "n_matches": int(n_match)}
            continue
        fits = [fit_frame(k1, k2, f, B) for k1, k2 in zip(kp1s, kp2s)]
        rolls = np.array([r["roll_deg"] for r in fits])
        iqr0 = np.concatenate([k1[:, 0] - k2[:, 0] for k1, k2 in zip(kp1s, kp2s)])
        q1, q3 = np.percentile(iqr0, [25, 75])
        sp = spread_optimum(kp1s, kp2s)
        print(f"  model fit   : median roll {np.median(rolls):+.2f} deg "
              f"(std {rolls.std():.2f}, range {rolls.min():+.2f}..{rolls.max():+.2f})")
        print(f"  spread fit  : optimum {sp['angle_deg']:+.2f} deg")
        print(f"  disparity IQR: as shipped {q3-q1:.2f} px -> corrected {sp['iqr_px']:.2f} px "
              f"({(q3-q1)/sp['iqr_px']:.1f}x tighter)")
        print(f"  residual rms with full model: {np.median([r['rms_full_px'] for r in fits]):.2f} px")
        out[date] = {
            "model_roll_median_deg": float(np.median(rolls)),
            "model_roll_std_deg": float(rolls.std()),
            "model_roll_min_deg": float(rolls.min()),
            "model_roll_max_deg": float(rolls.max()),
            "spread_optimum_deg": sp["angle_deg"],
            "iqr_uncorrected_px": float(q3 - q1),
            "iqr_corrected_px": sp["iqr_px"],
            "n_frames": len(kp1s),
        }
    both = [v for v in out.values() if not v.get("unusable")]
    if len(both) == 2:
        a, b = out["2025-05-22"], out["2025-05-23"]
        print("\n=== session drift ===")
        print(f"  model roll : {a['model_roll_median_deg']:+.2f} deg vs "
              f"{b['model_roll_median_deg']:+.2f} deg  "
              f"(difference {abs(a['model_roll_median_deg']-b['model_roll_median_deg']):.2f} deg)")
        print(f"  spread fit : {a['spread_optimum_deg']:+.2f} deg vs "
              f"{b['spread_optimum_deg']:+.2f} deg  "
              f"(difference {abs(a['spread_optimum_deg']-b['spread_optimum_deg']):.2f} deg)")
    else:
        print("\n=== session drift ===")
        print("  NOT MEASURABLE: only one session carries usable stereo signal.")
    json.dump(out, open(os.path.join(HERE, "roll_by_session.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'roll_by_session.json')}")


if __name__ == "__main__":
    main()
