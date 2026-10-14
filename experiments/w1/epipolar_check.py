# PURPOSE: Measure the residual relative ROLL between the two rectified images, and
#          the cloud height, from the disparity field alone — no Sun, no ground
#          truth, nothing at infinity. This is the scene-based calibration check
#          recommended in docs/plan/w1-calibration.md W1.10 route (b).
#
#          Why a roll is separable: for a calibrated horizontal-baseline pair the
#          shift field of a horizontal plane at height D is fully determined by the
#          geometry. A residual roll of the second camera about its optical axis
#          adds a displacement tangential to the image centre. The two have
#          different signatures in the field, so fitting both at once separates
#          them. The fit is over (D, roll) for the whole frame, and the residual it
#          leaves is the part explained by neither — real cloud relief plus match
#          noise.
#
#          Falsifiable prediction: if the ~4 degree gap reported in W1.10 is a real
#          calibration error, the fitted roll should be near 4 deg consistently
#          across frames. If it is a blob-shape artefact of the Sun estimator, the
#          fitted roll should be near zero. Either outcome settles the question.
# INPUTS: frames sampled from a session; LoFTR correspondences on the rectified pair.
# OUTPUTS: per-frame (D, roll) and the fit residual; a summary on stdout.
# KEYWORDS: calibration, roll, epipolar, relative_pose, scene_based, self_check
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

from clouds.config import load_config                       # noqa: E402
from clouds.geometry.imaging import center_crop             # noqa: E402
from clouds.matching.filters import get_corresponding_feature_points  # noqa: E402

CACHE = os.path.join(HERE, "matches_cache.npz")


# PURPOSE: Collect correspondences for a set of frames and cache them, because the
#          matcher is the expensive step and the fit below is cheap.
# INPUTS: cfg (Config), step (take every Nth frame), limit (max frames).
# OUTPUTS: writes CACHE with per-frame kp1/kp2 stacked, plus the index arrays.
# KEYWORDS: cache, correspondences, loftr
def build_cache(cfg, step=12, limit=40):
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    full = cfg.camera.original_size[0]
    off = (full - crop) / 2.0
    win = (dt.datetime.strptime(cfg.dataset.window_start, "%H:%M:%S")
           - dt.timedelta(hours=cfg.dataset.clock_offset_h)).strftime("%H-%M-%S")
    win_end = (dt.datetime.strptime(cfg.dataset.window_end, "%H:%M:%S")
               - dt.timedelta(hours=cfg.dataset.clock_offset_h)).strftime("%H-%M-%S")
    names = sorted(n for n in os.listdir(cfg.dataset.images_dir)
                   if n.endswith("devID1.jpg") and n.startswith("img-" + cfg.dataset.date))
    picked = []
    for n in names:
        stamp = n.replace("img-", "").replace("devID1.jpg", "")
        t = stamp[11:16].replace("-", "")
        if win <= stamp[11:16] <= win_end:
            picked.append(n)
    picked = picked[::step][:limit]
    kp1s, kp2s, stamps, frames = [], [], [], []
    for n in picked:
        base = n.replace("devID1.jpg", "")
        p1 = os.path.join(cfg.dataset.images_dir, n)
        p2 = os.path.join(cfg.dataset.images_dir, base + "devID2.jpg")
        if not os.path.exists(p2):
            continue
        i1 = cv2.imread(p1)
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
        frames.append(n)
        print(f"  cached {len(kp1s):3d}  {base}  matches={len(k1)}")
    np.savez_compressed(CACHE, kp1=np.array(kp1s, dtype=object),
                        kp2=np.array(kp2s, dtype=object),
                        stamps=np.array(stamps), frames=np.array(frames))
    print(f"wrote {CACHE}")


# PURPOSE: Predict where a match seen in camera 1 appears in camera 2, for a
#          horizontal plane at height D with an additional roll of camera 2.
# INPUTS: uv1 (N,2) in the rectified frame; f, baseline, plane height D, roll_deg.
# OUTPUTS: (N,2) predicted positions in camera 2's rectified frame.
# KEYWORDS: forward_model, plane, roll
def predict_uv2(uv1, f, baseline, D, roll_deg, centre=(960.0, 960.0)):
    d = uv1 - np.asarray(centre, float)
    r = np.hypot(d[:, 0], d[:, 1])
    th = r / f
    safe = np.where(r > 0, r, 1.0)
    z = np.cos(th)
    x = np.sin(th) * d[:, 0] / safe
    y = np.sin(th) * d[:, 1] / safe
    # scene point on the plane Z = D, along the ray from camera 1
    P = np.stack([D * x / z, D * y / z, np.full_like(z, D)], axis=1)
    P2 = P.copy()
    P2[:, 0] -= baseline
    n = np.linalg.norm(P2, axis=1)
    th2 = np.arccos(np.clip(P2[:, 2] / n, -1, 1))
    ph2 = np.arctan2(P2[:, 1], P2[:, 0])
    u2 = centre[0] + f * th2 * np.cos(ph2)
    v2 = centre[1] + f * th2 * np.sin(ph2)
    # roll of camera 2 about the principal point
    a = math.radians(roll_deg)
    du, dv = u2 - centre[0], v2 - centre[1]
    u2r = centre[0] + math.cos(a) * du - math.sin(a) * dv
    v2r = centre[1] + math.sin(a) * du + math.cos(a) * dv
    return np.stack([u2r, v2r], axis=1)


# PURPOSE: Fit (D, roll, tx, ty) for one frame.
#          The translation is separated from the roll deliberately: a translation
#          displaces every correspondence by the same vector, while a roll
#          displaces them tangentially, in proportion to the radius. They have
#          different signatures, so a fit that omits the translation would let the
#          roll absorb it, and the reported roll would not be a roll.
#          For any (D, roll) the optimal translation is the median of the residual
#          components, so the grid stays two-dimensional and cheap.
# INPUTS: uv1, uv2 (N,2) matches; f; baseline.
# OUTPUTS: dict with D, roll_deg, tx, ty and residual statistics. Also returns the
#          residual with roll and translation both forced to zero, and with the
#          translation alone, so the value of each term is visible.
# KEYWORDS: fit, grid_search, robust, roll, translation, degeneracy
def fit_frame(uv1, uv2, f, baseline):
    f = float(f)
    baseline = float(baseline)
    centre = np.array([960.0, 960.0])

    def solve(D, roll):
        pred = predict_uv2(uv1, f, baseline, D, roll)
        res = uv2 - pred
        tx = float(np.median(res[:, 0]))
        ty = float(np.median(res[:, 1]))
        cost = float(np.median(np.linalg.norm(res - np.array([tx, ty]), axis=1)))
        return cost, tx, ty

    best = (None, None, None, None, np.inf)
    for D in range(400, 6001, 200):
        for roll in np.arange(-6.0, 6.01, 1.0):
            c, tx, ty = solve(D, roll)
            if c < best[4]:
                best = (D, roll, tx, ty, c)
    D0, r0, _, _, _ = best
    for D in np.arange(max(100, D0 - 200), D0 + 201, 50):
        for roll in np.arange(r0 - 1.0, r0 + 1.01, 0.25):
            c, tx, ty = solve(D, roll)
            if c < best[4]:
                best = (D, roll, tx, ty, c)
    D0, r0, _, _, _ = best
    for D in np.arange(max(100, D0 - 50), D0 + 51, 10):
        for roll in np.arange(r0 - 0.25, r0 + 0.251, 0.05):
            c, tx, ty = solve(D, roll)
            if c < best[4]:
                best = (D, roll, tx, ty, c)

    D0, r0, tx, ty, c = best

    def rms(D, roll, txy):
        pred = predict_uv2(uv1, f, baseline, D, roll) + np.asarray(txy, float)
        return float(np.sqrt(np.mean(np.sum((uv2 - pred) ** 2, axis=1))))

    neg = np.zeros(2)
    t_only = np.array([np.median(uv2[:, 0] - uv1[:, 0]), np.median(uv2[:, 1] - uv1[:, 1])])
    return {"D": float(D0), "roll_deg": float(r0), "tx": float(tx), "ty": float(ty),
            "residual_px": float(c),
            "rms_full_px": rms(D0, r0, (tx, ty)),
            "rms_no_roll_px": rms(D0, 0.0, (tx, ty)),
            "rms_translation_only_px": rms(D0, 0.0, t_only),
            "rms_plain_plane_px": rms(D0, 0.0, neg)}


# PURPOSE: Load the cache and fit every frame.
# INPUTS: config path.
# OUTPUTS: list of per-frame fits; saves experiments/w1/roll_fit.json.
# KEYWORDS: sweep, report
def main():
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026.yaml"))
    f_used = cfg.camera.image_width_px / math.pi
    if not os.path.exists(CACHE):
        print("building correspondence cache (this runs the matcher)...")
        build_cache(cfg)
    data = np.load(CACHE, allow_pickle=True)
    kp1s, kp2s, stamps = data["kp1"], data["kp2"], data["stamps"]
    print(f"\nassumed f = {f_used:.1f} px/rad; baseline = {cfg.camera.stereo_base_m} m")
    print(f"{'time':>14} {'matches':>8} {'D (m)':>7} {'roll':>7} {'tx':>7} {'ty':>7} | "
          f"{'rms full':>9} {'no roll':>8} {'trans only':>11} {'plane only':>11}")
    out = []
    for k1, k2, st in zip(kp1s, kp2s, stamps):
        r = fit_frame(k1, k2, f_used, cfg.camera.stereo_base_m)
        r["stamp"] = str(st)
        r["n"] = int(len(k1))
        out.append(r)
        print(f"{st[11:]:>14} {len(k1):>8} {r['D']:>7.0f} {r['roll_deg']:>7.2f} "
              f"{r['tx']:>7.2f} {r['ty']:>7.2f} | {r['rms_full_px']:>9.2f} "
              f"{r['rms_no_roll_px']:>8.2f} {r['rms_translation_only_px']:>11.2f} "
              f"{r['rms_plain_plane_px']:>11.2f}")
    rolls = np.array([r["roll_deg"] for r in out])
    txs = np.array([r["tx"] for r in out]); tys = np.array([r["ty"] for r in out])
    print(f"\nroll across {len(out)} frames: median {np.median(rolls):+.2f} deg, "
          f"mean {rolls.mean():+.2f}, std {rolls.std():.2f}")
    print(f"tx   : median {np.median(txs):+.2f} px, std {txs.std():.2f}")
    print(f"ty   : median {np.median(tys):+.2f} px, std {tys.std():.2f}")
    for key,label in (("rms_full_px","full (D,roll,tx,ty)"),
                      ("rms_no_roll_px","D,tx,ty"),
                      ("rms_translation_only_px","D,tx,ty with roll fixed 0"),
                      ("rms_plain_plane_px","D only")):
        v=np.array([r[key] for r in out])
        print(f"  median rms {label:>28}: {np.median(v):.2f} px")
    print("\nIf removing the roll barely worsens the rms, the roll was absorbing a")
    print("translation; if it worsens it a lot, the roll is a real rotation error.")
    json.dump(out, open(os.path.join(HERE, "roll_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
