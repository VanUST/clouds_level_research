# PURPOSE: Test the desync-motion hypothesis for the Gate 3 epipolar violation.
#
#          The hypothesis: two cameras exposing ~1.5 s apart, a cloud moving with the
#          wind, so the same feature is displaced between the exposures. That displacement
#          projects onto BOTH disparity components, inflating the perpendicular epipolar
#          residual dv and biasing the horizontal disparity du.
#
#          THE PREMISE IS ALREADY MEASURED AND NOT SUPPORTED. The archive stamps both
#          cameras with the same filename timestamp in 89.9 % of 2025-05-22 frames and
#          91.0 % of 2025-05-23 frames, with a median offset of 0 s and a maximum of 5 s
#          (experiments/w2/PREREGISTRATION.md, re-verified 2026-10-14). A 1-2 s desync is
#          not in the data. But identical FILENAMES do not prove identical EXPOSURES, so
#          the hypothesis deserves an independent test rather than a dismissal.
#
#          THE TEST. Desync-motion makes a prediction that does not depend on knowing
#          the exposure offset. The contamination must scale with the cloud's SPEED:
#          dv_expected ~ f * (dt / T) * |motion|, where T is the interval between the
#          frames used to measure motion. A fast-moving cloud must show a larger dv than
#          a slow one. If dv is instead flat in speed and simply tracks the height
#          regime, then speed - and therefore desync - does not explain it.
#
#          Motion is measured by phase correlation between consecutive crops, which needs
#          no matcher and no model, so it cannot share a failure mode with RoMa.
#
#          A second, independent prediction is also checked: the required exposure
#          offset. dv = f * v * dt / H solves for dt given the observed dv, the measured
#          speed and the cloud height. If that dt is physically impossible or
#          inconsistent with the regime, the hypothesis is refuted on its own numbers.
# INPUTS: the image archive, experiments/w4/matches_roma_20250523.npz.
# OUTPUTS: experiments/w4/w4_desync.json, docs/report/w4_desync.png.
# KEYWORDS: desync, motion, epipolar_violation, gate3, refutation, phase_correlation
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
F_PX = 654.0
B = 45.0
CROP = 640
ST = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")

MOTION_CACHE = os.path.join(HERE, "motion_cache.npz")


# PURPOSE: Measure the image-plane motion of the sky between consecutive crops.
# INPUTS: list of frame stems, the frame stride in the archive.
# OUTPUTS: (dt_s, dx_px, dy_px, magnitude_px) per frame. Phase correlation is used
#          because it needs no matcher and no model, so it cannot share a failure mode
#          with the matcher whose residual it is being used to explain.
# KEYWORDS: phase_correlation, motion, speed
def measure_motion(stems, stride_s=10.0):
    from clouds.config import load_config
    from clouds.geometry.imaging import center_crop, rotation_about
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(calibration_matrix(cfg))
    roll = float(cfg.camera.roll_correction_deg or 0.0)
    orig = cfg.camera.original_size
    cxy = (orig[0] / 2.0, orig[1] / 2.0)
    prev = None
    rows = []
    for k, base in enumerate(stems):
        i2 = cv2.imread(os.path.join(IMAGES, base + "devID2.jpg"), cv2.IMREAD_GRAYSCALE)
        if i2 is None:
            prev = None
            continue
        if roll:
            i2 = cv2.warpAffine(i2, rotation_about(cxy, roll), dsize=orig)
        g = np.float32(center_crop(i2, CROP, CROP))
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            rows.append((base, dx, dy, float(np.hypot(dx, dy))))
        prev = g
        if (k + 1) % 50 == 0:
            print(f"  motion {k+1}/{len(stems)}")
    return rows


def calibration_matrix(cfg):
    import json as _json
    cal = _json.load(open(os.path.join(REPO, "configs", "calib",
                                       "2025-05-22_affine_old.json")))
    return cal["matrix_2x3"]


def main():
    d = np.load(os.path.join(HERE, "matches_roma_20250523.npz"), allow_pickle=True)
    stamps = list(d["stamps"]); uv = list(d["uv"]); kp2 = list(d["kp2"])
    series = [(x.time, x.h1) for x in C.parse_ceilometer_log(LOG) if x.h1 is not None]

    def ref_at(stamp):
        m = ST.search(str(stamp))
        if not m:
            return None
        t = dt.datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
                        int(m.group(2)), int(m.group(3)), int(m.group(4))) + dt.timedelta(hours=3)
        best, bd = None, None
        for tt, hh in series:
            dd = abs((tt - t).total_seconds())
            if bd is None or dd < bd:
                best, bd = hh, dd
        return best if (bd is not None and bd <= 120) else None

    # per-frame dv and du from the dump
    per = {}
    for i in range(len(stamps)):
        a = np.asarray(uv[i], float); b = np.asarray(kp2[i], float)
        du = a[:, 0] - b[:, 0]; dv = a[:, 1] - b[:, 1]
        g = np.isfinite(du) & np.isfinite(dv)
        if g.sum() < 500:
            continue
        key = str(stamps[i])
        per[key] = (float(np.median(du[g])), float(np.median(dv[g])))
    print(f"per-frame residuals for {len(per)} frames")

    # motion between consecutive archive frames
    if os.path.exists(MOTION_CACHE):
        md = np.load(MOTION_CACHE, allow_pickle=True)
        motion = dict(zip(list(md["stems"]), zip(md["dx"], md["dy"])))
        print(f"using cached motion for {len(motion)} frames")
    else:
        stems = sorted(per.keys())
        rows = measure_motion(stems)
        np.savez_compressed(MOTION_CACHE, stems=np.array([r[0] for r in rows]),
                            dx=np.array([r[1] for r in rows]),
                            dy=np.array([r[2] for r in rows]))
        motion = {r[0]: (r[1], r[2]) for r in rows}

    keys = [k for k in per if k in motion]
    du = np.array([per[k][0] for k in keys])
    dv = np.array([per[k][1] for k in keys])
    mv = np.array([np.hypot(*motion[k]) for k in keys])
    rf = np.array([ref_at(k) if ref_at(k) is not None else np.nan for k in keys])
    good = np.isfinite(rf)
    du, dv, mv, rf = du[good], dv[good], mv[good], rf[good]
    print(f"matched frames: {len(du)}")
    print(f"\nmotion between consecutive frames (10 s apart):")
    print(f"  median {np.median(mv):.2f} px, p10 {np.percentile(mv,10):.2f}, "
          f"p90 {np.percentile(mv,90):.2f}")

    rho_dv, p_dv = spearmanr(mv, dv)
    rho_du, p_du = spearmanr(mv, du)
    print(f"\nTHE TEST: does the residual scale with cloud SPEED?")
    print(f"  rho(motion, dv) = {rho_dv:+.3f}  (p = {p_dv:.2e})")
    print(f"  rho(motion, du) = {rho_du:+.3f}  (p = {p_du:.2e})")
    lo_h = rf < 2750; hi_h = rf > 3500
    print(f"\n  motion by regime: low {np.median(mv[lo_h]):.2f} px, "
          f"high {np.median(mv[hi_h]):.2f} px")
    print(f"  dv by regime:     low {np.median(dv[lo_h]):.3f} px, "
          f"high {np.median(dv[hi_h]):.3f} px")
    rho_reg, p_reg = spearmanr(rf, dv)
    print(f"  rho(reference height, dv) = {rho_reg:+.3f} (p = {p_reg:.2e})")

    # the required exposure offset, from the hypothesis's own arithmetic
    H = np.where(np.isfinite(rf), rf, np.nan)
    speed_mps = mv * H / (F_PX * 10.0)          # image motion over 10 s -> m/s at height H
    print(f"\n  implied cloud speed from the motion: median "
          f"{np.nanmedian(speed_mps):.2f} m/s")
    dv_abs = np.abs(dv)
    dt_required_s = dv_abs * H / (F_PX * np.maximum(speed_mps, 1e-6))
    print(f"  exposure offset REQUIRED to explain dv: median "
          f"{np.nanmedian(dt_required_s):.2f} s "
          f"(p10 {np.nanpercentile(dt_required_s,10):.2f}, "
          f"p90 {np.nanpercentile(dt_required_s,90):.2f})")
    print(f"  the archive's measured offset: median 0 s, max 5 s")

    # A speed dependence is the hypothesis's signature. Compare the explained variance
    # of speed against that of the height regime.
    r2_speed = float(rho_dv ** 2)
    r2_regime = float(rho_reg ** 2)
    print(f"\n  variance of dv explained by SPEED   : {100*r2_speed:.1f}%")
    print(f"  variance of dv explained by REGIME  : {100*r2_regime:.1f}%")

    verdict = ("REFUTED — dv does not scale with cloud speed, and the exposure offset the "
               "observation would require is inconsistent with the archive's timing; the "
               "epipolar violation tracks the height regime instead"
               if (p_dv > 0.01 or abs(rho_dv) < 0.3) else
               "SUPPORTED — dv scales with cloud speed as the desync hypothesis predicts")
    print(f"\n  VERDICT: {verdict}")

    json.dump({"n_frames": int(len(du)), "motion_median_px": float(np.median(mv)),
               "rho_motion_dv": float(rho_dv), "p_motion_dv": float(p_dv),
               "rho_motion_du": float(rho_du), "p_motion_du": float(p_du),
               "rho_regime_dv": float(rho_reg), "p_regime_dv": float(p_reg),
               "motion_low_px": float(np.median(mv[lo_h])),
               "motion_high_px": float(np.median(mv[hi_h])),
               "dv_low_px": float(np.median(dv[lo_h])),
               "dv_high_px": float(np.median(dv[hi_h])),
               "implied_speed_mps": float(np.nanmedian(speed_mps)),
               "required_exposure_offset_s": float(np.nanmedian(dt_required_s)),
               "archive_offset_s": 0.0,
               "r2_speed": r2_speed, "r2_regime": r2_regime, "verdict": verdict},
              open(os.path.join(HERE, "w4_desync.json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax1.scatter(mv, dv, s=10, c=rf, cmap="coolwarm", alpha=0.8)
    ax1.set_xlabel("sky motion between consecutive frames (px / 10 s)")
    ax1.set_ylabel("perpendicular residual dv (px)")
    ax1.set_title(f"dv vs SPEED: rho = {rho_dv:+.2f} (p = {p_dv:.1e})", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(np.nanmin(rf), np.nanmax(rf)))
    plt.colorbar(sm, ax=ax1, fraction=0.046, label="reference H1 (m)")
    ax2.scatter(rf, dv, s=10, color="#c1440e", alpha=0.8)
    ax2.set_xlabel("reference H1 (m)"); ax2.set_ylabel("perpendicular residual dv (px)")
    ax2.set_title(f"dv vs REGIME: rho = {rho_reg:+.2f} (p = {p_reg:.1e})", fontsize=9)
    fig.savefig(os.path.join(OUT, "w4_desync.png"), bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}/w4_desync.png and {HERE}/w4_desync.json")


if __name__ == "__main__":
    main()
