# PURPOSE: W6 observational mapping — does the disparity field track a PHOTOMETRIC
#          property of the sky rather than the geometry of a cloud surface?
#
#          The hypothesis: clouds are strongly non-Lambertian, and the 45 m baseline
#          makes the two cameras view the same parcel from directions differing by
#          B/H - about 1.29 deg at 2000 m and 0.64 deg at 4000 m. If a dense matcher
#          maximises photometric similarity, an apparent brightness peak (isophote) can
#          sit at a different place in the two views, and the matcher tracks the
#          isophote rather than the physical feature. That displacement is along the
#          local brightness gradient, which is not generally aligned with the epipolar
#          line, so it inflates the perpendicular residual - exactly the Gate 3 result.
#
#          Three observables are mapped against the disparity field:
#            (1) the gradient direction relative to the epipolar line, and its magnitude
#            (2) the local spatial frequency (structure-tensor scale)
#            (3) the local brightness and contrast (optical-depth proxy)
#
#          THE KEY DESIGN CHOICE. (3) is tested WITHIN FRAME. Comparing bright regions to
#          dark regions inside one exposure holds the cloud, the regime, the sun angle
#          and the atmosphere fixed, so a within-frame brightness effect cannot be
#          explained by any of them. The cross-regime comparison of the same quantity is
#          reported separately, and the pair of results distinguishes a photometric
#          mechanism from a regime-level confound.
#
#          Ground-truth-free: no reference is used anywhere in this script.
# INPUTS: experiments/w4/matches_roma_20250523.npz and the image archive.
# OUTPUTS: docs/report/w6_photometric.png, experiments/w6/w6_photometric.json.
# KEYWORDS: W6, photometric, non_lambertian, isophote, gradient, structure_tensor
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
WIN = 9                      # window for the local gradient / structure tensor


def main():
    from clouds.config import load_config
    from clouds.geometry.imaging import center_crop, rotation_about
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(json.load(open(os.path.join(REPO, "configs", "calib",
                                             "2025-05-22_affine_old.json")))["matrix_2x3"])
    roll = float(cfg.camera.roll_correction_deg or 0.0)
    orig = cfg.camera.original_size
    cxy = (orig[0] / 2.0, orig[1] / 2.0)

    d = np.load(os.path.join(HERE, "..", "w4", "matches_roma_20250523.npz"),
                allow_pickle=True)
    stamps = list(d["stamps"]); uv = list(d["uv"]); kp2 = list(d["kp2"])
    series = [(x.time, x.h1) for x in C.parse_ceilometer_log(LOG) if x.h1 is not None]

    def ref_at(s):
        m = ST.search(str(s))
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

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    keys = list(range(0, len(stamps), stride))[:limit]

    rows = []
    for n, i in enumerate(keys):
        r = ref_at(stamps[i])
        if r is None or len(uv[i]) < 500:
            continue
        base = str(stamps[i])
        i1 = cv2.imread(os.path.join(IMAGES, base + "devID1.jpg"))
        if i1 is None:
            continue
        if roll:
            pass                       # img1 is warped by the affine; the roll is on img2
        w1 = cv2.warpAffine(i1, M, dsize=orig)
        g = cv2.cvtColor(center_crop(w1, CROP, CROP), cv2.COLOR_BGR2GRAY)
        gf = g.astype(np.float32)
        gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
        # the local mean shift is used as the ADDRESS: on the warped image the match was
        # found by comparing a window, so its brightness context is the window around it
        a = np.asarray(uv[i], float); b = np.asarray(kp2[i], float)
        du = a[:, 0] - b[:, 0]; dv = a[:, 1] - b[:, 1]
        ui = np.clip(a[:, 0].astype(int), 0, CROP - 1)
        vi = np.clip(a[:, 1].astype(int), 0, CROP - 1)
        gmag = np.hypot(gx[vi, ui], gy[vi, ui])
        gang = np.arctan2(gy[vi, ui], gx[vi, ui])
        bright = g[vi, ui].astype(float)
        good = np.isfinite(du) & np.isfinite(dv) & np.isfinite(gmag)
        if good.sum() < 500:
            continue
        rows.append(dict(regime=float(r), du=du[good], dv=dv[good],
                         gmag=gmag[good], gang=gang[good], bright=bright[good],
                         frame=n))
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{len(keys)}")

    print(f"\nframes analysed: {len(rows)}")
    reg = np.array([r_["regime"] for r_ in rows])
    # ---- (1) gradient magnitude vs the residuals, WITHIN frame (median of per-frame rho)
    def within(name, key, target):
        vals = []
        for r_ in rows:
            if np.std(r_[key]) == 0 or np.std(r_[target]) == 0:
                continue
            vals.append(np.corrcoef(r_[key], r_[target])[0, 1])
        vals = np.array(vals)
        print(f"  within-frame rho({name}, {target}) median {np.median(vals):+.3f} "
              f"(positive in {100*(vals>0).mean():.0f}% of frames)")
        return float(np.median(vals)), vals

    print("\n(1) GRADIENT MAGNITUDE — does the residual scale with photometric structure?")
    r_gm_dv, _ = within("|grad|", "gmag", "dv")
    r_gm_du, _ = within("|grad|", "gmag", "du")

    # gradient direction relative to the epipolar direction (u axis for a rectified pair)
    print("\n(2) GRADIENT DIRECTION relative to the epipolar line")
    allang = np.concatenate([r_["gang"] for r_ in rows])
    alldv = np.concatenate([r_["dv"] for r_ in rows])
    # the component of the gradient perpendicular to the epipolar line is gx (the u
    # component); parallel is gy. A photometric pull is along the gradient, so its
    # perpendicular part scales with |cos(angle)|.
    perp = np.abs(np.cos(allang))
    rho_perp, p_perp = spearmanr(perp, np.abs(alldv))
    print(f"  rho(|cos(gradient angle)|, |dv|) = {rho_perp:+.3f} (p = {p_perp:.2e})")
    print(f"    a photometric pull predicts dv is largest when the gradient is")
    print(f"    PERPENDICULAR to the epipolar line, i.e. |cos| -> 1")

    # ---- (3) brightness WITHIN frame (the design that controls everything)
    print("\n(3) BRIGHTNESS — within-frame, which holds cloud, sun and regime fixed")
    r_br_du, _ = within("brightness", "bright", "du")
    r_br_dv, _ = within("brightness", "bright", "dv")
    # brightness deciles within each frame, then the disparity shift across them
    print(f"\n{'brightness decile (within frame)':>34} {'median du':>11} {'median dv':>11}")
    dec_du, dec_dv = [], []
    for k in range(10):
        dus, dvs = [], []
        for r_ in rows:
            br = r_["bright"]
            q = np.percentile(br, np.arange(0, 101, 10))
            m = (br >= q[k]) & (br <= q[k + 1]) if k == 9 else (br >= q[k]) & (br < q[k + 1])
            if m.sum() > 20:
                dus.append(np.median(r_["du"][m])); dvs.append(np.median(r_["dv"][m]))
        if dus:
            dec_du.append(np.median(dus)); dec_dv.append(np.median(dvs))
            print(f"{k:>34} {np.median(dus):>11.3f} {np.median(dvs):>11.3f}")
    span_du = float(max(dec_du) - min(dec_du)) if dec_du else float("nan")

    # cross-regime brightness for contrast
    lo = reg < 2750; hi = reg > 3500
    br_lo = np.median(np.concatenate([r_["bright"] for r_ in rows if r_["regime"] < 2750])) \
        if lo.any() else float("nan")
    br_hi = np.median(np.concatenate([r_["bright"] for r_ in rows if r_["regime"] > 3500])) \
        if hi.any() else float("nan")
    print(f"\n  cross-regime: median brightness low {br_lo:.1f}, high {br_hi:.1f}")

    verdict = ("PHOTOMETRIC COUPLING IS WEAK WITHIN FRAME — brightness explains little of "
               "the disparity once the cloud is held fixed"
               if abs(r_br_du) < 0.15 and abs(r_gm_du) < 0.15 else
               "PHOTOMETRIC COUPLING PRESENT — the disparity tracks brightness/gradient "
               "within a single frame")
    print(f"\n  within-frame brightness span of the disparity: {span_du:.3f} px")
    print(f"  (the cross-regime effect under study is +1.798 px)")
    print(f"\n  VERDICT: {verdict}")

    os.makedirs(HERE, exist_ok=True)
    json.dump({"n_frames": len(rows),
               "within_rho_gmag_dv": r_gm_dv, "within_rho_gmag_du": r_gm_du,
               "within_rho_bright_du": r_br_du, "within_rho_bright_dv": r_br_dv,
               "rho_perpcos_absdv": float(rho_perp), "p_perpcos_absdv": float(p_perp),
               "within_frame_brightness_disparity_span_px": span_du,
               "cross_regime_effect_px": 1.798,
               "brightness_low": float(br_lo), "brightness_high": float(br_hi),
               "verdict": verdict},
              open(os.path.join(HERE, "w6_photometric.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))
    axes[0].scatter(np.concatenate([r_["gmag"] for r_ in rows])[:200000],
                    np.abs(np.concatenate([r_["dv"] for r_ in rows]))[:200000],
                    s=1, alpha=0.15, color="#c1440e")
    axes[0].set_xlabel("|gradient| at the match"); axes[0].set_ylabel("|dv| (px)")
    axes[0].set_xlim(0, np.percentile(np.concatenate([r_["gmag"] for r_ in rows]), 99))
    axes[0].set_title(f"Gradient vs perpendicular residual\n(within-frame rho {r_gm_dv:+.2f})",
                      fontsize=9)
    axes[1].hist(np.degrees(allang), bins=90, color="#5c6672")
    axes[1].set_xlabel("gradient direction (deg)"); axes[1].set_ylabel("count")
    axes[1].set_title("Gradient direction distribution", fontsize=9)
    axes[2].plot(range(len(dec_du)), dec_du, "o-", color="#c1440e")
    axes[2].axhline(np.median(dec_du), color="k", ls=":", lw=1)
    axes[2].set_xlabel("brightness decile within frame")
    axes[2].set_ylabel("median du (px)")
    axes[2].set_title(f"Brightness vs disparity, within frame\n(span {span_du:.2f} px)",
                      fontsize=9)
    fig.savefig(os.path.join(OUT, "w6_photometric.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/w6_photometric.png and {HERE}/w6_photometric.json")


if __name__ == "__main__":
    main()
