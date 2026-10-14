# PURPOSE: W4 gate 4 — is the regime-dependent shift a property of the SCENE or specific
#          to RoMa's dense architecture and priors?
#
#          An independent matcher answers this only if it is genuinely independent. LoFTR
#          is used rather than SGBM or DIS on measured grounds: SGBM invents a full field
#          on a cloudless sky (6,400 matches/frame) and so cannot distinguish cloud from
#          nothing, whereas LoFTR returns 11 on clear sky against 2,189 on cloud and
#          therefore has a real rejection behaviour. LoFTR is also detector-free like
#          RoMa, so a difference between them isolates the feature representation and the
#          regression head, not the class of method.
#
#          The preparation is byte-identical to the RoMa dump - the same affine warp, the
#          same roll correction, the same centre crop - so the two matchers see the same
#          pixels and the comparison is not confounded by preprocessing.
#
#          The shift and its uncertainty are reported per frame and bootstrapped over
#          frames, because the frames are the independent unit; bootstrapping over
#          matches would treat ~12,000 correlated correspondences from one sky as 12,000
#          independent observations.
# INPUTS: the image archive, configs/paper_dlcp2026_roll.yaml, the ceilometer log.
# OUTPUTS: experiments/w4/gate4_result.json, docs/report/w4_gate4.png.
# KEYWORDS: gate4, cross_matcher, loftr, independence, bootstrap
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
F_PX = 654.0
B = 45.0
CACHE = os.path.join(HERE, "loftr_matches.npz")
ST = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def collect(limit=250, stride=2):
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"using cached LoFTR matches: {len(d['stamps'])} frames")
        return list(d["stamps"]), list(d["uv"]), list(d["kp2"])
    from clouds.config import load_config
    from clouds.geometry.imaging import center_crop, rotation_about
    from clouds.matching.matchers import create_matcher
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    roll = float(cfg.camera.roll_correction_deg or 0.0)
    crop = cfg.camera.target_size[0]
    orig = cfg.camera.original_size
    cxy = (orig[0] / 2.0, orig[1] / 2.0)
    names = sorted(f for f in os.listdir(IMAGES)
                   if f.startswith("img-2025-05-23") and f.endswith("devID1.jpg")
                   and "09-" <= f[15:18] <= "16-")[::stride][:limit]
    matcher = create_matcher("kornia_loftr", {}, "cuda")
    stamps, uv, kp2 = [], [], []
    for k, nm in enumerate(names):
        base = nm.replace("devID1.jpg", "")
        i1 = cv2.imread(os.path.join(IMAGES, nm))
        i2 = cv2.imread(os.path.join(IMAGES, base + "devID2.jpg"))
        if i1 is None or i2 is None:
            continue
        if roll:
            i2 = cv2.warpAffine(i2, rotation_about(cxy, roll), dsize=orig)
        w1 = cv2.warpAffine(i1, M, dsize=orig)
        g1 = cv2.cvtColor(center_crop(w1, crop, crop), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(center_crop(i2, crop, crop), cv2.COLOR_BGR2GRAY)
        a, b = matcher(g1, g2)
        a = np.asarray(a, float).reshape(-1, 2); b = np.asarray(b, float).reshape(-1, 2)
        stamps.append(base); uv.append(a.astype(np.float32)); kp2.append(b.astype(np.float32))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(names)}")
    np.savez_compressed(CACHE, stamps=np.array(stamps, dtype="U32"),
                        uv=np.array(uv, dtype=object), kp2=np.array(kp2, dtype=object))
    return stamps, uv, kp2


def main():
    stamps, uv, kp2 = collect()
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

    per = []
    for i in range(len(stamps)):
        r = ref_at(stamps[i])
        a, b = uv[i], kp2[i]
        if r is None or len(a) < 20:
            continue
        du = a[:, 0] - b[:, 0]
        dv = a[:, 1] - b[:, 1]
        good = np.isfinite(du) & np.isfinite(dv)
        if good.sum() < 20:
            continue
        per.append((r, float(np.median(du[good])), float(np.median(dv[good])),
                    int(good.sum())))
    r = np.array([p[0] for p in per]); du = np.array([p[1] for p in per])
    dv = np.array([p[2] for p in per]); n = np.array([p[3] for p in per])
    lo = r < 2750; hi = r > 3500
    print(f"\nLoFTR frames: {len(per)}   low regime {int(lo.sum())}, high {int(hi.sum())}")
    print(f"  matches per frame: median {np.median(n):.0f} "
          f"(low {np.median(n[lo]) if lo.sum() else float('nan'):.0f}, "
          f"high {np.median(n[hi]) if hi.sum() else float('nan'):.0f})")
    if lo.sum() < 5 or hi.sum() < 5:
        print("  INSUFFICIENT_DATA: fewer than 5 frames in a regime")
        return
    shift = float(np.median(du[hi]) - np.median(du[lo]))
    shift_dv = float(np.median(dv[hi]) - np.median(dv[lo]))
    # bootstrap over FRAMES, the independent unit
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(4000):
        i_lo = rng.choice(np.where(lo)[0], lo.sum(), replace=True)
        i_hi = rng.choice(np.where(hi)[0], hi.sum(), replace=True)
        boots.append(np.median(du[i_hi]) - np.median(du[i_lo]))
    boots = np.array(boots)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"\nLoFTR regime shift (high - low)")
    print(f"  along  du: {shift:+.3f} px   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  across dv: {shift_dv:+.3f} px")
    print(f"  median du: low {np.median(du[lo]):.3f}, high {np.median(du[hi]):.3f}")

    roma = {"shift_along_px": 1.798, "shift_across_px": 1.0185,
            "median_du_low": 12.273, "median_du_high": 14.070}
    same_sign = np.sign(shift) == np.sign(roma["shift_along_px"])
    ci_excludes_zero = not (ci[0] <= 0 <= ci[1])
    print(f"\nagainst RoMa: RoMa {roma['shift_along_px']:+.3f} px  "
          f"LoFTR {shift:+.3f} px")
    print(f"  same direction: {same_sign};  LoFTR CI excludes zero: {ci_excludes_zero}")
    if same_sign and ci_excludes_zero:
        verdict = ("MATCHER-INDEPENDENT — an independent matcher sees a shift in the same "
                   "direction with a CI excluding zero; the phenomenon is a property of "
                   "the imagery, not of RoMa")
    elif not same_sign:
        verdict = ("MATCHER-SPECIFIC — LoFTR's shift has the opposite sign, so the "
                   "phenomenon belongs to RoMa's representation, not the scene")
    else:
        verdict = "INCONCLUSIVE — LoFTR's shift is in the same direction but its CI includes zero"
    print(f"\n  VERDICT: {verdict}")

    json.dump({"n_frames": len(per), "n_low": int(lo.sum()), "n_high": int(hi.sum()),
               "matches_per_frame_median": float(np.median(n)),
               "loftr_shift_along_px": shift, "loftr_ci95": list(ci),
               "loftr_shift_across_px": shift_dv,
               "loftr_du_low": float(np.median(du[lo])),
               "loftr_du_high": float(np.median(du[hi])),
               "roma_shift_along_px": roma["shift_along_px"],
               "same_sign": bool(same_sign), "ci_excludes_zero": bool(ci_excludes_zero),
               "verdict": verdict},
              open(os.path.join(HERE, "gate4_result.json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    bins = np.linspace(min(du.min(), 0), np.percentile(du, 99), 40)
    ax1.hist(du[lo], bins=bins, alpha=0.6, color="#1f6feb", label="low regime", density=True)
    ax1.hist(du[hi], bins=bins, alpha=0.6, color="#c1440e", label="high regime", density=True)
    ax1.set_xlabel("LoFTR median disparity per frame (px)"); ax1.set_ylabel("density")
    ax1.set_title("LoFTR: per-frame median disparity by regime", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.hist(boots, bins=50, color="#5c6672")
    ax2.axvline(shift, color="#c1440e", lw=2, label=f"LoFTR {shift:+.2f} px")
    ax2.axvline(roma["shift_along_px"], color="#1a7f37", lw=2, ls="--",
                label=f"RoMa {roma['shift_along_px']:+.2f} px")
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("bootstrap shift (px)"); ax2.set_ylabel("count")
    ax2.set_title("LoFTR shift, bootstrapped over frames", fontsize=9)
    ax2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w4_gate4.png"), bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}/w4_gate4.png and {HERE}/gate4_result.json")


if __name__ == "__main__":
    main()
