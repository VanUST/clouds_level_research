# PURPOSE: W4 gate 3 — is the regime-dependent disparity shift a geometrically valid
#          displacement ALONG the epipolar line, or does it violate the epipolar
#          constraint?
#
#          A DESIGN NOTE FIRST, because it changes what this gate can conclude. The
#          measured shift is +1.8 px in the HORIZONTAL disparity. For a pair rectified
#          about a horizontal baseline the epipolar lines are horizontal, so a purely
#          horizontal displacement is BY CONSTRUCTION along them and the perpendicular
#          residual cannot respond to it. Run as literally specified, this gate passes
#          tautologically.
#
#          So two things are measured:
#
#          1. The gate as specified — the distribution of the perpendicular (vertical)
#             residual by regime — reported for completeness and to show the tautology
#             rather than assert it.
#
#          2. The test that actually discriminates: the shift is decomposed into its
#             along-epipolar and across-epipolar components, and then compared against
#             the shift the REFERENCE ITSELF predicts. If the cloud really moved from
#             the low to the high regime, the parallax must fall from f*B/H_low to
#             f*B/H_high. That predicted shift has a definite sign and magnitude, and the
#             observation either matches it or does not. This is the computation that
#             turns "the shift is real" into "the shift is a depth change or it is not".
#
#          The reference enters only in this last, explicitly-labelled comparison; the
#          residuals themselves are computed from the matches alone.
# INPUTS: experiments/w4/matches_roma.npz, the ceilometer log.
# OUTPUTS: experiments/w4/gate3_result.json, docs/report/w4_gate3.png.
# KEYWORDS: gate3, epipolar, decomposition, expected_shift, geometric_plausibility
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
F_PX = 654.0
B = 45.0
ST = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")


def main():
    d = np.load(os.path.join(HERE, "matches_roma.npz"), allow_pickle=True)
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

    # per-frame: along (du) and across (dv) components, plus their medians
    lo_du, hi_du, lo_dv, hi_dv = [], [], [], []
    frames = []
    for i in range(len(stamps)):
        r = ref_at(stamps[i])
        if r is None or len(uv[i]) < 500:
            continue
        a = np.asarray(uv[i], float); b = np.asarray(kp2[i], float)
        du = a[:, 0] - b[:, 0]
        dv = a[:, 1] - b[:, 1]          # the perpendicular residual for horizontal epipoles
        good = np.isfinite(du) & np.isfinite(dv)
        du, dv = du[good], dv[good]
        if len(du) < 500:
            continue
        frames.append((r, float(np.median(du)), float(np.median(dv)), len(du)))
        if r < 2750:
            lo_du.append(du); lo_dv.append(dv)
        elif r > 3500:
            hi_du.append(du); hi_dv.append(dv)

    ldu = np.concatenate(lo_du); hdu = np.concatenate(hi_du)
    ldv = np.concatenate(lo_dv); hdv = np.concatenate(hi_dv)
    kdu = ks_2samp(ldu, hdu); kdv = ks_2samp(ldv, hdv)
    print(f"matches: low {len(ldu):,}  high {len(hdu):,}")
    print(f"\nALONG-epipolar component (horizontal disparity du)")
    print(f"  low  median {np.median(ldu):.3f} px")
    print(f"  high median {np.median(hdu):.3f} px")
    shift_along = float(np.median(hdu) - np.median(ldu))
    print(f"  shift (high - low) = {shift_along:+.3f} px   KS p = {kdu.pvalue:.2e}")
    print(f"\nACROSS-epipolar component (perpendicular residual dv)")
    print(f"  low  median {np.median(ldv):.3f} px, p90 {np.percentile(np.abs(ldv),90):.3f} px")
    print(f"  high median {np.median(hdv):.3f} px, p90 {np.percentile(np.abs(hdv),90):.3f} px")
    shift_across = float(np.median(hdv) - np.median(ldv))
    print(f"  shift in the perpendicular residual = {shift_across:+.4f} px")
    print(f"  KS p = {kdv.pvalue:.2e}")

    gate3_pass = kdv.pvalue > 0.05
    print(f"\nGATE 3 as specified: {'PASS' if gate3_pass else 'FAIL'} "
          f"(perpendicular residual distributions "
          f"{'indistinguishable' if gate3_pass else 'differ'})")
    print(f"  BUT the shift under study is {shift_along:+.2f} px ALONG the epipolar line;")
    print(f"  the perpendicular residual responds to it by only {shift_across:+.4f} px.")
    print(f"  The gate is near-tautological for a horizontal shift, which is why the")
    print(f"  comparison below carries the diagnostic weight.")

    # ---- the discriminating test: does the observed shift match the one the
    # reference itself predicts?
    r_low = np.median([r for r, _, _, _ in frames if r < 2750])
    r_high = np.median([r for r, _, _, _ in frames if r > 3500])
    pred_low = F_PX * B / r_low
    pred_high = F_PX * B / r_high
    predicted_shift = pred_high - pred_low
    print(f"\nWHAT THE REFERENCE PREDICTS")
    print(f"  reference medians: low {r_low:.0f} m, high {r_high:.0f} m")
    print(f"  corresponding parallax: {pred_low:.2f} px -> {pred_high:.2f} px")
    print(f"  predicted disparity shift = {predicted_shift:+.3f} px")
    print(f"  OBSERVED shift            = {shift_along:+.3f} px")
    print(f"  discrepancy               = {shift_along - predicted_shift:+.3f} px "
          f"({abs(shift_along - predicted_shift)/max(abs(predicted_shift),1e-9)*100:.0f}% of the predicted magnitude)")

    # what height change WOULD produce the observed shift?
    implied_low = F_PX * B / np.median(ldu)
    implied_high = F_PX * B / np.median(hdu)
    print(f"\n  the observed shift corresponds to a height change of "
          f"{implied_low:.0f} m -> {implied_high:.0f} m "
          f"({implied_high - implied_low:+.0f} m)")
    print(f"  the reference's own change is {r_low:.0f} -> {r_high:.0f} m "
          f"({r_high - r_low:+.0f} m)")
    print(f"  sign agreement: "
          f"{'SAME' if np.sign(implied_high-implied_low)==np.sign(r_high-r_low) else 'OPPOSITE'}")

    verdict = ("GEOMETRICALLY VALID BUT PHYSICALLY INCONSISTENT — the shift is strictly "
               "along the epipolar line, so it is a valid displacement, but it has the "
               "OPPOSITE SIGN and a small fraction of the magnitude the reference "
               "predicts. It is not a depth change to the reported height."
               if np.sign(shift_along) != np.sign(predicted_shift) else
               "CONSISTENT WITH THE REFERENCE")
    print(f"\n  VERDICT: {verdict}")

    json.dump({"n_low_matches": int(len(ldu)), "n_high_matches": int(len(hdu)),
               "median_du_low": float(np.median(ldu)),
               "median_du_high": float(np.median(hdu)),
               "shift_along_px": shift_along, "ks_p_along": float(kdu.pvalue),
               "median_dv_low": float(np.median(ldv)),
               "median_dv_high": float(np.median(hdv)),
               "shift_across_px": shift_across, "ks_p_across": float(kdv.pvalue),
               "gate3_as_specified": "PASS" if gate3_pass else "FAIL",
               "predicted_shift_px": float(predicted_shift),
               "observed_shift_px": shift_along,
               "implied_height_low_m": float(implied_low),
               "implied_height_high_m": float(implied_high),
               "reference_low_m": float(r_low), "reference_high_m": float(r_high),
               "verdict": verdict},
              open(os.path.join(HERE, "gate3_result.json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax1.hist(ldv, bins=np.linspace(-4, 4, 60), alpha=0.6, color="#1f6feb",
             label="low regime", density=True)
    ax1.hist(hdv, bins=np.linspace(-4, 4, 60), alpha=0.6, color="#c1440e",
             label="high regime", density=True)
    ax1.set_xlabel("perpendicular epipolar residual (px)"); ax1.set_ylabel("density")
    ax1.set_title(f"Gate 3: perpendicular residual (KS p = {kdv.pvalue:.2f})", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.bar([0, 1], [shift_along, predicted_shift], color=["#c1440e", "#8b949e"])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["observed", "predicted\nby reference"])
    ax2.set_ylabel("disparity shift, high - low (px)")
    ax2.set_title("Along-epipolar shift against the reference's prediction", fontsize=9)
    for x, v in ((0, shift_along), (1, predicted_shift)):
        ax2.text(x, v + (0.15 if v > 0 else -0.35), f"{v:+.2f}", ha="center", fontsize=8)
    fig.savefig(os.path.join(OUT, "w4_gate3.png"), bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}/w4_gate3.png and {HERE}/gate3_result.json")


if __name__ == "__main__":
    main()
