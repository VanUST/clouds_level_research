# PURPOSE: W4 gate 1, reframed. Ask whether the regime changes the MATCHED POPULATION at
#          equal certainty, rather than whether the extra matches are simply low
#          certainty.
#
#          The directive's original gate read "if the high regime has 2x the matches, are
#          they predominantly low-certainty? If so the matcher is likely hallucinating."
#          That inference has already been tested on this rig and rejected: RoMa's
#          low-certainty interior predictions agree with the reliable disparity field to
#          0.68 px against a 7.02 px spatially-shuffled null, and with LoFTR's independent
#          matches to 0.62 px (R-7). Low certainty here means low texture, not a wrong
#          answer. A gate that stopped at "they are low-certainty" would conclude the
#          wrong thing by construction.
#
#          So the gate is a MATCHED comparison: within a band of equal certainty, does the
#          high regime place its matches at a different disparity than the low regime?
#          Certainty cannot explain a difference that survives holding certainty fixed.
#
#          Three quantities are reported:
#            - where the extra matches live in certainty (are they actually low?)
#            - the certainty distribution by regime
#            - the disparity by regime WITHIN each certainty decile, which is the test
# INPUTS: experiments/w4/matches_roma.npz, the ceilometer log.
# OUTPUTS: docs/report/w4_certainty.png, experiments/w4/w4_certainty.json.
# KEYWORDS: W4, certainty, population, matched comparison, regime, G1
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
F = 654.0
B = 45.0
REGIME_LOW_MAX = 2750.0
REGIME_HIGH_MIN = 3500.0


def main():
    d = np.load(os.path.join(HERE, "matches_roma.npz"), allow_pickle=True)
    stamps = list(d["stamps"]); uv = list(d["uv"]); kp2 = list(d["kp2"])
    conf = list(d["conf"])
    series = [(x.time, x.h1) for x in C.parse_ceilometer_log(LOG) if x.h1 is not None]

    import re as _re
    _st = _re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")

    def ref_at(stamp):
        # stamps may carry the "img-" prefix from the filename; take the date and time
        m = _st.search(str(stamp))
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

    # pool matches, keeping certainty, disparity and regime
    certs, disps, regs, frames = [], [], [], []
    per_frame = []
    for i in range(len(stamps)):
        r = ref_at(stamps[i])
        a, b, c = uv[i], kp2[i], conf[i]
        if r is None or len(a) == 0:
            continue
        du = a[:, 0] - b[:, 0]
        keep = np.isfinite(du) & np.isfinite(c)
        certs.append(c[keep]); disps.append(du[keep])
        regs.append(np.full(int(keep.sum()), r))
        frames.append(np.full(int(keep.sum()), i))
        per_frame.append((float(r), int(keep.sum()),
                          float(np.median(c[keep])),
                          float(np.median(du[keep]))))
    cert = np.concatenate(certs); disp = np.concatenate(disps)
    reg = np.concatenate(regs); fid = np.concatenate(frames)
    lo = reg < REGIME_LOW_MAX
    hi = reg > REGIME_HIGH_MIN
    print(f"frames {len(per_frame)}  matches {len(cert):,}")
    print(f"  low regime  {int(lo.sum()):,} matches in {int(len(set(fid[lo])))} frames")
    print(f"  high regime {int(hi.sum()):,} matches in {int(len(set(fid[hi])))} frames")
    print(f"  match rate  low {lo.sum()/max(len(set(fid[lo])),1):.0f}/frame, "
          f"high {hi.sum()/max(len(set(fid[hi])),1):.0f}/frame "
          f"(ratio {hi.sum()/max(len(set(fid[hi])),1) / max(lo.sum()/max(len(set(fid[lo])),1), 1e-9):.2f})")

    print(f"\ncertainty distribution")
    for lab, m in (("low regime", lo), ("high regime", hi)):
        q = np.percentile(cert[m], [10, 25, 50, 75, 90])
        print(f"  {lab:>12}: p10 {q[0]:.4f}  p25 {q[1]:.4f}  median {q[2]:.4f}  "
              f"p75 {q[3]:.4f}  p90 {q[4]:.4f}")

    # where do the extra matches live? compare the certainty histogram, normalised
    edges = np.linspace(0, 1, 21)
    hl, _ = np.histogram(cert[lo], bins=edges)
    hh, _ = np.histogram(cert[hi], bins=edges)
    hl = hl / max(hl.sum(), 1); hh = hh / max(hh.sum(), 1)
    excess = hh - hl
    # the extra matches are the high regime's surplus over the low regime's shape
    extra_cert = float(np.average(0.5 * (edges[1:] + edges[:-1]), weights=np.maximum(excess, 0)))
    print(f"\n  the high regime's surplus population has median certainty ~{extra_cert:.3f} "
          f"(low regime overall median {np.median(cert[lo]):.3f})")

    # THE TEST: within each certainty decile, do the regimes differ in disparity?
    print(f"\nmatched-certainty test: median disparity by certainty decile")
    print(f"{'certainty decile':>18} {'n low':>9} {'n high':>9} {'du low':>9} "
          f"{'du high':>9} {'difference':>11}")
    qs = np.percentile(cert, np.arange(0, 101, 10))
    rows = []
    for k in range(10):
        a0, b0 = qs[k], qs[k + 1]
        m = (cert >= a0) & (cert <= b0) if k == 9 else (cert >= a0) & (cert < b0)
        ml = m & lo; mh = m & hi
        if ml.sum() < 500 or mh.sum() < 500:
            continue
        dl = float(np.median(disp[ml])); dh = float(np.median(disp[mh]))
        rows.append({"decile": k, "cert_lo": float(a0), "cert_hi": float(b0),
                     "n_low": int(ml.sum()), "n_high": int(mh.sum()),
                     "du_low": dl, "du_high": dh, "diff": dh - dl})
        print(f"{a0:7.3f}-{b0:5.3f} {ml.sum():>9,} {mh.sum():>9,} {dl:>9.2f} "
              f"{dh:>9.2f} {dh-dl:>+11.2f}")
    if rows:
        diffs = np.array([r["diff"] for r in rows])
        print(f"\n  differences across {len(rows)} deciles: "
              f"median {np.median(diffs):+.2f} px, all same sign: "
              f"{'yes' if np.all(diffs > 0) or np.all(diffs < 0) else 'no'}")
        print(f"  weighted mean difference {np.average(diffs, weights=[r['n_high'] for r in rows]):+.2f} px")

    baseline = float(np.median(disp[hi]) - np.median(disp[lo]))
    print(f"\n  POOLED difference (high - low) = {baseline:+.2f} px")
    print(f"  certainty alone cannot produce it if the within-decile differences persist")

    verdict = ("CERTAINTY DOES NOT EXPLAIN THE DIFFERENCE — the regimes differ in "
               "disparity at EVERY certainty level"
               if rows and (np.all(diffs > 0.2) or np.all(diffs < -0.2))
               else "CERTAINTY EXPLAINS IT — the difference vanishes at matched certainty"
               if rows else "insufficient data")
    print(f"\n  VERDICT: {verdict}")

    json.dump({"n_frames": len(per_frame), "n_matches": int(len(cert)),
               "rate_low_per_frame": float(lo.sum() / max(len(set(fid[lo])), 1)),
               "rate_high_per_frame": float(hi.sum() / max(len(set(fid[hi])), 1)),
               "median_cert_low": float(np.median(cert[lo])),
               "median_cert_high": float(np.median(cert[hi])),
               "extra_population_median_cert": extra_cert,
               "pooled_difference_px": baseline,
               "deciles": rows, "verdict": verdict},
              open(os.path.join(HERE, "w4_certainty.json"), "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ax1.step(ctr, hl, where="mid", color="#1f6feb", label="low regime")
    ax1.step(ctr, hh, where="mid", color="#c1440e", label="high regime")
    ax1.set_xlabel("RoMa certainty"); ax1.set_ylabel("share of matches")
    ax1.set_title("Certainty distribution by regime", fontsize=9); ax1.legend(fontsize=8)
    if rows:
        cs = [0.5 * (r["cert_lo"] + r["cert_hi"]) for r in rows]
        ax2.plot(cs, [r["du_low"] for r in rows], "o-", color="#1f6feb", label="low regime")
        ax2.plot(cs, [r["du_high"] for r in rows], "o-", color="#c1440e", label="high regime")
        ax2.set_xlabel("certainty decile centre"); ax2.set_ylabel("median disparity (px)")
        ax2.set_title("Disparity by regime at MATCHED certainty", fontsize=9)
        ax2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w4_certainty.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/w4_certainty.png and {HERE}/w4_certainty.json")


if __name__ == "__main__":
    main()
