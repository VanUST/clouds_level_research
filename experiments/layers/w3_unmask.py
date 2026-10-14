# PURPOSE: Test whether removing the INSTRUMENTAL component of the disparity field
#          unmasks a correctly-signed response to the cloud regime.
#
#          W3 established that the static spatial pattern decomposes into an
#          instrumental part (66 % of the structured spatial variance, identical under
#          two different cloud regimes) and a regime-dependent part (34 %). It also
#          found that the regime-dependent part has the INVERTED sign: the high regime
#          carries a HIGHER disparity, i.e. a LOWER estimated height.
#
#          That raises a sharp, falsifiable question. If the instrumental component is
#          additive and independent, subtracting it leaves the regime-dependent field.
#          If that field is real cloud structure seen through bad geometry, the
#          corrected series should respond to the reference with the CORRECT sign. If
#          the corrected series still responds with the inverted sign or not at all,
#          the inversion is in the matched population rather than in the calibration,
#          and no calibration correction will fix it.
#
#          The instrument correction is a per-cell subtraction of the session-mean
#          disparity, which is exactly the component that is reproducible across
#          regimes. It is NOT fitted to the ceilometer: the reference is consulted only
#          after the corrected series exists, so a positive result is not circular.
#          This distinction is the whole value of the test and is why the mean is taken
#          over the session rather than per regime.
# INPUTS: experiments/layers/heights.npz, the ceilometer log.
# OUTPUTS: experiments/layers/w3_unmask.json and a printed verdict.
# KEYWORDS: instrumental, removal, unmask, sign, falsifiable, W3, D2
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
F = 654.0
B = 45.0
GRID = 32
CELL = 20
CROP = 640


def ref_series():
    return [(x.time, x.h1) for x in C.parse_ceilometer_log(LOG) if x.h1 is not None]


def ref_at(series, stamp):
    t = dt.datetime.fromisoformat(
        f"{str(stamp)[:10]}T{str(stamp)[11:].replace('-', ':')}") + dt.timedelta(hours=3)
    best, bd = None, None
    for tt, hh in series:
        dd = abs((tt - t).total_seconds())
        if bd is None or dd < bd:
            best, bd = hh, dd
    return best if (bd is not None and bd <= 120) else None


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    stamps = list(d["stamps"])
    series = ref_series()

    def cells(i):
        a = K1[i]
        return (np.clip((a[:, 0] / CELL).astype(int), 0, GRID - 1),
                np.clip((a[:, 1] / CELL).astype(int), 0, GRID - 1))

    # the instrumental map: the per-cell median over the WHOLE session
    acc = [[[] for _ in range(GRID)] for _ in range(GRID)]
    for i, (a, b) in enumerate(zip(K1, K2)):
        if len(a) == 0:
            continue
        du = a[:, 0] - b[:, 0]
        ui, vi = cells(i)
        for x, y, v in zip(ui, vi, du):
            if np.isfinite(v):
                acc[y][x].append(v)
    amap = np.full((GRID, GRID), np.nan)
    for y in range(GRID):
        for x in range(GRID):
            if len(acc[y][x]) >= 50:
                amap[y, x] = float(np.median(acc[y][x]))
    ok = np.isfinite(amap)
    print(f"instrumental map: {int(ok.sum())} cells, "
          f"mean {np.nanmean(amap):.3f} px, spatial std {np.nanstd(amap):.3f} px")

    def series_from(correct):
        ts, hs = [], []
        for i, (a, b) in enumerate(zip(K1, K2)):
            r = ref_at(series, stamps[i])
            if r is None or len(a) < 30:
                continue
            du = a[:, 0] - b[:, 0]
            if correct:
                ui, vi = cells(i)
                good = np.isfinite(amap[vi, ui])
                du = du.copy()
                # remove the cell's own instrumental offset, keeping the session mean
                du[good] = du[good] - (amap[vi[good], ui[good]] - np.nanmean(amap))
                du = du[np.isfinite(du)]
            if len(du) < 30:
                continue
            # a positive disparity is required; take the median as the frame's estimate
            du = du[du > 0.5]
            if len(du) < 30:
                continue
            ts.append(r)
            hs.append(F * B / float(np.median(du)))
        return np.array(hs), np.array(ts)

    raw, ref = series_from(False)
    cor, ref2 = series_from(True)
    print(f"frames: {len(raw)} raw, {len(cor)} corrected\n")

    def report(name, est, r):
        # level: rank correlation, which does not assume linearity
        from scipy.stats import spearmanr, mannwhitneyu
        rho, p = spearmanr(est, r)
        lo = est[r < 2750]; hi = est[r > 3500]
        shift = (np.median(hi) - np.median(lo)) if len(lo) > 5 and len(hi) > 5 else np.nan
        try:
            u, pu = mannwhitneyu(lo, hi, alternative="two-sided") if len(lo) > 5 and len(hi) > 5 else (np.nan, np.nan)
        except ValueError:
            pu = np.nan
        print(f"{name}")
        print(f"  frames {len(est)}   median height {np.median(est):7.0f} m   "
              f"std {np.std(est):6.0f} m")
        print(f"  rank correlation with the reference  rho = {rho:+.3f}  (p = {p:.2e})")
        print(f"  low regime n={len(lo)} median {np.median(lo) if len(lo) else float('nan'):.0f} m")
        print(f"  high regime n={len(hi)} median {np.median(hi) if len(hi) else float('nan'):.0f} m")
        print(f"  regime shift (high - low)            {shift:+.0f} m  (p = {pu:.2e})")
        sign = "CORRECT" if shift > 0 else "INVERTED"
        print(f"  -> sign of the response: {sign}\n")
        return {"n": int(len(est)), "median_h": float(np.median(est)),
                "std_h": float(np.std(est)), "rho": float(rho), "p": float(p),
                "regime_shift_m": float(shift), "sign": sign}

    out = {"raw": report("AS SHIPPED (no instrumental removal)", raw, ref),
           "instrumental_removed": report("INSTRUMENTAL COMPONENT REMOVED", cor, ref2)}

    r0 = out["raw"]; r1 = out["instrumental_removed"]
    print("Interpretation")
    if r1["regime_shift_m"] > 0 and r0["regime_shift_m"] < 0:
        verdict = ("REMOVING THE INSTRUMENTAL COMPONENT FLIPS THE SIGN — the inversion "
                   "was the calibration's, and a geometry correction can fix it")
    elif r1["regime_shift_m"] < 0:
        verdict = ("THE SIGN STAYS INVERTED — the inversion is in the matched "
                   "population, not the calibration; no geometric correction fixes it")
    else:
        verdict = "INCONCLUSIVE"
    print(f"  raw shift {r0['regime_shift_m']:+.0f} m -> corrected "
          f"{r1['regime_shift_m']:+.0f} m")
    print(f"\n  VERDICT: {verdict}")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(HERE, "w3_unmask.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'w3_unmask.json')}")


if __name__ == "__main__":
    main()
