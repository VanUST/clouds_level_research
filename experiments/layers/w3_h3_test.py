# PURPOSE: Separate an INSTRUMENTAL static pattern (H1/H2) from a CLOUD-CLIMATOLOGY
#          confound (H3) without the foreground control, which W2 showed is unavailable.
#
#          The logic. The static pattern is a fact: the height map correlates r = 0.98
#          between the two halves of the session. The question is what produces it.
#
#            H3 (climatic) says clouds prefer certain image regions, so the map is a
#            climatology of where cloud happens to be. If that is true, the map MUST
#            change when the cloud regime changes: a day whose cloud base swings through
#            1450 m cannot produce the same per-cell heights in its low and high states.
#
#            H1/H2 (instrumental) says the map is the calibration's, so it is the same
#            whatever the sky does.
#
#          So: partition the session by the CEILOMETER's own reported regime, build the
#          height map separately in each partition, and compare. The ceilometer is used
#          only to SORT frames into two groups; it never enters the estimate, so
#          guardrail G1 is respected - the estimator's output is compared with itself
#          across two disjoint sets of frames.
#
#          Both the disparity field and the height field are reported, because the
#          height inverts the disparity nonlinearly and the two can disagree in
#          magnitude even when they agree in structure.
# INPUTS: experiments/layers/heights.npz, the ceilometer log, the session clock offset.
# OUTPUTS: docs/report/w3_regime_maps.png, experiments/layers/w3_h3_test.json.
# KEYWORDS: H3, climatology, confound, instrumental, regime, static_pattern, falsifiable
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
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402

OUT = os.path.join(REPO, "docs", "report")
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
CLOCK_OFFSET_H = 3.0        # image filename clocks are UTC; the log is MSK
CROP = 640
GRID = 32
CELL = CROP // GRID

REGIME_LOW_MAX = 2750.0     # the ceilometer's own lower population
REGIME_HIGH_MIN = 3500.0    # and its upper one


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    stamps = list(d["stamps"]) if "stamps" in d else None
    if stamps is None:
        print("heights.npz has no stamps; cannot partition by regime")
        return

    # the reference's own state at each frame, on the ceilometer clock
    series = C.parse_ceilometer_log(LOG)
    ref = [(x.time, x.h1) for x in series if x.h1 is not None]

    def regime_at(stamp):
        # `stamp` is the image (UTC) clock; the log is MSK, so add the offset
        try:
            # the archive stamps time with dashes: "2025-05-23T09-00-03"
            raw = str(stamp).replace("T", " ").replace("-", " ", 2)
            date_part, time_part = str(stamp)[:10], str(stamp)[11:].replace("-", ":")
            t = dt.datetime.fromisoformat(f"{date_part}T{time_part}")
        except ValueError:
            return None, None
        t = t + dt.timedelta(hours=CLOCK_OFFSET_H)
        best, bd = None, None
        for tt, hh in ref:
            dd = abs((tt - t).total_seconds())
            if bd is None or dd < bd:
                best, bd = hh, dd
        if bd is None or bd > 120:
            return None, None
        return best, t

    groups = {"low": [], "high": []}
    labels = []
    for fi, (a, b) in enumerate(zip(K1, K2)):
        st = str(stamps[fi])
        h, _t = regime_at(st)
        labels.append(h)
        if h is None or len(a) == 0:
            continue
        if h < REGIME_LOW_MAX:
            groups["low"].append(fi)
        elif h > REGIME_HIGH_MIN:
            groups["high"].append(fi)
    print(f"frames: {len(K1)}   low regime {len(groups['low'])}, "
          f"high regime {len(groups['high'])}")

    def build(idx, field):
        """Median per image cell over the selected frames."""
        acc = [[[] for _ in range(GRID)] for _ in range(GRID)]
        for fi in idx:
            a, b = K1[fi], K2[fi]
            if len(a) == 0:
                continue
            if field == "du":
                v = a[:, 0] - b[:, 0]
            else:
                raise ValueError(field)
            ui = np.clip((a[:, 0] / CELL).astype(int), 0, GRID - 1)
            vi = np.clip((a[:, 1] / CELL).astype(int), 0, GRID - 1)
            for x, y, vv in zip(ui, vi, v):
                if np.isfinite(vv):
                    acc[y][x].append(vv)
        m = np.full((GRID, GRID), np.nan)
        for yy in range(GRID):
            for xx in range(GRID):
                if len(acc[yy][xx]) >= 20:
                    m[yy, xx] = float(np.median(acc[yy][xx]))
        return m

    lo = build(groups["low"], "du")
    hi = build(groups["high"], "du")
    ok = np.isfinite(lo) & np.isfinite(hi)
    print(f"cells populated in both regime maps: {int(ok.sum())}/{GRID*GRID}")
    if ok.sum() < 50:
        print("too few cells; cannot compare")
        return
    r = float(np.corrcoef(lo[ok], hi[ok])[0, 1])
    rng = np.random.default_rng(7)
    null = []
    for _ in range(600):
        y = hi[ok].copy(); rng.shuffle(y)
        null.append(float(np.corrcoef(lo[ok], y)[0, 1]))
    null = np.array(null)

    # the same comparison, but within ONE regime, split by time. This is the control:
    # it bounds how much two arbitrary halves of the SAME regime can differ, so a
    # high cross-regime correlation can be read against a measured floor.
    half = len(groups["low"]) // 2
    lo_a = build(groups["low"][:half], "du")
    lo_b = build(groups["low"][half:], "du")
    ok2 = np.isfinite(lo_a) & np.isfinite(lo_b)
    r_within = float(np.corrcoef(lo_a[ok2], lo_b[ok2])[0, 1]) if ok2.sum() > 50 else np.nan

    print(f"\nH3 test: is the spatial pattern the same in BOTH cloud regimes?")
    print(f"  correlation of the per-cell disparity map between regimes: r = {r:+.3f}")
    print(f"  shuffled null: mean {null.mean():+.3f}, 95th pct {np.percentile(null, 95):+.3f}")
    print(f"  control, within-regime split (low vs low):                 "
          f"r = {r_within:+.3f}")
    print(f"  the two regimes' median disparities: low {np.nanmedian(lo):.2f} px, "
          f"high {np.nanmedian(hi):.2f} px")
    q = (np.nanpercentile(lo, 90) - np.nanpercentile(lo, 10))
    print(f"  spatial spread of the maps: {q:.2f} px (p10-p90)")

    if r > 0.7 and r > np.percentile(null, 99):
        verdict = ("INSTRUMENTAL — the pattern is the same in both cloud regimes, "
                   "so clouds cannot be producing it")
    elif r < 0.3:
        verdict = ("CLIMATIC — the pattern moves with the cloud regime, which is what "
                   "a climatology of cloud position would do")
    else:
        verdict = "MIXED — the pattern partially follows the regime"
    print(f"\n  VERDICT: {verdict}")

    json.dump({"n_frames": len(K1), "n_low": len(groups["low"]),
               "n_high": len(groups["high"]), "r_between_regimes": r,
               "null_mean": float(null.mean()),
               "null_p95": float(np.percentile(null, 95)),
               "r_within_regime_control": float(r_within),
               "median_du_low": float(np.nanmedian(lo)),
               "median_du_high": float(np.nanmedian(hi)),
               "verdict": verdict},
              open(os.path.join(HERE, "w3_h3_test.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    vmin = np.nanpercentile(np.concatenate([lo[ok], hi[ok]]), 2)
    vmax = np.nanpercentile(np.concatenate([lo[ok], hi[ok]]), 98)
    for ax, m, t in ((axes[0], lo, f"ceilometer LOW (n={len(groups['low'])})"),
                     (axes[1], hi, f"ceilometer HIGH (n={len(groups['high'])})"),
                     (axes[2], lo - hi, "LOW minus HIGH")):
        im = ax.imshow(m, origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Per-cell disparity under two cloud regimes — r = {r:+.2f} "
                 f"(within-regime control {r_within:+.2f})", fontsize=10, y=1.04)
    fig.savefig(os.path.join(OUT, "w3_regime_maps.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/w3_regime_maps.png and {HERE}/w3_h3_test.json")


if __name__ == "__main__":
    main()
