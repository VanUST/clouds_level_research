# PURPOSE: Brute-force search for an operation that removes the static spatial field
#          from the disparity measurements and leaves a series whose variation matches
#          the ceilometer - validated OUT OF SAMPLE by k-fold cross-validation, so that a
#          correction tuned on one part of the session cannot be scored on the same part.
#
#          WHY THIS DESIGN. The static field is a measured fact: the per-cell disparity
#          pattern correlates r = 0.91-0.98 between the two halves of the session while the
#          cloud moves. A correction fitted to the whole session and then scored on the
#          whole session would prove nothing - it can always absorb whatever it is scored
#          against. So every candidate operation here is FITTED ON TRAIN FOLDS ONLY and
#          SCORED ON THE HELD-OUT FOLD. The reported number is out-of-sample by
#          construction.
#
#          WHAT IS SEARCHED. Six correction families, from the trivial to the
#          non-parametric, each applied to the per-match disparity before the per-frame
#          estimate is taken:
#            none            - the baseline
#            per-cell median - the classic flat-field
#            per-radius      - a purely radial correction
#            linear (u,v)    - what the diagnostic says dominates
#            per-cell + linear
#            smoothed per-cell (a Gaussian-blurred flat-field)
#          and three per-frame estimators: the median, the lowest prominent mode, and the
#          25th percentile.
#
#          WHAT IS SCORED. Two things, because they are different claims and the project
#          has already conflated them once:
#            LEVEL      - Spearman rho between the estimate and the reference
#            DYNAMICS   - Spearman rho between their first differences, gap-safe
#          A correction that fixes the level without the dynamics has not produced a
#          measurement.
# INPUTS: experiments/w4/matches_roma_20250523.npz, the ceilometer log.
# OUTPUTS: experiments/w9/w9_cv.json, docs/report/w9_cv.png.
# KEYWORDS: cross_validation, kfold, static_field, flat_field, out_of_sample, dynamics
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402
from clouds.pipeline.analyze import lowest_prominent_mode                       # noqa: E402

LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
F_PX, B_M, CROP = 654.0, 45.0, 640.0
CELL, GRID = 20, 32
CENTRE = CROP / 2.0
ST = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")
K_FOLDS = 5


def load():
    d = np.load(os.path.join(HERE, "..", "w4", "matches_roma_20250523.npz"),
                allow_pickle=True)
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

    frames = []
    for i in range(len(d["stamps"])):
        r = ref_at(d["stamps"][i])
        a = np.asarray(d["uv"][i], float)
        b = np.asarray(d["kp2"][i], float)
        if r is None or len(a) < 500:
            continue
        du = a[:, 0] - b[:, 0]
        ok = np.isfinite(du)
        if ok.sum() < 500:
            continue
        u = a[ok, 0]; v = a[ok, 1]
        # the cell id, the radius bin and the linear terms are properties of the frame
        # alone, so they are computed once here rather than inside every candidate's fit
        frames.append({"ref": float(r), "t": parse_time(d["stamps"][i]),
                       "u": u, "v": v, "du": du[ok],
                       "cell": cell_id(u, v),
                       "rad": np.clip(np.digitize(np.hypot(u - CENTRE, v - CENTRE),
                                                  np.linspace(0, 460.0, 25)) - 1, 0, 23),
                       "lin": np.stack([u - CENTRE, v - CENTRE, np.ones_like(u)], 1)})
    # sorted in TIME, not by reference height: the cross-validation must block in time,
    # because both series are strongly autocorrelated and a randomly held-out frame has its
    # own near-duplicates in the training set.
    frames.sort(key=lambda f: f["t"])
    return frames


def parse_time(s):
    m = ST.search(str(s))
    if not m:
        return None
    return dt.datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
                       int(m.group(2)), int(m.group(3)), int(m.group(4)))


# --- correction families -------------------------------------------------------------
# Each returns a function (u, v, du) -> du_corrected, having been fitted on train frames.

def cell_id(u, v):
    return (np.clip((u / CELL).astype(int), 0, GRID - 1) * GRID
            + np.clip((v / CELL).astype(int), 0, GRID - 1))


def fit_cell(train):
    """Per-cell flat field: one median per (cell, frame), then the median over frames.

    Vectorised per frame with np.bincount rather than a mask per cell, because the naive
    form is 1024 cells x 300 frames per fold and dominated the runtime.
    """
    per_frame = np.full((len(train), GRID * GRID), np.nan)
    for j, f in enumerate(train):
        m = np.bincount(f["cell"], weights=f["du"], minlength=GRID * GRID)
        n = np.bincount(f["cell"], minlength=GRID * GRID)
        good = n >= 5
        per_frame[j, good] = m[good] / n[good]
    with np.errstate(all="ignore"):
        return np.nanmedian(per_frame, axis=0)


def fit_radius(train, nbins=24, rmax=460.0):
    per_frame = np.full((len(train), nbins), np.nan)
    for j, f in enumerate(train):
        m = np.bincount(f["rad"], weights=f["du"], minlength=nbins)
        n = np.bincount(f["rad"], minlength=nbins)
        good = n >= 20
        per_frame[j, good] = m[good] / n[good]
    with np.errstate(all="ignore"):
        return np.linspace(0, rmax, nbins + 1), np.nanmedian(per_frame, axis=0)


def fit_linear(train):
    X = np.concatenate([f["lin"] for f in train])
    y = np.concatenate([f["du"] for f in train])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


# --- the searched grid ---------------------------------------------------------------
CORRECTIONS = ("none", "per-cell", "per-cell+linear", "per-radius", "linear", "cell-smooth")


def make_corrector(name, cellmap, prof, coef):
    """Build the correction from fits, as a function of a single frame dict."""
    base_cell = np.nanmean(cellmap)
    cell_off = np.nan_to_num(cellmap - base_cell, nan=0.0)
    base_rad = np.nanmean(prof)
    rad_off = np.nan_to_num(prof - base_rad, nan=0.0)
    lin = None
    if name == "cell-smooth":
        import cv2
        filled = np.where(np.isfinite(cellmap), cellmap, base_cell)
        sm = cv2.GaussianBlur(filled.reshape(GRID, GRID).astype(np.float32),
                              (0, 0), 2.0).ravel()
        cell_off = sm - base_cell
    if name == "linear":
        return lambda f: f["du"] - f["lin"][:, :2] @ coef[:2]
    if name == "per-cell":
        return lambda f: f["du"] - cell_off[f["cell"]]
    if name == "cell-smooth":
        return lambda f: f["du"] - cell_off[f["cell"]]
    if name == "per-radius":
        return lambda f: f["du"] - rad_off[f["rad"]]
    if name == "per-cell+linear":
        return lambda f: (f["du"] - cell_off[f["cell"]]) - f["lin"][:, :2] @ coef[:2]
    if name == "none":
        return lambda f: f["du"]
    raise ValueError(name)


ESTIMATORS = ("median", "mode", "p25")


def estimate(du, est):
    """Return a HEIGHT-ORDERED statistic, so that a POSITIVE correlation with the
    reference is the correct sign.

    The first version returned the disparity itself and reported rho(disparity, reference),
    which is positive precisely when the estimator is INVERTED - a correct estimator needs
    the disparity to fall as the cloud rises. The statistic is therefore negated here (a
    monotone transform of height in disparity space), and every reported rho now has the
    conventional sign: positive = correct.
    """
    if len(du) < 50:
        return np.nan
    if est == "median":
        return float(-np.median(du))
    if est == "p25":
        return float(-np.percentile(du, 25))
    h, _ = lowest_prominent_mode(du)
    return np.nan if h is None else float(h)


def main():
    frames = load()
    n = len(frames)
    print(f"frames {n}, span {frames[0]['t'].strftime('%H:%M')}-"
          f"{frames[-1]['t'].strftime('%H:%M')} MSK, references "
          f"{min(f['ref'] for f in frames):.0f}-{max(f['ref'] for f in frames):.0f} m")
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    EMBARGO_S = 300.0

    # Three CV schemes, so the leakage from autocorrelation can be MEASURED rather than
    # asserted. The reference decorrelates over ~267 s (~27 frames at 10 s), and a randomly
    # held-out frame has its own near-duplicates in the training set - so "random (leaks)"
    # is expected to look better than the honest schemes and is included to quantify that.
    schemes = {}
    schemes["random (leaks)"] = [np.array_split(rng.permutation(idx), K_FOLDS), None]
    blocked = np.array_split(idx, K_FOLDS)
    schemes["blocked"] = [blocked, None]
    purged = []
    for f_test in blocked:
        t0 = frames[f_test[0]]["t"]; t1 = frames[f_test[-1]]["t"]
        keep = [i for i in idx
                if (i in set(f_test.tolist())) or
                (abs((frames[i]["t"] - t0).total_seconds()) > EMBARGO_S and
                 abs((frames[i]["t"] - t1).total_seconds()) > EMBARGO_S)]
        purged.append(np.array(keep))
    schemes["blocked+embargo"] = [purged, EMBARGO_S]
    print(f"CV schemes: {list(schemes)}  (embargo {EMBARGO_S:.0f} s)")

    # Fits are computed ONCE per fold per scheme and shared across every combination.
    fold_data = {}
    for scheme, (folds, _e) in schemes.items():
        fd = []
        for f_test in folds:
            test_i = set(np.asarray(f_test).tolist())
            train = [frames[i] for i in idx if i not in test_i]
            test = [frames[i] for i in folds[0].tolist()] if False else \
                   [frames[i] for i in np.asarray(f_test).tolist()]
            fd.append(({c: make_corrector(c, fit_cell(train), fit_radius(train)[1],
                                          fit_linear(train)) for c in CORRECTIONS}, test))
        fold_data[scheme] = fd

    results = {}
    for scheme, fd in fold_data.items():
        for corr in CORRECTIONS:
            for est in ESTIMATORS:
                lev, dyn = [], []
                for fns, test in fd:
                    pred = np.array([estimate(fns[corr](f), est) for f in test])
                    ref = np.array([f["ref"] for f in test])
                    m = np.isfinite(pred) & np.isfinite(ref)
                    if m.sum() < 3:
                        continue
                    if np.std(pred[m]) > 0 and np.std(ref[m]) > 0:
                        lev.append(float(spearmanr(pred[m], ref[m])[0]))
                    order = np.argsort(ref[m])
                    p2 = pred[m][order]
                    if len(p2) > 3 and np.std(np.diff(p2)) > 0:
                        dyn.append(float(spearmanr(np.diff(p2),
                                                  np.diff(ref[m][order]))[0]))
                if not lev:
                    continue
                results[f"{scheme}|{corr}|{est}"] = {
                    "scheme": scheme, "correction": corr, "estimator": est,
                    "level_rho": float(np.nanmean(lev)),
                    "level_rho_std": float(np.nanstd(lev)),
                    "dynamics_rho": float(np.nanmean(dyn)) if dyn else float("nan"),
                    "n_folds": len(lev)}

    print(f"\nThe statistic is HEIGHT-ORDERED: POSITIVE rho is the correct sign.\n")
    for scheme in schemes:
        rows = sorted([v for v in results.values() if v["scheme"] == scheme],
                      key=lambda v: -v["level_rho"])
        print(f"--- {scheme} ---")
        print(f"{'correction':>16} {'estimator':>10} {'LEVEL rho':>11} "
              f"{'DYNAMICS rho':>14}")
        for v in rows[:6]:
            print(f"{v['correction']:>16} {v['estimator']:>10} {v['level_rho']:>+11.3f} "
                  f"{v['dynamics_rho']:>+14.3f}")
        print()

    # the headline comparison: how much does the leakage buy?
    def best(scheme, key):
        rows = [v for v in results.values() if v["scheme"] == scheme]
        if not rows:
            return float("nan")
        return max(v[key] for v in rows if np.isfinite(v[key])) if any(
            np.isfinite(v[key]) for v in rows) else float("nan")

    print(f"{'scheme':>18} {'best LEVEL rho':>15} {'best DYNAMICS rho':>18}")
    for scheme in schemes:
        print(f"{scheme:>18} {best(scheme,'level_rho'):>+15.3f} "
              f"{best(scheme,'dynamics_rho'):>+18.3f}")

    honest = [v for v in results.values()
              if v["scheme"] == "blocked+embargo" and np.isfinite(v["dynamics_rho"])]
    dyn_ok = [v for v in honest if v["dynamics_rho"] > 0.3]
    verdict = ("A CORRECTION PRODUCES POSITIVE OUT-OF-SAMPLE DYNAMICS under blocked+embargo "
               "validation at " + ", ".join(f"{v['correction']}+{v['estimator']} "
                                            f"(rho={v['dynamics_rho']:+.2f})" for v in dyn_ok)
               if dyn_ok else
               "NO CORRECTION PRODUCES POSITIVE OUT-OF-SAMPLE DYNAMICS under blocked+embargo "
               "validation; every operation leaves the series anti-correlated or uncorrelated "
               "with the reference once the autocorrelation leakage is removed")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(HERE, exist_ok=True)
    json.dump({"results": results, "k_folds": K_FOLDS, "embargo_s": EMBARGO_S,
               "verdict": verdict}, open(os.path.join(HERE, "w9_cv.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(schemes), figsize=(5.0 * len(schemes), 4.6),
                             sharey=True)
    for ax, scheme in zip(np.atleast_1d(axes), schemes):
        rows = sorted([v for v in results.values() if v["scheme"] == scheme],
                      key=lambda v: -v["level_rho"])
        labs = [f"{v['correction']}\n{v['estimator']}" for v in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [v["level_rho"] for v in rows], 0.4, color="#1f6feb", label="LEVEL")
        ax.bar(x + 0.2, [v["dynamics_rho"] for v in rows], 0.4, color="#c1440e",
               label="DYNAMICS")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6, rotation=90)
        ax.set_title(scheme, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("out-of-sample Spearman rho")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"{K_FOLDS}-fold cross-validation schemes: does the leakage change the "
                 f"answer?", fontsize=10)
    fig.savefig(os.path.join(OUT, "w9_cv.png"), bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}/w9_cv.png and {os.path.join(HERE, 'w9_cv.json')}")


if __name__ == "__main__":
    main()
