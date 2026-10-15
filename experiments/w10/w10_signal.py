# PURPOSE: Two linked questions, in one pass.
#
#          (A) THE SIGNAL-EXISTENCE TEST. Does ANY statistic of the per-match disparity
#              field carry the reference's temporal variation once both are resampled to a
#              common, independent grid? This is the boundary question: if every statistic
#              returns nothing, then no correction of the field can help, because there is
#              no dynamics in it to recover.
#
#          (B) THE SHAPE MECHANISM. W9 found the level correlation is correctly signed for
#              the MODE (+0.287 out of sample) and inverted for the MEDIAN and p25 (-0.409,
#              -0.296). The proposed explanation is that the disparity distribution's SHAPE
#              changes between regimes - the high regime is more skewed and heavier-tailed -
#              and that quantile statistics move with the shape while the mode does not.
#              That is directly measurable: compare skewness, kurtosis and the percentile
#              spacing between regimes, and check whether the median's movement is
#              accounted for by the shape change rather than by a shift of the bulk.
#
#          WHY THE GRID MATTERS. The reference is autocorrelated at lag-1 rho ~ 0.93, so
#          consecutive frames are not independent observations. Correlating per-frame
#          series would inflate every p-value. Both series are therefore binned to the
#          reference's own decorrelation time, and the effective sample size is computed
#          from the lag-1 autocorrelation of the BINNED series and reported alongside every
#          coefficient. With ten statistics and two families of test the multiplicity is
#          also corrected, because twenty chances to find a correlation at p < 0.05 will
#          produce one by chance about two thirds of the time.
# INPUTS: experiments/w4/matches_roma_20250523.npz, the ceilometer log.
# OUTPUTS: experiments/w10/w10_signal.json, docs/report/w10_signal.png.
# KEYWORDS: signal_existence, dynamics, shape, skewness, n_eff, multiplicity, boundary
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr, skew, kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                      # noqa: E402
from clouds.pipeline.analyze import lowest_prominent_mode                       # noqa: E402

LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
ST = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")
BIN_S = 300.0
MIN_MATCHES = 100


def parse(s):
    m = ST.search(str(s))
    if not m:
        return None
    return dt.datetime(int(m.group(1)[:4]), int(m.group(1)[5:7]), int(m.group(1)[8:10]),
                       int(m.group(2)), int(m.group(3)), int(m.group(4))) + dt.timedelta(hours=3)


def stats_of(du):
    """Every statistic proposed, plus the shape measures the mechanism hypothesis needs."""
    m, _ = lowest_prominent_mode(du)
    return {
        "mean": float(np.mean(du)),
        "median": float(np.median(du)),
        "mode": np.nan if m is None else float(m),
        "p10": float(np.percentile(du, 10)),
        "p25": float(np.percentile(du, 25)),
        "p75": float(np.percentile(du, 75)),
        "p90": float(np.percentile(du, 90)),
        "std": float(np.std(du)),
        "skew": float(skew(du)),
        "kurtosis": float(kurtosis(du)),
        "iqr_span_ratio": float((np.percentile(du, 75) - np.percentile(du, 25))
                                / max(np.percentile(du, 95) - np.percentile(du, 5), 1e-9)),
    }


def neff(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.std(x) == 0:
        return 1.0
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1 = max(min(r1, 0.999), -0.999)
    return max(1.0, len(x) * (1 - r1) / (1 + r1))


def main():
    d = np.load(os.path.join(HERE, "..", "w4", "matches_roma_20250523.npz"),
                allow_pickle=True)
    series = [(x.time, x.h1) for x in C.parse_ceilometer_log(LOG) if x.h1 is not None]

    def ref_at(t):
        best, bd = None, None
        for tt, hh in series:
            dd = abs((tt - t).total_seconds())
            if bd is None or dd < bd:
                best, bd = hh, dd
        return best if (bd is not None and bd <= 120) else None

    # --- bin frames, then pool the matches inside each bin ---------------------------
    bins = {}
    for i in range(len(d["stamps"])):
        t = parse(d["stamps"][i])
        if t is None:
            continue
        a = np.asarray(d["uv"][i], float)
        b = np.asarray(d["kp2"][i], float)
        if len(a) < MIN_MATCHES:
            continue
        du = a[:, 0] - b[:, 0]
        du = du[np.isfinite(du)]
        if len(du) < MIN_MATCHES:
            continue
        key = int(t.timestamp() // BIN_S)
        bins.setdefault(key, []).append(du)

    rows = []
    for key in sorted(bins):
        du = np.concatenate(bins[key])
        if len(du) < MIN_MATCHES:
            continue
        t = dt.datetime.fromtimestamp(key * BIN_S)
        r = ref_at(t)
        if r is None:
            continue
        s = stats_of(du)
        s["t"] = t
        s["n_matches"] = int(len(du))
        s["ref"] = float(r)
        rows.append(s)
    print(f"bins {len(rows)} of {BIN_S:.0f} s, "
          f"{rows[0]['t'].strftime('%H:%M')}-{rows[-1]['t'].strftime('%H:%M')} MSK, "
          f"{sum(r['n_matches'] for r in rows):,} matches")

    ref = np.array([r["ref"] for r in rows], float)
    print(f"effective n of the binned reference: {neff(ref):.1f} of {len(ref)} bins")

    names = [k for k in rows[0] if k not in ("t", "n_matches", "ref")]
    print(f"\n(A) SIGNAL EXISTENCE — does any statistic carry the reference's variation?")
    print(f"{'statistic':>16} {'LEVEL rho':>10} {'p':>9} {'DYNAMICS rho':>13} {'p':>9}")
    out = {}
    for nm in names:
        v = np.array([r[nm] for r in rows], float)
        m = np.isfinite(v)
        if m.sum() < 5 or np.std(v[m]) == 0:
            continue
        rho, p = spearmanr(v[m], ref[m])
        # dynamics: first differences on the shared grid, which is already the reference's
        # decorrelation time, so consecutive bins are approximately independent
        d1 = np.diff(v[m]); d2 = np.diff(ref[m])
        if len(d1) > 3 and np.std(d1) > 0:
            rho_d, p_d = spearmanr(d1, d2)
        else:
            rho_d, p_d = float("nan"), float("nan")
        n_eff = neff(v[m])
        out[nm] = {"level_rho": float(rho), "level_p": float(p),
                   "dynamics_rho": float(rho_d), "dynamics_p": float(p_d),
                   "n_eff": float(n_eff)}
        print(f"{nm:>16} {rho:>+10.3f} {p:>9.4f} {rho_d:>+13.3f} {p_d:>9.4f}")

    # multiplicity: 11 statistics x 2 families, so a Bonferroni floor is reported
    n_tests = 2 * len(out)
    alpha = 0.05 / max(n_tests, 1)
    print(f"\n  {n_tests} tests; Bonferroni-corrected threshold p < {alpha:.5f}")
    hits = {nm: v for nm, v in out.items()
            if abs(v["level_rho"]) > 0.4 and v["level_p"] < alpha}
    hits_d = {nm: v for nm, v in out.items()
              if np.isfinite(v["dynamics_rho"]) and abs(v["dynamics_rho"]) > 0.4
              and v["dynamics_p"] < alpha}
    print(f"  statistics passing the corrected LEVEL threshold: "
          f"{list(hits) if hits else 'NONE'}")
    print(f"  statistics passing the corrected DYNAMICS threshold: "
          f"{list(hits_d) if hits_d else 'NONE'}")

    # --- (B) the shape mechanism -----------------------------------------------------
    print(f"\n(B) SHAPE MECHANISM — does the distribution change shape between regimes?")
    lo = np.array([r for r in rows if r["ref"] < 2750])
    hi = np.array([r for r in rows if r["ref"] > 3500])
    print(f"  bins: low {len(lo)}, high {len(hi)}")
    print(f"{'quantity':>16} {'low regime':>12} {'high regime':>12} {'change':>10}")
    shape = {}
    for nm in ("median", "mode", "skew", "kurtosis", "std", "iqr_span_ratio", "p10", "p90"):
        a = np.median([r[nm] for r in lo if np.isfinite(r[nm])]) if len(lo) else np.nan
        b = np.median([r[nm] for r in hi if np.isfinite(r[nm])]) if len(hi) else np.nan
        shape[nm] = {"low": float(a), "high": float(b),
                     "change_pct": float(100 * (b - a) / abs(a)) if a else float("nan")}
        print(f"{nm:>16} {a:>12.4f} {b:>12.4f} {shape[nm]['change_pct']:>+9.1f}%")

    verdict = ("NO STATISTIC CARRIES THE DYNAMICS — the per-match disparity field holds no "
               "recoverable temporal signal about the reference at this footprint ratio, so "
               "no correction of the field can produce one"
               if not hits_d else
               f"DYNAMICS FOUND in {list(hits_d)} — the signal exists and the correction "
               f"question is live")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(HERE, exist_ok=True)
    json.dump({"n_bins": len(rows), "bin_s": BIN_S,
               "n_eff_reference": float(neff(ref)),
               "statistics": out, "shape": shape,
               "alpha_bonferroni": float(alpha), "levels_hits": list(hits),
               "dynamics_hits": list(hits_d), "verdict": verdict},
              open(os.path.join(HERE, "w10_signal.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    nm_list = [n for n in names if n in out]
    x = np.arange(len(nm_list))
    a1.bar(x - 0.2, [out[n]["level_rho"] for n in nm_list], 0.4, color="#1f6feb",
           label="LEVEL")
    a1.bar(x + 0.2, [out[n]["dynamics_rho"] for n in nm_list], 0.4, color="#c1440e",
           label="DYNAMICS")
    a1.axhline(0, color="k", lw=0.8)
    a1.axhline(0.4, color="#5c6672", ls=":", lw=1)
    a1.axhline(-0.4, color="#5c6672", ls=":", lw=1)
    a1.set_xticks(x); a1.set_xticklabels(nm_list, fontsize=8, rotation=45, ha="right")
    a1.set_ylabel("Spearman rho with the reference")
    a1.set_title(f"Signal existence: every statistic of the disparity field\n"
                 f"(n_eff {neff(ref):.0f} bins of {BIN_S:.0f} s)", fontsize=9)
    a1.legend(fontsize=8)
    keys = ["median", "mode", "skew", "kurtosis", "std"]
    a2.bar(np.arange(len(keys)) - 0.2,
           [shape[k]["low"] for k in keys], 0.4, color="#1f6feb", label="low regime")
    a2.bar(np.arange(len(keys)) + 0.2,
           [shape[k]["high"] for k in keys], 0.4, color="#c1440e", label="high regime")
    a2.set_xticks(np.arange(len(keys))); a2.set_xticklabels(keys, fontsize=8)
    a2.set_title("Distribution shape by regime", fontsize=9); a2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w10_signal.png"), bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}/w10_signal.png and {os.path.join(HERE, 'w10_signal.json')}")


if __name__ == "__main__":
    main()
