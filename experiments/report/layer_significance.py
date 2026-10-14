# PURPOSE: Test whether the layer-regime improvement reported in the findings
#          survives the null test that destroyed the plain windowed correlation.
#
#          The temporal-alignment audit showed that the metric's per-window
#          "best shift" is an artefact of maximising a correlation curve over
#          noise: with a null that rolls the estimate against a fixed ceilometer,
#          the observed peak was exceeded by ~70 % of null draws. If that is true
#          of the plain comparison, it may also be true of the regime-split
#          comparison, in which case the apparent 0.18 -> 0.48/0.57 improvement is
#          an artefact of the same machinery and must not be quoted.
#
#          The null preserves both series' autocorrelation (a circular roll of the
#          estimate) and destroys only their relative timing, which is exactly the
#          hypothesis under test: "does this pairing carry information?"
# INPUTS: the roll-corrected run's series.json and the ceilometer log.
# OUTPUTS: observed versus null distributions per regime, with p-values.
# KEYWORDS: null_test, significance, layer_regime, selection_inflation, validation
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.metrics import alignment as align                              # noqa: E402
from clouds.sources import ceilometer as C                                 # noqa: E402

RUN = "/home/iustimov/clouds_data/logs/20261013T112649Z_evaluate_clouds_2025-05-23_3565c5"
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"
N_DRAWS = 120


# PURPOSE: Mean of the project's windowed correlation over a set of ceilometer
#          samples, using the same settings as the pipeline.
# INPUTS: algo times/heights, ceilometer times/heights, window/shift/tolerance.
# OUTPUTS: (mean r over scored windows, number of windows).
# KEYWORDS: windowed_correlation, metric
def mean_r(at, ah, ct, ch, window_min=30, shift_s=300, tol_s=60, hop=15):
    t = [x for x, v in zip(at, ah) if np.isfinite(v)]
    h = [v for v in ah if np.isfinite(v)]
    if len(t) < 10 or len(ct) < 10:
        return float("nan"), 0
    rows = align.windowed_correlation(t, h, ct, ch, window_min, shift_s, tol_s, hop)
    rs = [r["r"] for r in rows if np.isfinite(r["r"])]
    return (float(np.mean(rs)) if rs else float("nan")), len(rs)


def main():
    s = json.load(open(os.path.join(RUN, "series.json")))
    at = [dt.datetime.fromisoformat(x) for x in s["times_ceilometer_clock"]]
    ah = np.array([np.nan if v is None else v for v in s["filtered_h"]], float)
    samples = C.parse_ceilometer_log(LOG)
    lo, hi = dt.time(12, 0, 0), dt.time(17, 0, 0)
    ce = [x for x in samples if x.time.date() == dt.date(2025, 5, 23)
          and lo <= x.time.time() <= hi and x.h1 is not None]
    ct = [x.time for x in ce]
    ch1 = np.array([x.h1 for x in ce], float)
    print(f"algorithm frames {len(at)}, ceilometer samples {len(ct)}\n")

    regimes = [("all samples (as reported)", np.ones(len(ch1), bool)),
               ("lower layer only (<2750 m)", ch1 < 2750),
               ("upper layer only (>3500 m)", ch1 > 3500)]

    rng = np.random.default_rng(20251013)
    out = {}
    print(f"{'regime':>28} {'obs mean r':>11} {'null mean':>10} {'null p95':>9} "
          f"{'p':>7} {'verdict':>12}")
    for name, mask in regimes:
        ctm = [ct[i] for i in range(len(ct)) if mask[i]]
        chm = list(ch1[mask])
        obs, nw = mean_r(at, ah, ctm, chm)
        # null: circularly roll the estimate, preserving its autocorrelation
        null = []
        n = len(ah)
        for _ in range(N_DRAWS):
            k = int(rng.integers(1, n))
            rolled = np.roll(ah, k)
            r, _ = mean_r(at, rolled, ctm, chm)
            if np.isfinite(r):
                null.append(r)
        null = np.array(null)
        p = float(np.mean(null >= obs)) if len(null) else float("nan")
        verdict = "real" if p < 0.05 else ("weak" if p < 0.2 else "NOT above null")
        out[name] = dict(observed=obs, null_mean=float(null.mean()),
                         null_p95=float(np.percentile(null, 95)),
                         p=p, n_windows=nw, n_null=len(null))
        print(f"{name:>28} {obs:>11.3f} {null.mean():>10.3f} "
              f"{np.percentile(null,95):>9.3f} {p:>7.3f} {verdict:>12}")

    json.dump(out, open(os.path.join(HERE, "layer_significance.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'layer_significance.json')}")
    print("\nA p-value here is the fraction of rolled-series draws that matched or")
    print("beat the observation. Above 0.05 the comparison carries no demonstrated")
    print("information, regardless of how large the apparent correlation is.")


if __name__ == "__main__":
    main()
