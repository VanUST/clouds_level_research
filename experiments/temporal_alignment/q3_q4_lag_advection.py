# PURPOSE: Answer questions 3 and 4 — what lag does the measured cloud motion
#          imply, and does the sequence of per-window "best shifts" behave like
#          advection over a fixed instrument separation? The advection model is
#          lag = (r . v) / |v|^2, where r is the (unknown) camera-to-ceilometer
#          baseline vector and v is the measured cloud velocity; the baseline is
#          FITTED and its explanatory power is compared against a shuffled null,
#          so the answer does not depend on inventing a separation.
# INPUTS: the q1 shift-scan JSON, the q2 motion JSON, and the two stored runs'
#         metrics.json for the pipeline's own best-shift sequences.
# OUTPUTS: experiments/temporal_alignment/q3_q4_lag_advection.json and
#          q3_q4_lag_advection.png, plus a printed report.
# KEYWORDS: advection, lag, baseline, significance, best_shift_sequence
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import common  # noqa: E402


# PURPOSE: Read the pipeline's own per-window best shift and r from a run's
#          metrics.json.
# INPUTS: run_dir (str); which ("raw_windows_detail"/"filtered_windows_detail").
# OUTPUTS: list of dicts {start, best_shift, r} in run order.
# KEYWORDS: metrics, pipeline_output, io
def load_pipeline_windows(run_dir: str, which: str) -> list:
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        d = json.load(fh)
    return [{"start": w["start"], "best_shift": w["best_shift"], "r": w["r"]}
            for w in d[which]]


# PURPOSE: Fit the advection model lag_t = (r . v_t)/|v_t|^2 to a best-shift
#          sequence, where v_t is the measured cloud velocity in an arbitrary
#          fixed 2-D frame (the image frame). The baseline length |r| is the
#          implied instrument separation; the model's R^2 is compared against a
#          shuffled null so a good-looking fit cannot be pure chance.
# INPUTS: shifts (np.ndarray s), vx, vy (np.ndarray m/s), n_null (int draws).
# OUTPUTS: dict {rx, ry, sep_m, r2, r2_null_p95, p_value, predict}; predict is
#          the fitted lag per window. NaN fields when there are < 4 windows.
# KEYWORDS: least_squares, baseline_fit, permutation, advection_model
def fit_baseline(shifts, vx, vy, n_null: int = 2000):
    s = np.asarray(shifts, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    ok = np.isfinite(s) & np.isfinite(vx) & np.isfinite(vy)
    s, vx, vy = s[ok], vx[ok], vy[ok]
    if s.size < 4:
        return {"sep_m": float("nan"), "r2": float("nan"), "p_value": float("nan")}
    v2 = vx ** 2 + vy ** 2
    good = v2 > 1e-6
    s, vx, vy, v2 = s[good], vx[good], vy[good], v2[good]
    A = np.column_stack([vx / v2, vy / v2])
    sol, *_ = np.linalg.lstsq(A, s, rcond=None)
    pred = A @ sol
    ss_res = float(np.sum((s - pred) ** 2))
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rng = np.random.default_rng(20250523)
    null = []
    for _ in range(n_null):
        sp = rng.permutation(s)
        solp, *_ = np.linalg.lstsq(A, sp, rcond=None)
        pp = A @ solp
        ssr = float(np.sum((sp - pp) ** 2))
        sst = float(np.sum((sp - sp.mean()) ** 2))
        null.append(1.0 - ssr / sst if sst > 0 else 0.0)
    null = np.array(null)
    return {
        "rx": float(sol[0]), "ry": float(sol[1]),
        "sep_m": float(np.hypot(sol[0], sol[1])),
        "r2": float(r2), "r2_null_median": float(np.median(null)),
        "r2_null_p95": float(np.percentile(null, 95)),
        "p_value": float(np.mean(null >= r2)),
        "predict": [float(x) for x in pred],
    }


# PURPOSE: Summarise how smooth or jumpy a best-shift sequence is, in absolute
#          terms and against a shuffle null.
# INPUTS: shifts (np.ndarray), n_null (int).
# OUTPUTS: dict {lag1_autocorr, mean_abs_jump_s, frac_jump_gt_200s,
#          max_abs_jump_s}; NaN when fewer than 3 windows.
# KEYWORDS: smoothness, autocorrelation, jumpiness, time_series
def sequence_diagnostics(shifts: np.ndarray):
    s = np.asarray(shifts, dtype=float)
    if s.size < 3:
        return {"lag1_autocorr": float("nan")}
    d = np.diff(s)
    ac = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if np.std(s[:-1]) > 0 else float("nan")
    return {"lag1_autocorr": ac,
            "mean_abs_jump_s": float(np.mean(np.abs(d))),
            "median_abs_jump_s": float(np.median(np.abs(d))),
            "frac_jump_gt_200s": float(np.mean(np.abs(d) > 200)),
            "max_abs_jump_s": float(np.max(np.abs(d)))}


# PURPOSE: Merge the per-window motion from q2 with the best-shift and r from q1
#          (or the pipeline) by window start time, tolerating a 1 s grid offset.
# INPUTS: q1_rows (list), q2_windows (list), shift_key (str in q1 rows).
# OUTPUTS: dict of np.ndarrays {start_s, shift, r, speed, angle, height} for the
#          windows present in both.
# KEYWORDS: merge, window_match, alignment
def merge(q1_rows, q2_windows, shift_key="pipe_peak_shift"):
    q2_by_t = {}
    for w in q2_windows:
        t = dt.datetime.fromisoformat(w["start_ceil"]).timestamp()
        q2_by_t[round(t)] = w
    starts, shifts, rs, speeds, angles, heights = [], [], [], [], [], []
    for row in q1_rows:
        if shift_key not in row or row[shift_key] is None:
            continue
        t = dt.datetime.fromisoformat(row["start"]).timestamp()
        w = None
        for cand in (round(t) - 1, round(t), round(t) + 1):
            if cand in q2_by_t:
                w = q2_by_t[cand]
                break
        if w is None:
            continue
        starts.append(t)
        shifts.append(float(row[shift_key]))
        rs.append(row.get("pipe_peak_r") if row.get("pipe_peak_r") is not None else row.get("r"))
        speeds.append(float(w["speed_mean"]))
        angles.append(float(w["angle_mean_deg"]))
        heights.append(float(w["height_mean"]))
    ang = np.array(angles, dtype=float)
    sp = np.array(speeds, dtype=float)
    rad = np.radians(ang)
    return {"start_s": np.array(starts, dtype=float),
            "shift": np.array(shifts, dtype=float),
            "r": np.array(rs, dtype=float),
            "speed": sp, "angle": ang, "height": np.array(heights, dtype=float),
            "vx": sp * np.cos(rad), "vy": sp * np.sin(rad)}


# PURPOSE: Report the implied-lag arithmetic: for the measured speed range, the
#          separation a given observed lag would require, and the lag a plausible
#          separation would produce.
# INPUTS: speeds (np.ndarray m/s), observed_abs_shift_s (np.ndarray).
# OUTPUTS: dict with speed p10/median/p90, required separation for the median
#          and max observed |shift|, and lag at separations 50, 250, 1000, 2000 m
#          using the median speed.
# KEYWORDS: implied_lag, separation_bound, plausibility
def implied_lag(speeds, observed_abs_shift_s):
    v = np.asarray(speeds, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    s = np.asarray(observed_abs_shift_s, dtype=float)
    if v.size == 0:
        return {}
    med = float(np.median(v))
    out = {"speed_p10_mps": float(np.percentile(v, 10)),
           "speed_median_mps": med,
           "speed_p90_mps": float(np.percentile(v, 90)),
           "sep_for_median_shift_m": float(np.median(s) * med),
           "sep_for_max_shift_m": float(np.max(s) * med)}
    for d in (50, 250, 1000, 2000):
        out[f"lag_at_{d}m_s"] = float(d / med)
    return out


# PURPOSE: Draw the question 3/4 figure: shift sequence, wind speed/direction,
#          |shift| vs speed (with separation curves), fitted vs observed shift.
# INPUTS: merged (dict), fit (dict), pipeline_seqs (dict), out_png (str).
# OUTPUTS: writes the PNG; returns None.
# KEYWORDS: figure, matplotlib, advection
def make_figure(merged, fit, pipeline_seqs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = float(np.asarray(merged["start_s"])[0])
    tt = (np.asarray(merged["start_s"]) - t0) / 60.0
    speed = np.asarray(merged["speed"])
    shift = np.asarray(merged["shift"])
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(tt, shift, "o-", label="filtered +-300 s peak (my scan)")
    for name, seq in pipeline_seqs.items():
        st = np.array([dt.datetime.fromisoformat(x["start"]).timestamp()
                       for x in seq])
        ax[0, 0].plot((st - t0) / 60.0, [x["best_shift"] for x in seq],
                      ".--", alpha=0.6, label=name)
    ax[0, 0].axhline(300, color="k", ls=":")
    ax[0, 0].axhline(-300, color="k", ls=":")
    ax[0, 0].set_xlabel("minutes from window start")
    ax[0, 0].set_ylabel("best shift (s)")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_title("(a) per-window best shift sequence")
    ax[0, 1].plot(tt, speed, "o-")
    ax[0, 1].set_xlabel("minutes")
    ax[0, 1].set_ylabel("cloud speed (m/s)")
    ax[0, 1].set_title("(b) measured cloud speed per window")
    ax[1, 0].scatter(speed, np.abs(shift))
    vv = np.linspace(max(0.1, speed.min()), speed.max() + 0.5, 50)
    for d in (250, 500, 1000, 2000):
        ax[1, 0].plot(vv, d / vv, "--", lw=1, label=f"separation {d} m")
    ax[1, 0].set_xlabel("cloud speed (m/s)")
    ax[1, 0].set_ylabel("|best shift| (s)")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title("(c) |shift| vs speed: separation an advection lag needs")
    if fit.get("predict"):
        ax[1, 1].scatter(shift, fit["predict"])
        lim = [min(shift.min(), min(fit["predict"])),
               max(shift.max(), max(fit["predict"]))]
        ax[1, 1].plot(lim, lim, "k--")
        ax[1, 1].set_xlabel("observed best shift (s)")
        ax[1, 1].set_ylabel("advection-model shift (s)")
        ax[1, 1].set_title(f"(d) fitted baseline |r|={fit['sep_m']:.0f} m, "
                           f"R2={fit['r2']:.2f} (null p={fit['p_value']:.2f})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"figure written: {out_png}")


# PURPOSE: Entry point: load everything, fit, report, save JSON and figure.
# INPUTS: none.
# OUTPUTS: q3_q4_lag_advection.json and q3_q4_lag_advection.png.
# KEYWORDS: main, entry_point
def main():
    with open(os.path.join(HERE, "q1_shift_scan.json")) as fh:
        q1 = json.load(fh)
    with open(os.path.join(HERE, "q2_cloud_motion.json")) as fh:
        q2 = json.load(fh)

    pipeline_seqs = {
        "roll-corrected raw (pipeline)":
            load_pipeline_windows(common.RUN_ROLL_CORRECTED, "raw_windows_detail"),
        "roll-corrected filt (pipeline)":
            load_pipeline_windows(common.RUN_ROLL_CORRECTED, "filtered_windows_detail"),
        "roll=0 raw (pipeline)":
            load_pipeline_windows(common.RUN_ROLL_ZERO, "raw_windows_detail"),
    }
    out = {"pipeline_sequence_diagnostics": {}, "fits": {}}
    for name, seq in pipeline_seqs.items():
        out["pipeline_sequence_diagnostics"][name] = sequence_diagnostics(
            np.array([x["best_shift"] for x in seq], dtype=float))

    for key, label in (("filtered", "filtered"), ("raw", "raw")):
        merged = merge(q1[key], q2["windows"])
        fit = fit_baseline(merged["shift"], merged["vx"], merged["vy"])
        out["fits"][label] = {"merge": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in merged.items()},
                              "baseline": fit}
        print(f"\n=== {label}: advection baseline fit ===")
        print(f"windows merged: {merged['shift'].size}")
        print(f"fitted baseline |r| = {fit['sep_m']:.0f} m   R2 = {fit['r2']:.3f}   "
              f"null R2 median = {fit.get('r2_null_median', float('nan')):.3f}, "
              f"p = {fit['p_value']:.3f}")
        print("implied lag:", json.dumps(implied_lag(merged["speed"],
                                                     np.abs(merged["shift"])), indent=1))
        print("sequence diagnostics (this merged sequence):",
              json.dumps(sequence_diagnostics(merged["shift"]), indent=1))

    print("\n=== pipeline best-shift sequence smoothness (both runs) ===")
    for name, diag in out["pipeline_sequence_diagnostics"].items():
        print(f"  {name}: {json.dumps(diag)}")

    # Do the two runs' sequences agree? Same code, only the 1.5 deg roll differs.
    a = np.array([x["best_shift"] for x in pipeline_seqs["roll-corrected raw (pipeline)"]],
                 dtype=float)
    b = np.array([x["best_shift"] for x in pipeline_seqs["roll=0 raw (pipeline)"]],
                 dtype=float)
    n = min(a.size, b.size)
    out["run_agreement_raw"] = {
        "frac_identical": float(np.mean(a[:n] == b[:n])),
        "mean_abs_diff_s": float(np.mean(np.abs(a[:n] - b[:n]))),
        "corr": float(np.corrcoef(a[:n], b[:n])[0, 1]) if np.std(a[:n]) > 0 else float("nan"),
    }
    print("\n=== roll-corrected vs roll=0 best-shift agreement (raw) ===")
    print(json.dumps(out["run_agreement_raw"], indent=1))

    make_figure(out["fits"]["filtered"]["merge"], out["fits"]["filtered"]["baseline"],
                pipeline_seqs, os.path.join(HERE, "q3_q4_lag_advection.png"))
    with open(os.path.join(HERE, "q3_q4_lag_advection.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
