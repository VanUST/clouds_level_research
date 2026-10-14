# PURPOSE: Aggregate the per-candidate JSON written by evaluate.py into the ranked
#          table and the figures the pre-registration asks for, without touching the
#          GPU or the data. The decision rule applied here is the one fixed in
#          PREREGISTRATION.md: rank by the median per-frame plane-fit residual, and
#          call a candidate "within noise" when its paired difference against the
#          baseline is both below 5 % of the baseline and not significant.
# INPUTS: experiments/matchers/results/*.json (one file per backend/tag/session).
# OUTPUTS: stdout tables, experiments/matchers/report.json, films under
#          experiments/matchers/figures/.
# KEYWORDS: aggregation, ranking, figures, paired_test, report
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures")

LABELS = {"kornia_loftr": "LoFTR (kornia)", "efficient_loftr": "EfficientLoFTR",
          "roma": "RoMa (romatch)", "roma_c05": "RoMa (cert>=0.05)",
          "dkm": "DKM", "dkm_c05": "DKM (cert>=0.05)",
          "mast3r": "MASt3R (conf>=0)", "mast3r_conf05": "MASt3R (conf>=0.5)",
          "mast3r_mc12": "MASt3R (conf>=12, too high)",
          "da3": "Depth Anything 3",
          "cv2_dis_flow": "DIS optical flow", "cv2_sgbm": "StereoSGBM",
          "disk_lightglue": "DISK+LightGlue"}
# Preferred order for tables and figures.
ORDER = ["kornia_loftr", "efficient_loftr", "roma", "dkm", "mast3r",
         "mast3r_conf05", "da3", "cv2_dis_flow", "cv2_sgbm", "disk_lightglue",
         "roma_c05", "dkm_c05", "mast3r_mc12"]


# PURPOSE: Read every result file and index it by the result stem
#          (<backend><tag>_<date>), so a tagged variant is not overwritten by the
#          untagged one.
# INPUTS: none.
# OUTPUTS: dict stem -> parsed JSON.
# KEYWORDS: load, results
def load_all():
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        stem = os.path.basename(path)[:-5]
        if stem == "throughput":
            continue
        with open(path) as fh:
            d = json.load(fh)
        if "backend" not in d or "frames" not in d:
            continue
        out[stem] = d
    return out


# PURPOSE: Read the isolated matcher timings written by benchmark_throughput.py.
# INPUTS: none.
# OUTPUTS: dict backend -> ms_median (or {} when absent).
# KEYWORDS: throughput, load
def load_throughput():
    p = os.path.join(RESULTS, "throughput.json")
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        raw = json.load(fh)
    return {k: v.get("ms_median") for k, v in raw.items() if "ms_median" in v}


def _get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def agg(frames, path, fn=np.median):
    vals = [v for v in (_get(fr, path) for fr in frames)
            if v is not None and np.isfinite(v)]
    return float(fn(vals)) if vals else float("nan")


def by_stamp(frames, path):
    out = {}
    for fr in frames:
        v = _get(fr, path)
        if v is not None and np.isfinite(v):
            out[fr["stamp"]] = float(v)
    return out


# PURPOSE: Two-sided paired sign test on the per-frame differences of a metric
#          between a candidate and the baseline.
# INPUTS: a (dict stamp->candidate), b (baseline).
# OUTPUTS: (n_pairs, median_diff, p_value). p is 1.0 when n < 5.
# KEYWORDS: sign_test, paired, significance
def sign_test(a, b):
    stamps = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in stamps])
    if len(d) < 5:
        return len(d), float(np.median(d)) if len(d) else float("nan"), 1.0
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return len(d), float(np.median(d)), 1.0
    from scipy.stats import binomtest
    return len(d), float(np.median(d)), float(min(1.0, binomtest(min(pos, neg), n, 0.5).pvalue * 2.0))


# PURPOSE: Build the comparison rows for one session, one row per result stem that
#          is present, ordered by the preferred ORDER (tagged variants last).
# INPUTS: data (load_all output); date; throughput dict; baseline stem.
# OUTPUTS: list of row dicts.
# KEYWORDS: table, ranking
def table(data, date, tp, baseline="kornia_loftr"):
    base = data.get(f"{baseline}_{date}")
    stems = [s for s in data if s.endswith("_" + date)]
    stems.sort(key=lambda s: (ORDER.index(s[:-len(date) - 1])
                              if s[:-len(date) - 1] in ORDER else 99, s))
    rows = []
    for stem in stems:
        d = data[stem]
        fr = d["frames"]
        tag = stem[:-len(date) - 1]
        plane = lambda k, x=fr: agg(x, ("plane", k))
        row = {
            "stem": stem, "tag": tag, "backend": d["backend"],
            "label": LABELS.get(tag, tag), "n_frames": len(fr),
            "rms_full_px": plane("rms_full_px"),
            "med_abs_res_px": plane("med_abs_res_px"),
            "rms_trimmed_px": plane("rms_trimmed_px"),
            "med_abs_lowgrad_px": agg(fr, ("plane_lowgrad", "med_abs_res_px")),
            "rms_lowgrad_px": agg(fr, ("plane_lowgrad", "rms_full_px")),
            "med_abs_highgrad_px": agg(fr, ("plane_highgrad", "med_abs_res_px")),
            "n_lowgrad": agg(fr, ("plane_lowgrad", "n")),
            "ray_miss_median_m": plane("ray_miss_median_m"),
            "roll_fit_deg": plane("roll_deg"),
            "dv_rand_rms_px": agg(fr, ("epi", "dv_rand_rms_px")),
            "dv_struct_rms_px": agg(fr, ("epi", "dv_struct_rms_px")),
            "dv_median_px": agg(fr, ("epi", "dv_median_px")),
            "slope_m_per_px": agg(fr, ("radial", "slope_m_per_px")),
            "height_median_m": agg(fr, ("radial", "height_median_m")),
            "height_iqr_m": agg(fr, ("radial", "height_iqr_m")),
            "low_grad_frac": agg(fr, ("grad", "low_grad_frac")),
            "low_grad_density": agg(fr, ("grad", "low_grad_density")),
            "n_matches": float(np.median([f["n_matches"] for f in fr])) if fr else float("nan"),
        }
        row["frac_frames_ge100"] = (float(np.mean([f["n_matches"] >= 100 for f in fr]))
                                    if fr else float("nan"))
        row["ms"] = tp.get(d["backend"], float("nan"))
        tk = [f["topk"] for f in fr if f.get("topk")]
        row["topk_med_abs_px"] = float(np.median([t["plane"]["med_abs_res_px"] for t in tk])) if tk else float("nan")
        row["topk_rms_px"] = float(np.median([t["plane"]["rms_full_px"] for t in tk])) if tk else float("nan")
        pf = d.get("perturb", {}).get("frames", [])
        row["perturb_ddu_rms_px"] = float(np.median([x["ddu_rms_px"] for x in pf
                                                     if x.get("ddu_rms_px") is not None])) if pf else float("nan")
        row["perturb_ddv_rms_px"] = float(np.median([x["ddv_rms_px"] for x in pf
                                                     if x.get("ddv_rms_px") is not None])) if pf else float("nan")
        if base is not None and stem != f"{baseline}_{date}":
            a = by_stamp(fr, ("plane", "rms_full_px"))
            b = by_stamp(base["frames"], ("plane", "rms_full_px"))
            n, med, p = sign_test(a, b)
            row["paired_n"], row["paired_median_diff_px"], row["paired_p"] = n, med, p
        rows.append(row)
    return rows


HDR = (f"{'matcher':<20}{'nf':>4}{'match':>7}{'ms':>7}{'rms':>6}{'medabs':>7}"
       f"{'lg_med':>7}{'lg_n':>6}{'lg%':>6}{'topk':>6}{'dv_rand':>8}{'slope':>7}"
       f"{'hIQR':>6}{'pert':>6}{'p':>6}")


def _print(rows, date):
    print(f"\n===== {date} =====")
    print(HDR)
    print("-" * len(HDR))
    for r in rows:
        print(f"{r['label']:<20}{r['n_frames']:>4}{r['n_matches']:>7.0f}{r['ms']:>7.0f}"
              f"{r['rms_full_px']:>6.2f}{r['med_abs_res_px']:>7.2f}"
              f"{r['med_abs_lowgrad_px']:>7.2f}{r['n_lowgrad']:>6.0f}{r['low_grad_frac']:>6.2f}"
              f"{r['topk_med_abs_px']:>6.2f}{r['dv_rand_rms_px']:>8.2f}{r['slope_m_per_px']:>7.2f}"
              f"{r['height_iqr_m']:>6.0f}{r['perturb_ddu_rms_px']:>6.2f}"
              f"{r.get('paired_p', float('nan')):>6.3f}")


# PURPOSE: Draw the radial height profile and the criteria bars per session.
# INPUTS: data; date.
# OUTPUTS: PNG files in FIGDIR.
# KEYWORDS: figure, radial_profile, bar_chart
def figures(data, date):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGDIR, exist_ok=True)
    stems = sorted([s for s in data if s.endswith("_" + date)],
                   key=lambda s: (ORDER.index(s[:-len(date) - 1])
                                  if s[:-len(date) - 1] in ORDER else 99, s))
    if not stems:
        return []
    made = []

    # 1. radial profiles
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for stem in stems:
        d = data[stem]
        tag = stem[:-len(date) - 1]
        fr = [f for f in d["frames"]
              if len(f["radial"].get("bin_h", [])) > 0
              and len(f["radial"]["bin_h"]) == len(f["radial"]["bin_r"])]
        if not fr:
            continue
        rs = np.asarray(fr[0]["radial"]["bin_r"], float)
        hs = np.array([[b for b in f["radial"]["bin_h"]] for f in fr], float)
        ax.plot(rs, np.nanmedian(hs, axis=0), marker="o", ms=3,
                label=LABELS.get(tag, tag))
    ax.set_xlabel("radius from principal point (px)")
    ax.set_ylabel("median height (m)")
    ax.set_title(f"Radial height profile — median over frames, {date}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = os.path.join(FIGDIR, f"radial_profile_{date}.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 2. criteria bars
    rows = table(data, date, load_throughput())
    keys = [("med_abs_res_px", "plane-fit median |res| (px)"),
            ("med_abs_lowgrad_px", "median |res| in LOW-gradient (px)"),
            ("dv_rand_rms_px", "vertical disparity random rms (px)"),
            ("perturb_ddu_rms_px", "perturbation rms (px)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.1 * len(keys), 4.2))
    for ax, (k, title) in zip(np.atleast_1d(axes), keys):
        labs = [r["label"] for r in rows]
        vals = [r.get(k, float("nan")) for r in rows]
        ax.barh(range(len(labs)), vals, color="#4477aa")
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, axis="x"); ax.invert_yaxis()
    p = os.path.join(FIGDIR, f"criteria_{date}.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # 3. coverage: matches per frame, and low-gradient match fraction
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for stem in stems:
        d = data[stem]; tag = stem[:-len(date) - 1]; fr = d["frames"]
        axes[0].plot(range(len(fr)), [f["n_matches"] for f in fr], marker=".", ms=4,
                     label=LABELS.get(tag, tag))
    axes[0].set_yscale("log"); axes[0].set_xlabel("frame index")
    axes[0].set_ylabel("matches per frame"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7); axes[0].set_title(f"Coverage, {date}")
    labs = [r["label"] for r in rows]; vals = [r["low_grad_frac"] for r in rows]
    axes[1].barh(range(len(labs)), vals, color="#aa7744")
    axes[1].axvline(1 / 3, ls="--", c="k", lw=1, label="uniform expectation (0.33)")
    axes[1].set_yticks(range(len(labs))); axes[1].set_yticklabels(labs, fontsize=8)
    axes[1].invert_yaxis(); axes[1].grid(alpha=0.3, axis="x")
    axes[1].set_title("fraction of matches in lowest-gradient tercile")
    axes[1].legend(fontsize=8)
    p = os.path.join(FIGDIR, f"coverage_{date}.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    return made


def main():
    data = load_all()
    tp = load_throughput()
    summary = {}
    for date in ("2025-05-23", "2025-05-22"):
        rows = table(data, date, tp)
        if not rows:
            print(f"(no results for {date})")
            continue
        _print(rows, date)
        summary[date] = rows
        print("figures:", *figures(data, date), sep="\n  ")
    with open(os.path.join(HERE, "report.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote report.json")


if __name__ == "__main__":
    main()
