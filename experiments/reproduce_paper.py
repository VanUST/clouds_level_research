# PURPOSE: Score the honest evaluation against the quantitative claims of the
#          DLCP2026 submission, so each published number is either reproduced
#          from a hashed configuration or explicitly reported as not reproduced.
#          This script is a *reporter*: the pipeline, the temporal filter and all
#          metrics live in src/clouds/ and are shared with tools/evaluate.py, so
#          there is exactly one implementation of the honest path.
# INPUTS: --config (required) and the evaluation options accepted by
#         clouds.pipeline.evaluate.
# OUTPUTS: a run directory containing run_manifest.json, run.log, series.json,
#          metrics.json and verdicts.json, plus a claim-by-claim table on stdout.
# KEYWORDS: reproduction, claims, dlcp2026, evaluation, provenance
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clouds.config import ConfigError, describe, load_config  # noqa: E402
from clouds.pipeline.evaluate import STRATEGIES, evaluate, write_artefacts  # noqa: E402
from clouds.provenance import Run, start_run  # noqa: E402

# Every quantitative claim the written work makes, in the units it prints:
#   claim id -> (value, human label, unit)
# The values are transcribed from submission/37-Ustimov/tex/Ustimov.tex
# (Table I and the abstract); they are the target, not an input to the pipeline.
CLAIMS = {
    "raw_jump": (515.0, "Mean frame-to-frame |dh|, raw (m)", "m"),
    "filtered_jump": (120.0, "Mean frame-to-frame |dh|, Kalman-filtered (m)", "m"),
    "raw_autocorr": (0.11, "5-hour autocorrelation, raw", ""),
    "filtered_autocorr": (0.90, "5-hour autocorrelation, Kalman-filtered", ""),
    "raw_mean_r": (0.55, "Mean Pearson r in 30-min windows, raw", ""),
    "filtered_mean_r": (0.67, "Mean Pearson r in 30-min windows, filtered", ""),
    "raw_frac_gt_half": (0.61, "Fraction of windows with r > 0.5, raw", ""),
    "filtered_frac_gt_half": (0.72, "Fraction of windows with r > 0.5, filtered", ""),
    "raw_max_r": (0.88, "Maximum r in a single window, raw", ""),
    "filtered_max_r": (0.94, "Maximum r in a single window, filtered", ""),
    "global_raw_r": (0.04, "Global 5-hour raw correlation, no shift", ""),
    # The written work quotes 15-25%; 0.20 is the midpoint used for comparison.
    "spatial_reject_frac": (0.20, "Spatial filter rejection rate (paper band 15-25%)", ""),
}


# PURPOSE: Map an evaluated metric onto the claim it answers.
# INPUTS: key (claim id), metrics (dict from clouds.pipeline.evaluate).
# OUTPUTS: the measured value, or None when the metric is undefined for this run
#          (e.g. too few frames for a windowed correlation).
# KEYWORDS: mapping, claims, metrics
def measured_value(key: str, metrics: dict):
    sm = metrics.get("smoothing", {})
    mapping = {
        "raw_jump": sm.get("raw", {}).get("mean_dh"),
        "filtered_jump": sm.get("filtered", {}).get("mean_dh"),
        "raw_autocorr": sm.get("raw_autocorr"),
        "filtered_autocorr": sm.get("filtered_autocorr"),
        "raw_mean_r": metrics.get("raw_windows", {}).get("mean_r"),
        "filtered_mean_r": metrics.get("filtered_windows", {}).get("mean_r"),
        "raw_frac_gt_half": metrics.get("raw_windows", {}).get("frac_gt_half"),
        "filtered_frac_gt_half": metrics.get("filtered_windows", {}).get("frac_gt_half"),
        "raw_max_r": metrics.get("raw_windows", {}).get("max_r"),
        "filtered_max_r": metrics.get("filtered_windows", {}).get("max_r"),
        "global_raw_r": metrics.get("raw_global_r"),
    }
    if key == "spatial_reject_frac":
        seen = metrics.get("_spatial_seen")
        rej = metrics.get("_spatial_rejected")
        return (rej / seen) if (seen and rej is not None) else None
    return mapping.get(key)


# PURPOSE: Print and return the claim-by-claim verdict.
# INPUTS: metrics (dict from evaluate), stats (dict from evaluate).
# OUTPUTS: dict of verdicts (also written to verdicts.json by the caller).
# KEYWORDS: verdict, report, comparison
def report_claims(metrics: dict, stats: dict) -> dict:
    metrics = dict(metrics)
    metrics["_spatial_rejected"] = stats.get("spatial_rejections")
    metrics["_spatial_seen"] = stats.get("spatial_clusters_seen")

    print("\n" + "=" * 88)
    print(f"  {'claim':24s} {'paper':>10s} {'measured':>10s} {'diff':>10s}  note")
    print("=" * 88)
    verdicts = {}
    for key, (paper_v, label, unit) in CLAIMS.items():
        m = measured_value(key, metrics)
        if m is None or (isinstance(m, float) and m != m):  # None or NaN
            print(f"  {key:24s} {paper_v:>10.3f} {'n/a':>10s} {'-':>10s}  {label}")
            verdicts[key] = {"paper": paper_v, "measured": None, "holds": None,
                             "label": label, "unit": unit}
            continue
        diff = float(m) - paper_v
        rel = diff / paper_v if paper_v else float("nan")
        print(f"  {key:24s} {paper_v:>10.3f} {float(m):>10.3f} {diff:>+10.3f}  {label}")
        verdicts[key] = {"paper": paper_v, "measured": float(m),
                         "abs_diff": float(diff), "rel_diff": float(rel),
                         "label": label, "unit": unit}
    print("=" * 88)
    return verdicts


# PURPOSE: CLI driver.
# INPUTS: argv (optional).
# OUTPUTS: process exit code; 1 on configuration error.
# KEYWORDS: cli, main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the evaluation against the DLCP2026 claims.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pairs")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth pair")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default=None)
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--dump-clusters", action="store_true")
    ap.add_argument("--shift-scan", action="store_true")
    ap.add_argument("--out", default=None, help="explicit run directory")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1
    print(describe(cfg))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        run = Run(run_dir=out, entry_point="experiments/reproduce_paper.py", argv=sys.argv)
        run.open(repo_root=REPO_ROOT, config=cfg)
    else:
        run = start_run(base=cfg.output.logs_dir,
                        label=f"reproduce_{cfg.project.name}_{cfg.dataset.date}",
                        entry_point="experiments/reproduce_paper.py", argv=sys.argv,
                        repo_root=REPO_ROOT, config=cfg)
    run.add_input("ceilometer_log", cfg.dataset.ceilometer_log)

    overrides = {k: getattr(args, k) for k in
                 ("limit", "stride", "strategy", "dump_clusters", "shift_scan")
                 if getattr(args, k) not in (0, 1, None, False)}
    if args.no_spatial:
        overrides["no_spatial"] = True
    if overrides:
        run.record("overrides", **overrides)

    result = evaluate(cfg, run, limit=args.limit, stride=args.stride,
                      dump_clusters=args.dump_clusters, shift_scan=args.shift_scan,
                      spatial_override=(False if args.no_spatial else None),
                      strategy=args.strategy)
    written = write_artefacts(run, result, write_clusters=args.dump_clusters)

    verdicts = report_claims(result["metrics"], result["stats"])
    (run.run_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    print("artefacts: " + ", ".join(str(p) for p in written))
    run.record("results", verdicts=verdicts)
    print(f"manifest : {run.close(status='ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
