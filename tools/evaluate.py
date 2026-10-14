# PURPOSE: Command-line entry point for the honest cloud-base-height evaluation:
#          stereo pairs through LoFTR and HDBSCAN, a ground-truth-free cluster
#          selection rule and the Kalman temporal filter,
#          scored against the DOL-2 ceilometer. Every run writes a provenance
#          manifest, so a number produced here can always be traced to the code
#          revision, configuration hash, calibration identity and dataset it came
#          from.
# INPUTS: --config (required), --limit, --stride, --strategy,
#         --dump-clusters, --shift-scan, --out.
# OUTPUTS: a run directory containing run_manifest.json, run.log, series.json,
#          metrics.json and (optionally) clusters.jsonl; a summary on stdout.
# KEYWORDS: cli, evaluate, cloud_base_height, kalman, provenance
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The scientific stack emits per-call deprecation noise (kornia's
# image_to_tensor, sklearn's HDBSCAN copy default) that would bury the structured
# run log. Behaviour is unaffected; silence it explicitly.
warnings.filterwarnings("ignore")

from clouds.config import ConfigError, describe, load_config  # noqa: E402
from clouds.pipeline.evaluate import STRATEGIES, evaluate, write_artefacts  # noqa: E402
from clouds.provenance import Run, start_run  # noqa: E402


# PURPOSE: Print the headline metrics of an evaluation in a fixed, greppable form.
# INPUTS: result (dict from evaluate).
# OUTPUTS: None (writes to stdout).
# KEYWORDS: report, summary, metrics
def print_summary(result: dict) -> None:
    m = result["metrics"]
    sm = m["smoothing"]
    print("\n" + "=" * 74)
    print("  CLOUD BASE HEIGHT EVALUATION")
    print("=" * 74)
    print(f"  frames with clusters : {result['stats']['n_frames_with_clusters']}"
          f" / {result['stats']['n_frames_processed']}"
          f"  (no result: {result['stats']['n_no_result']})")
    print(f"  ceilometer points    : {m['n_ceilo_points']}")
    print("-" * 74)
    print(f"  frame-to-frame jump  : raw {sm['raw']['mean_dh']:.1f} m"
          f" (median {sm['raw']['median_dh']:.1f})"
          f"  ->  filtered {sm['filtered']['mean_dh']:.1f} m"
          f" (median {sm['filtered']['median_dh']:.1f})")
    print(f"  autocorrelation      : raw {sm['raw_autocorr']:.3f}"
          f"  ->  filtered {sm['filtered_autocorr']:.3f}")
    for name in ("raw", "filtered"):
        w = m[f"{name}_windows"]
        print(f"  {name:8s} windows    : mean r {w['mean_r']:.3f},"
              f" >0.5 in {w['frac_gt_half']:.2f} of {w['n_scored']}/{w['n_windows']},"
              f" max r {w['max_r']:.3f}")
        print(f"  {name:8s} global r   : {m[f'{name}_global_r']} (n={m[f'{name}_global_n']})")
    if "shift_scan" in m:
        s = m["shift_scan"]
        print(f"  shift scan           : best {s.get('best_shift_h')} h"
              f" (r={s.get('best_r'):.3f}, n={s.get('best_n')})"
              f"   r@0h={s.get('r_at_0h')}  r@+3h={s.get('r_at_plus3h')}")
    print("=" * 74)


# PURPOSE: CLI driver.
# INPUTS: argv (optional).
# OUTPUTS: process exit code: 0 on success, 1 on configuration error.
# KEYWORDS: cli, main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate stereo cloud-base height against the DOL-2 ceilometer.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pairs")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth pair")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default=None,
                    help="cluster selection rule; defaults to the config value")
    ap.add_argument("--dump-clusters", action="store_true",
                    help="write every per-frame cluster to clusters.jsonl")
    ap.add_argument("--shift-scan", action="store_true",
                    help="also scan a wide global time-shift range")
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
        run = Run(run_dir=out, entry_point="tools/evaluate.py", argv=sys.argv)
        run.open(repo_root=REPO_ROOT, config=cfg)
    else:
        run = start_run(base=cfg.output.logs_dir,
                        label=f"evaluate_{cfg.project.name}_{cfg.dataset.date}",
                        entry_point="tools/evaluate.py", argv=sys.argv,
                        repo_root=REPO_ROOT, config=cfg)
    run.add_input("ceilometer_log", cfg.dataset.ceilometer_log)

    overrides = {}
    for key in ("limit", "stride", "strategy", "dump_clusters",
                "shift_scan"):
        val = getattr(args, key)
        if val not in (0, 1, None, False):
            overrides[key] = val
    if overrides:
        run.record("overrides", **overrides)
        run.event("overrides", **overrides)

    result = evaluate(cfg, run, limit=args.limit, stride=args.stride,
                      dump_clusters=args.dump_clusters, shift_scan=args.shift_scan,
                                    strategy=args.strategy)
    written = write_artefacts(run, result, write_clusters=args.dump_clusters)
    print_summary(result)
    print("artefacts: " + ", ".join(str(p) for p in written))
    print(f"manifest : {run.close(status='ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
