# PURPOSE: Compare the two calibrated affine matrices on the same session and data,
#          so the choice of calibration is an explicit, reproducible experiment
#          rather than an edit to a shared constant. Each variant runs the honest
#          evaluation against the same ceilometer, and the windowed agreement of
#          the raw and Kalman-filtered series is reported side by side.
#          This supersedes the previous version, which computed its correlations
#          through a cluster-selection rule that consulted the ceilometer (the
#          shipped parse_algorithm_logs, deleted in ADR 0007). That rule is
#          ground-truth leakage, not a measurement; see docs/repro/dlcp2026.md.
# INPUTS: --config (a base YAML config) and --calibrations (ids from
#         configs/calib/). The base config supplies the dataset, window and
#         pipeline parameters; only the calibration identity varies per variant.
# OUTPUTS: per-variant run directories each with their own manifest, plus
#         <logs>/calibration_comparison.json and a table on stdout.
# KEYWORDS: comparison, calibration, affine, ablation, provenance
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clouds.config import ConfigError, load_calibration, load_config  # noqa: E402
from clouds.pipeline.evaluate import evaluate, write_artefacts  # noqa: E402
from clouds.provenance import start_run  # noqa: E402

DEFAULT_CALIBRATIONS = ["2025-05-22_affine_old", "2025-05-22_affine_new"]


# PURPOSE: Return a copy of a configuration bound to a different calibration
#          registry entry. Config is frozen, so this yields a new object instead
#          of mutating shared state.
# INPUTS: cfg (Config), calib_id (str).
# OUTPUTS: Config with camera.calibration and calibration_record replaced.
# KEYWORDS: config, calibration, immutability
def with_calibration(cfg, calib_id: str):
    record = load_calibration(Path(cfg.calibration_dir), calib_id)
    return dataclasses.replace(
        cfg,
        camera=dataclasses.replace(cfg.camera, calibration=calib_id),
        calibration_record=record,
    )


# PURPOSE: CLI driver.
# INPUTS: argv (optional).
# OUTPUTS: exit code; 1 on configuration error.
# KEYWORDS: cli, main, comparison
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare calibrations on one session.")
    ap.add_argument("--config", required=True, help="base YAML config")
    ap.add_argument("--calibrations", nargs="*", default=DEFAULT_CALIBRATIONS,
                    help="calibration registry ids to compare")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)

    try:
        base_cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    results = []
    for calib_id in args.calibrations:
        cfg = with_calibration(base_cfg, calib_id)
        run = start_run(base=cfg.output.logs_dir,
                        label=f"calibration_{calib_id}_{cfg.dataset.date}",
                        entry_point="run_comparison.py", argv=sys.argv,
                        repo_root=REPO_ROOT, config=cfg)
        run.add_input("ceilometer_log", cfg.dataset.ceilometer_log)
        result = evaluate(cfg, run, limit=args.limit, stride=args.stride)
        write_artefacts(run, result)
        metrics = result["metrics"]
        m = cfg.calibration_record.matrix_2x3
        seen = result["stats"]["clusters_seen"]
        summary = {
            "calibration": calib_id,
            "translation_px": [m[0][2], m[1][2]],
            "n_frames_with_clusters": result["stats"]["n_frames_with_clusters"],
            "raw_windows": metrics["raw_windows"],
            "filtered_windows": metrics["filtered_windows"],
            "smoothing": metrics["smoothing"],
            "run_dir": str(run.run_dir),
        }
        run.record("results", comparison=summary)
        run.close(status="ok")
        results.append(summary)

    out_path = Path(base_cfg.output.logs_dir) / "calibration_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, default=str))

    print("\n" + "=" * 92)
    print(f"  {'calibration':28s} {'raw mean r':>10s} {'filt mean r':>12s} "
          f"{'raw max r':>10s} {'filt max r':>11s}")
    print("=" * 92)
    for r in results:
        print(f"  {r['calibration']:28s} {r['raw_windows']['mean_r']:>10.3f} "
              f"{r['filtered_windows']['mean_r']:>12.3f} "
              f"{r['raw_windows']['max_r']:>10.3f} "
              f"{r['filtered_windows']['max_r']:>11.3f}")

    print("=" * 92)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
