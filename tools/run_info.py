# PURPOSE: Command-line entry point that reports everything a run would record —
#          the fully resolved configuration, the calibration identity together
#          with when and how it was calibrated, and the identity of the dataset
#          it would read — without loading a model or processing any image. This
#          is the check that answers "what data, what time, what calibration"
#          before committing compute to a run.
# INPUTS: --config PATH (required), --emit-run (also create a run directory and
#         write run_manifest.json), --json (machine-readable output).
# OUTPUTS: a human-readable or JSON report on stdout; exit code 1 on any config
#          or calibration error so it can gate a pipeline.
# KEYWORDS: cli, config, provenance, calibration, dataset, dry_run
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clouds.config import ConfigError, describe, load_config  # noqa: E402
from clouds.provenance import file_sha256, start_run  # noqa: E402


# PURPOSE: Assemble the report describing what a run would do and with what.
# INPUTS: cfg (clouds.config.Config).
# OUTPUTS: dict combining the resolved config, the calibration record and the
#          dataset identity (including content hashes where the files exist).
# KEYWORDS: report, provenance, dataset, calibration
def build_report(cfg) -> dict:
    cal = cfg.calibration_record
    return {
        "config": {"path": cfg.source_path, "sha256": cfg.sha256},
        "project": {"name": cfg.project.name, "purpose": cfg.project.purpose},
        "dataset": {
            "date": cfg.dataset.date,
            "images_dir": cfg.dataset.images_dir,
            "images_dir_exists": Path(cfg.dataset.images_dir).is_dir(),
            "window_start": cfg.dataset.window_start,
            "window_end": cfg.dataset.window_end,
            "image_clock": cfg.dataset.image_clock,
            "ceilometer_clock": cfg.dataset.ceilometer_clock,
            "clock_offset_h": cfg.dataset.clock_offset_h,
            "ceilometer_log": cfg.dataset.ceilometer_log,
            "ceilometer_log_exists": Path(cfg.dataset.ceilometer_log).is_file(),
            "ceilometer_log_sha256": file_sha256(cfg.dataset.ceilometer_log),
        },
        "camera": {
            "model": cfg.camera.model,
            "stereo_base_m": cfg.camera.stereo_base_m,
            "angle_of_view_deg": cfg.camera.angle_of_view_deg,
            "image_width_px": cfg.camera.image_width_px,
            "original_size": list(cfg.camera.original_size),
            "target_size": list(cfg.camera.target_size),
        },
        "calibration": {
            "id": cal.id,
            "status": cal.status,
            "matrix_2x3": [list(r) for r in cal.matrix_2x3],
            "calibrated_on": cal.calibrated_on,
            "calibrated_on_source": cal.calibrated_on_source,
            "calibrated_on_verified": cal.calibrated_on_verified,
            "method": cal.method,
            "applies_to_sessions": list(cal.applies_to_sessions),
            "applies_to_sessions_verified": cal.applies_to_sessions_verified,
            "applies_to_this_session": cal.applies_to(cfg.dataset.date),
            "source_artifact": cal.source_artifact,
            "notes": list(cal.notes),
        },
        "pipeline": {
            "matcher": cfg.matching.backend,
            "match_threshold": cfg.matching.match_threshold,
            "temporal_mode": cfg.temporal.mode,
            "cbh_strategy": cfg.temporal.cbh_strategy,
            "measurement_noise_scale": cfg.temporal.measurement_noise_scale,
            "sigma_h_accel": cfg.temporal.sigma_h_accel,
            "window_min": cfg.metrics.window_min,
            "hop_min": cfg.metrics.hop_min,
        },
    }


# PURPOSE: CLI driver.
# INPUTS: argv (optional; defaults to sys.argv[1:]).
# OUTPUTS: process exit code.
# KEYWORDS: cli, main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Describe a clouds run without executing it.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--emit-run", action="store_true",
                    help="create a run directory and write run_manifest.json")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    report = build_report(cfg)

    if args.emit_run:
        run = start_run(
            base=cfg.output.logs_dir,
            label=f"run_info_{cfg.project.name}",
            entry_point="tools/run_info.py",
            argv=sys.argv,
            repo_root=REPO_ROOT,
            config=cfg,
        )
        run.add_input("ceilometer_log", cfg.dataset.ceilometer_log)
        run.record("counts", images_dir_exists=report["dataset"]["images_dir_exists"])
        run.record("results", dry_run=True, purpose=cfg.project.purpose)
        manifest_path = run.close(status="dry_run")
        report["manifest_path"] = str(manifest_path)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(describe(cfg))
        print("-" * 72)
        d = report["dataset"]
        c = report["calibration"]
        print(f"images_dir exists        : {d['images_dir_exists']}")
        print(f"ceilometer log exists    : {d['ceilometer_log_exists']}")
        print(f"ceilometer sha256        : {d['ceilometer_log_sha256']}")
        print(f"calibration method       : {c['method']}")
        print(f"calibrated on            : {c['calibrated_on']} "
              f"({c['calibrated_on_source']}, verified={c['calibrated_on_verified']})")
        print(f"applies to this session  : {c['applies_to_this_session']}")
        if args.emit_run:
            print(f"manifest                 : {report['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
