# PURPOSE: Single dispatcher for every task the project can run. Which task runs
#          is a configuration value (processing.task), not a line a human edits
#          before each run, and the whole invocation is recorded in a provenance
#          manifest so a task's inputs can always be recovered.
# INPUTS: --config PATH (required), --task NAME (overrides processing.task),
#         plus the evaluation options forwarded to tools/evaluate.py.
# OUTPUTS: the artefacts and run manifest of the selected task; exit code 1 on a
#          configuration error or an unknown task.
# KEYWORDS: cli, dispatcher, pipeline, tasks, provenance
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clouds.config import ConfigError, describe, load_config  # noqa: E402
from clouds.geometry.camera import CameraModel, StereoCameraSystem  # noqa: E402
from clouds.pipeline import tasks  # noqa: E402
from clouds.pipeline.analyze import StereoAnalyzer  # noqa: E402

# Task name -> callable(cfg, stereo, analyzer). Each task takes what it needs
# explicitly; there is no shared runner object to reach into.
RUNNER_TASKS = {
    "process_directory": lambda cfg, stereo, an: tasks.process_directory(cfg, stereo, an),
    "process_time_series": lambda cfg, stereo, an: tasks.process_time_series(cfg, stereo, an),
    "ablate_homography": lambda cfg, stereo, an: tasks.run_homography_ablation(cfg, stereo, an),
    "trail": lambda cfg, stereo, an: tasks.analyze_trail(cfg, stereo),
    "iterative_refinement": lambda cfg, stereo, an: tasks.run_iterative_refinement(),
}


# PURPOSE: Build the stereo system and analyzer for a configuration.
# INPUTS: cfg (clouds.config.Config).
# OUTPUTS: (StereoCameraSystem, StereoAnalyzer).
# KEYWORDS: factory, camera, analyzer
def build_context(cfg):
    stereo = StereoCameraSystem(
        base=cfg.camera.stereo_base_m,
        angle_of_view=cfg.camera.angle_of_view_deg,
        image_width=cfg.camera.image_width_px,
        model=CameraModel(cfg.camera.model),
    )
    return stereo, StereoAnalyzer(cfg, stereo)


# PURPOSE: CLI driver.
# INPUTS: argv (optional).
# OUTPUTS: process exit code.
# KEYWORDS: cli, main, dispatch
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a clouds processing task.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--task", default=None,
                    help="override processing.task; one of "
                         f"{sorted(list(RUNNER_TASKS) + ['evaluate'])}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0, help="0 = use the config value")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--dump-clusters", action="store_true")
    ap.add_argument("--shift-scan", action="store_true")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    task = args.task or cfg.processing.task
    print(describe(cfg))
    print(f"task            : {task}")

    if task == "evaluate":
        # Delegate to the evaluate CLI so there is exactly one implementation of
        # the honest evaluation path and one place that writes the manifest.
        from tools.evaluate import main as evaluate_main
        forwarded = ["--config", args.config]
        if args.limit:
            forwarded += ["--limit", str(args.limit)]
        if args.stride:
            forwarded += ["--stride", str(args.stride)]
        if args.strategy:
            forwarded += ["--strategy", args.strategy]
        if args.no_spatial:
            forwarded.append("--no-spatial")
        if args.dump_clusters:
            forwarded.append("--dump-clusters")
        if args.shift_scan:
            forwarded.append("--shift-scan")
        return evaluate_main(forwarded)

    if task not in RUNNER_TASKS:
        print(f"Error: unknown task '{task}'. Available: "
              f"{sorted(list(RUNNER_TASKS) + ['evaluate'])}", file=sys.stderr)
        return 1

    stereo, analyzer = build_context(cfg)
    RUNNER_TASKS[task](cfg, stereo, analyzer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
