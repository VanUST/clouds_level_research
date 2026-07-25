# PURPOSE: Orchestrate full comparison pipeline between old and new affine calibration matrices.
#          Runs time-series processing for each affine variant, then unified time-shift analysis
#          on both log sets. Generates per-variant plots plus a comparison overlay.
# INPUTS: config.py (paths, affine matrices, UTS config).
# OUTPUTS: Algorithm logs in LOG_DIR/{variant}/, report assets in assets_root/{variant}/,
#           and a comparison overlay scatter plot.
# KEYWORDS: comparison, affine, orchestration, batch, unified_time_shift
import os
import sys
import numpy as np
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.camera import StereoCameraSystem
from src.pipeline import PipelineRunner
from tools.unified_time_shift import (
    parse_ceilometer_logs,
    parse_algorithm_logs,
    filter_by_time_window,
    intersect_time_ranges,
    count_unmatched_algo_points,
    _align_at_shift,
    plot_correlation,
)

VARIANTS = [
    {"name": "old_affine", "matrix": config.AFFINE_OLD, "label": "Old Affine"},
    {"name": "new_affine", "matrix": config.AFFINE_NEW, "label": "New Affine"},
]


def run_processing(variant):
    """Run time-series processing for one affine variant."""
    name = variant["name"]
    matrix = variant["matrix"]

    print("\n" + "=" * 70)
    print(f"  PROCESSING: {variant['label']} ({name})")
    print("=" * 70)

    stereo_system = StereoCameraSystem(
        base=config.STEREO_BASE,
        angle_of_view=config.ANGLE_OF_VIEW,
        image_width=config.IMAGE_WIDTH,
        model=config.CAMERA_MODEL,
    )

    runner = PipelineRunner(stereo_system, config)
    runner.process_time_series(backend=name, affine_matrix=matrix)


def run_uts_analysis(variant):
    """Run unified time-shift analysis for one affine variant's log directory."""
    name = variant["name"]
    label = variant["label"]

    print("\n" + "=" * 70)
    print(f"  UTS ANALYSIS: {label} ({name})")
    print("=" * 70)

    uts_cfg = config.UTS_CONFIG
    algo_dir = os.path.join(config.LOG_DIR, name)
    assets_dir = os.path.join(uts_cfg["assets_root"], name)

    ceilo_path = getattr(config, "CEILOMETER_LOG_PATH", uts_cfg.get("ceilo_path", ""))
    if not ceilo_path or not os.path.exists(ceilo_path):
        print(f"[ERROR] Ceilometer log not found: {ceilo_path}")
        return

    if not os.path.isdir(algo_dir):
        print(f"[ERROR] Algorithm log directory not found: {algo_dir}")
        return

    ceilo = parse_ceilometer_logs(ceilo_path)
    algo = parse_algorithm_logs(algo_dir, ceilo, 3, 5)

    if not ceilo or len(algo) < 2:
        print(f"[SKIP] Insufficient data for UTS analysis.")
        return

    # Time-window filter
    t_start = uts_cfg.get("analysis_time_start", None)
    t_end = uts_cfg.get("analysis_time_end", None)
    if t_start and t_end:
        ceilo = filter_by_time_window(ceilo, t_start, t_end, "ceilometer")
        algo = filter_by_time_window(algo, t_start, t_end, "algorithm")
        if not ceilo or not algo:
            print(f"[SKIP] No data after time-window filter.")
            return

    ceilo, algo = intersect_time_ranges(ceilo, algo)
    if not ceilo or not algo:
        print(f"[SKIP] No overlapping time range.")
        return

    tolerance = uts_cfg.get("time_tolerance", 10)
    count_unmatched_algo_points(algo, ceilo, tolerance)

    os.makedirs(assets_dir, exist_ok=True)

    # Import and run the full pipeline
    from tools.unified_time_shift import run_pipeline as uts_run
    pipe_cfg = {
        "ceilo_path": ceilo_path,
        "algo_dir": algo_dir,
        "assets_dir": assets_dir,
        "mode": uts_cfg.get("mode", "dynamic"),
        "shift_range": uts_cfg.get("shift_range", 300),
        "time_tolerance": tolerance,
        "smooth": uts_cfg.get("smooth", True),
        "smooth_window": uts_cfg.get("smooth_window", 5),
        "window_min": uts_cfg.get("window_min", 5),
        "distance_m": uts_cfg.get("distance_m", 200.0),
        "bias_correction": uts_cfg.get("bias_correction", False),
        "dtw": uts_cfg.get("dtw", False),
        "dtw_window": uts_cfg.get("dtw_window", 20),
        "analysis_time_start": t_start,
        "analysis_time_end": t_end,
        "affine_label": label,
    }
    uts_run(pipe_cfg)

    return {
        "label": label,
        "assets_dir": assets_dir,
        "algo": algo,
        "ceilo": ceilo,
    }


# PURPOSE: Create a comparison overlay scatter plot showing both affine variants
#          on the same axes against the ceilometer ground truth.
# INPUTS: results (list of dicts from run_uts_analysis), tolerance_s (int)
# OUTPUTS: Saved PNG at assets_root/comparison_scatter.png
# KEYWORDS: comparison, overlay, scatter, affine, side_by_side
def plot_comparison_scatter(results, tolerance_s, save_path):
    if len(results) < 2:
        print(f"[SKIP] Need at least 2 results for comparison scatter.")
        return

    colors = ["blue", "orange"]
    markers = ["o", "^"]

    fig, ax = plt.subplots(figsize=(10, 10))

    global_ylim = 0
    for i, res in enumerate(results):
        ceilo = res["ceilo"]
        algo = res["algo"]
        label = res["label"]
        if not algo or not ceilo:
            continue

        ceilo_ts = np.array([d.timestamp() for d, _h in ceilo])
        ceilo_hs = np.array([h for _d, h in ceilo])
        algo_ts = np.array([t.timestamp() for t, _h in algo])
        algo_hs = np.array([h for _t, h in algo])

        from tools.unified_time_shift import _nearest_indices
        ceilo_idx = _nearest_indices(algo_ts, ceilo_ts)
        valid_mask = np.abs(ceilo_ts[ceilo_idx] - algo_ts) <= tolerance_s
        paired_algo_h = algo_hs[valid_mask]
        paired_ceilo_h = ceilo_hs[ceilo_idx[valid_mask]]

        if len(paired_algo_h) < 2:
            print(f"[SKIP] {label}: too few paired points ({len(paired_algo_h)}) for comparison scatter.")
            continue

        r = np.corrcoef(paired_ceilo_h, paired_algo_h)[0, 1]
        ax.scatter(paired_ceilo_h, paired_algo_h, alpha=0.5, s=18,
                   color=colors[i], marker=markers[i],
                   label=f"{label} (n={len(paired_algo_h)}, r={r:.3f})")
        global_ylim = max(global_ylim, np.max(paired_ceilo_h), np.max(paired_algo_h))

    if global_ylim == 0:
        print(f"[SKIP] No data to plot comparison scatter.")
        return

    lim = global_ylim * 1.05
    ymin = 0
    ax.plot([ymin, lim], [ymin, lim], "r--", linewidth=1.5, label="y = x")
    ax.set_title("Old vs New Affine: All-Points Correlation Comparison",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Ceilometer Height (m)", fontsize=12)
    ax.set_ylabel("Algorithm Height (m)", fontsize=12)
    ax.set_xlim(ymin, lim)
    ax.set_ylim(ymin, lim)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] Comparison scatter: {save_path}")
    plt.close(fig)


# PURPOSE: Generate a summary text report of the comparison.
# INPUTS: results (list of dicts)
# OUTPUTS: Side effect — writes summary.txt to assets_root.
# KEYWORDS: summary, report, comparison, text
def write_summary(results, assets_root):
    summary_path = os.path.join(assets_root, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Affine Matrix Comparison Summary\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'='*50}\n\n")
        for res in results:
            f.write(f"Variant: {res['label']}\n")
            f.write(f"  Assets: {res['assets_dir']}\n")
            f.write(f"  Ceilometer points: {len(res['ceilo'])}\n")
            f.write(f"  Algorithm points:  {len(res['algo'])}\n\n")
    print(f"[SAVED] Summary report: {summary_path}")


# =================================================================================
# Main
# =================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  AFFINE MATRIX COMPARISON PIPELINE")
    print(f"  Image dir:    {config.IMAGE_DIR}")
    print(f"  Log root:     {config.LOG_DIR}")
    print(f"  Assets root:  {config.UTS_CONFIG['assets_root']}")
    print(f"  Time window:  {config.TIME_SERIES_START_TIME} -- {config.TIME_SERIES_END_TIME}")
    print("=" * 70)

    # --- Phase 1: Run processing for each variant ---
    print("\n" + "#" * 70)
    print("# PHASE 1: TIME-SERIES PROCESSING")
    print("#" * 70)
    for variant in VARIANTS:
        run_processing(variant)

    # --- Phase 2: Run UTS analysis for each variant ---
    print("\n" + "#" * 70)
    print("# PHASE 2: UNIFIED TIME-SHIFT ANALYSIS")
    print("#" * 70)
    all_results = []
    for variant in VARIANTS:
        result = run_uts_analysis(variant)
        if result:
            all_results.append(result)

    # --- Phase 3: Comparison plots ---
    print("\n" + "#" * 70)
    print("# PHASE 3: COMPARISON VISUALIZATION")
    print("#" * 70)
    assets_root = config.UTS_CONFIG["assets_root"]
    tolerance = config.UTS_CONFIG.get("time_tolerance", 10)
    plot_comparison_scatter(
        all_results, tolerance,
        os.path.join(assets_root, "comparison_scatter.png"))

    write_summary(all_results, assets_root)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("  Output: " + assets_root)
    print("=" * 70)
