# PURPOSE: Compare time-series results from two feature matcher backends (e.g. Kornia LoFTR vs MatchAnything Eloftr).
#          Parses both log sets, runs unified time-shift analysis, and generates before/after comparison plots.
#          Time interval is driven by config.py.
# INPUTS: Ceilometer path, two algorithm log directories, output assets directory, shift parameters.
#         Reads TIME_SERIES_START_TIME and TIME_SERIES_END_TIME from config.py.
# OUTPUTS: Saved PNG plots: temporal/correlation before and after shift, shift profile, console summary.
# KEYWORDS: comparison, loftr, eloftr, time_shift, correlation, before_after, config_driven
import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from tools.unified_time_shift import (
    parse_ceilometer_logs, parse_algorithm_logs, smooth_data,
    find_constant_shift, compute_dynamic_shift,
)


# PURPOSE: Filter parsed time series data to a configured time window.
# INPUTS: data (list of (datetime, ...) tuples), start_time, end_time (datetime.time objects)
# OUTPUTS: Filtered list
# KEYWORDS: filter, time_window, interval, config
def filter_by_time_window(data, start_time, end_time):
    return [
        item for item in data
        if start_time <= item[0].time() <= end_time
    ]


# PURPOSE: Compute raw (unshifted) alignment of algorithm heights to nearest ceilometer readings.
# INPUTS: algo_data, ceilo_data (lists of tuples), tolerance_s (int)
# OUTPUTS: aligned list of (dt, algo_h, ceilo_h) triples, correlation (float)
# KEYWORDS: raw_alignment, unshifted, nearest_neighbor, correlation
def compute_raw_alignment(algo_data, ceilo_data, tolerance_s=10):
    if not algo_data or not ceilo_data:
        return [], 0
    ceilo_ts = np.array([d.timestamp() for d, _h in ceilo_data])
    ceilo_hs = np.array([h for _d, h in ceilo_data])
    aligned = []
    for t, h in algo_data:
        ts = t.timestamp()
        diffs = np.abs(ceilo_ts - ts)
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance_s:
            aligned.append((t, h, ceilo_hs[idx]))
    if len(aligned) < 2:
        return aligned, 0
    _t, ah, ch = zip(*aligned)
    return aligned, np.corrcoef(ch, ah)[0, 1]


# PURPOSE: Run full analysis for one backend, returning before/after shift results.
# INPUTS: ceilo_path, algo_dir, shift params, smoothing, time window from config
# OUTPUTS: dict with ceilo, raw, shifted results
# KEYWORDS: analysis, backend, before_shift, after_shift
def analyze_backend(ceilo_path, algo_dir, shift_range, tolerance,
                    smooth, smooth_window, min_cluster_size, label, window_min=10):
    print(f"\n=== {label} ===")
    ceilo = parse_ceilometer_logs(ceilo_path)
    algo = parse_algorithm_logs(algo_dir, ceilo, min_cluster_size=min_cluster_size)

    if not ceilo or len(algo) < 2:
        print(f"  Insufficient data for {algo_dir}")
        return None

    # Filter by config time window
    ts_start = datetime.datetime.strptime(config.TIME_SERIES_START_TIME, "%H-%M-%S").time()
    ts_end = datetime.datetime.strptime(config.TIME_SERIES_END_TIME, "%H-%M-%S").time()
    ceilo = filter_by_time_window(ceilo, ts_start, ts_end)
    algo = filter_by_time_window(algo, ts_start, ts_end)
    print(f"  After time-window filter: {len(algo)} algo points, {len(ceilo)} ceilo points")
    if len(algo) < 2:
        print("  Insufficient after filtering.")
        return None

    if smooth:
        ceilo = smooth_data(ceilo, smooth_window)
        algo = smooth_data(algo, smooth_window)
    else:
        ceilo_raw_ref = ceilo
        algo_raw_ref = algo

    # Raw (BEFORE SHIFT) alignment
    raw_aligned, raw_corr = compute_raw_alignment(algo, ceilo, tolerance)

    # Dynamic (AFTER SHIFT) alignment — sliding window
    shifts_list, shifted_aligned = compute_dynamic_shift(
        algo, ceilo, window_min, shift_range, tolerance)
    if shifted_aligned:
        _t, ah, ch = zip(*shifted_aligned)
        shifted_corr = np.corrcoef(ch, ah)[0, 1]
    else:
        shifted_corr = 0

    return {
        "ceilo": ceilo,
        "algo_raw": algo,
        "raw_aligned": raw_aligned,
        "raw_corr": raw_corr,
        "best_shift": "dynamic",
        "shifted_corr": shifted_corr,
        "shifted_aligned": shifted_aligned,
        "shifts_list": shifts_list,
        "n_pairs": len(algo),
        "n_raw_aligned": len(raw_aligned),
        "n_shifted_aligned": len(shifted_aligned) if shifted_aligned else 0,
    }


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_temporal(ax, raw_data, shifted_data, ceilo_data, labels, colors, title):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cloud Base Height (m)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    def _times(d): return [x[0] for x in d]
    def _heights(d): return [float(x[1]) for x in d]

    if raw_data:
        for i, (d, lbl) in enumerate(zip(raw_data, labels)):
            if d and len(d) > 0:
                ax.plot(_times(d), _heights(d), ".-", color=colors[i],
                        label=lbl, markersize=2, linewidth=0.5, alpha=0.6)
    if shifted_data:
        for i, (d, lbl) in enumerate(zip(shifted_data, labels)):
            if d and len(d) > 0:
                ax.plot(_times(d), _heights(d), "-", color=colors[i],
                        label=f"{lbl} (shifted)", linewidth=0.8, alpha=0.8)
    if ceilo_data:
        ax.plot(_times(ceilo_data), _heights(ceilo_data), "-", color="red",
                label="Ceilometer", linewidth=1.2, alpha=0.9)

    ax.set_ylim(bottom=0)
    ax.relim()
    ax.autoscale_view()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=9)


def _plot_correlation(ax, aligned_data, corr, label, color, title):
    if not aligned_data:
        ax.set_title(f"{title} (no data)")
        return
    _t, ah, ch = zip(*aligned_data)
    ax.scatter(ch, ah, alpha=0.6, s=10, c=color)
    lim = max(max(ch), max(ah)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", alpha=0.75)
    ax.set_title(f"{title}\nr={corr:.3f}, n={len(aligned_data)}", fontsize=12)
    ax.set_xlabel("Ceilometer (m)")
    ax.set_ylabel("Algorithm (m)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")


def generate_plots(ra, rb, label_a, label_b, assets_dir, mode="constant"):
    os.makedirs(assets_dir, exist_ok=True)
    ts_start = f"{config.TIME_SERIES_START_TIME}-{config.TIME_SERIES_END_TIME}"
    labels = [label_a, label_b]
    colors = ["blue", "green"]

    # --- TEMPORAL BEFORE SHIFT (raw) ---
    fig, ax = plt.subplots(figsize=(16, 8))
    _plot_temporal(ax,
        raw_data=[ra["algo_raw"], rb["algo_raw"]],
        shifted_data=None,
        ceilo_data=ra["ceilo"],
        labels=labels, colors=colors,
        title=f"Raw Algorithm Heights vs Ceilometer ({ts_start})")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "temporal_before.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {assets_dir}/temporal_before.png")

    # --- TEMPORAL AFTER SHIFT (aligned) ---
    fig, ax = plt.subplots(figsize=(16, 8))
    _plot_temporal(ax,
        raw_data=None,
        shifted_data=[ra["shifted_aligned"], rb["shifted_aligned"]],
        ceilo_data=ra["ceilo"],
        labels=labels, colors=colors,
        title=f"Dynamic Shift: Algorithm Heights vs Ceilometer ({ts_start})")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "temporal_after.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {assets_dir}/temporal_after.png")

    # --- CORRELATION BEFORE SHIFT (raw) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    _plot_correlation(ax1, ra["raw_aligned"], ra["raw_corr"], label_a, "blue",
                      f"{label_a} (raw)")
    _plot_correlation(ax2, rb["raw_aligned"], rb["raw_corr"], label_b, "green",
                      f"{label_b} (raw)")
    fig.suptitle(f"Correlation Before Shift ({ts_start})", fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "correlation_before.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {assets_dir}/correlation_before.png")

    # --- CORRELATION AFTER SHIFT ---
    shift_label = "dynamic"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    _plot_correlation(ax1, ra["shifted_aligned"], ra["shifted_corr"], label_a, "blue",
                      f"{label_a} ({shift_label})")
    _plot_correlation(ax2, rb["shifted_aligned"], rb["shifted_corr"], label_b, "green",
                      f"{label_b} ({shift_label})")
    fig.suptitle(f"Correlation After Dynamic Shift ({ts_start})", fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(assets_dir, "correlation_after.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {assets_dir}/correlation_after.png")


def print_summary(ra, rb, label_a, label_b):
    print("\n" + "=" * 75)
    t = f"{config.TIME_SERIES_START_TIME} – {config.TIME_SERIES_END_TIME}"
    print(f"Time Window: {t}")
    print("=" * 75)
    print(f"{'Metric':<35} {label_a:>18} {label_b:>18}")
    print("-" * 75)
    for r, lbl in [(ra, label_a), (rb, label_b)]:
        if r is None:
            print(f"\n{lbl}: FAILED (insufficient data)")
            return
    for metric, ka, kb in [
        ("Algorithm data points", "n_pairs", "n_pairs"),
        ("Aligned points (before shift)", "n_raw_aligned", "n_raw_aligned"),
        ("Correlation (before shift)", "raw_corr", "raw_corr"),
        ("Aligned points (after shift)", "n_shifted_aligned", "n_shifted_aligned"),
        ("Best shift (s)", "best_shift", "best_shift"),
        ("Correlation (after shift)", "shifted_corr", "shifted_corr"),
    ]:
        va = ra[ka] if ka != "best_shift" else str(ra[ka])
        vb = rb[kb] if kb != "best_shift" else str(rb[kb])
        fmt = f"{va:>18.4f}" if isinstance(va, float) else f"{va:>18}"
        fmt2 = f"{vb:>18.4f}" if isinstance(vb, float) else f"{vb:>18}"
        print(f"{metric:<35} {fmt} {fmt2}")
    print("=" * 75)


# --- CONSTANTS ---
CEILO_PATH = "/home/omega-luler/tasks/clouds/data/height data/Zve_Sci_250523.txt"
ALGO_DIR_A = "/home/omega-luler/tasks/clouds/logs/250523/kornia_loftr"
ALGO_DIR_B = "/home/omega-luler/tasks/clouds/logs/250523/matchanything_eloftr"
LABEL_A = "Kornia LoFTR"
LABEL_B = "MatchAnything Eloftr"
ASSETS_DIR = "report_assets/comparison"
SHIFT_RANGE = 300
TIME_TOLERANCE = 10
SMOOTH = True
SMOOTH_WINDOW = 5
MIN_CLUSTER_SIZE = 3
WINDOW_MIN = 10


if __name__ == "__main__":
    print(f"Time window from config: {config.TIME_SERIES_START_TIME} – {config.TIME_SERIES_END_TIME}")

    ra = analyze_backend(CEILO_PATH, ALGO_DIR_A, SHIFT_RANGE,
                         TIME_TOLERANCE, SMOOTH, SMOOTH_WINDOW,
                         MIN_CLUSTER_SIZE, LABEL_A, WINDOW_MIN)
    rb = analyze_backend(CEILO_PATH, ALGO_DIR_B, SHIFT_RANGE,
                         TIME_TOLERANCE, SMOOTH, SMOOTH_WINDOW,
                         MIN_CLUSTER_SIZE, LABEL_B, WINDOW_MIN)

    if ra is None or rb is None:
        print("\nERROR: One or both backends failed.")
        sys.exit(1)

    print_summary(ra, rb, LABEL_A, LABEL_B)
    print("\n--- Generating Comparison Plots ---")
    generate_plots(ra, rb, LABEL_A, LABEL_B, ASSETS_DIR)
    print(f"\nDone. Assets saved to {ASSETS_DIR}/")
