# PURPOSE: Unified time-shift analysis pipeline comparing ceilometer ground truth with algorithm-derived cloud heights.
#          Supports both constant (global) and dynamic (sliding-window) shift modes, computes temporal plots,
#          correlation plots, DTW alignment, bias correction, and wind-direction proxy estimation.
# INPUTS: Ceilometer log file, algorithm log directory, output assets directory, and CLI-tunable parameters.
# OUTPUTS: Saved PNG plots (temporal, correlation, error, DTW alignment, bias model, shift profile) and console logs.
# KEYWORDS: time_shift, ceilometer, correlation, dtw, bias_correction, wind_proxy, plot
import os
import sys
import re
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# PURPOSE: Extract (timestamp, height) pairs from a ceilometer text log.
# INPUTS: filepath (str)
# OUTPUTS: Sorted list of (datetime, int) tuples. Returns [] if file missing.
# KEYWORDS: ceilometer, parse, log, height
def parse_ceilometer_logs(filepath):
    print(f"[LOG] Parsing ceilometer log: {filepath}")
    data_points = []
    log_pattern = re.compile(r"^(\d{12})\s+MSK\s+H1\s+=\s+b'(\d+)'")
    if not os.path.exists(filepath):
        print(f"  - WARNING: Ceilometer log not found: {filepath}")
        return []
    with open(filepath) as f:
        for line in f:
            match = log_pattern.match(line)
            if not match:
                continue
            ts_str, h_str = match.groups()
            h = int(h_str)
            if h == 9999:
                continue
            try:
                ts = datetime.datetime.strptime(ts_str, "%y%m%d%H%M%S")
                data_points.append((ts, h))
            except ValueError:
                continue
    print(f"  - Found {len(data_points)} valid ceilometer measurements.")
    return sorted(data_points, key=lambda x: x[0])


# PURPOSE: Extract (timestamp, best_matching_height) pairs from algorithm log files,
#          aligning each frame to the nearest-in-time ceilometer measurement.
#          Only considers top-tier clusters (size > top_cluster_min) for ceilometer matching,
#          falling back to the dominant large cluster if no ceilometer is nearby.
# INPUTS: directory_path (str), ceilometer_data (list), min_cluster_size (int), top_cluster_min (int)
# OUTPUTS: Sorted list of (datetime, float) tuples. Returns [] if directory missing.
# KEYWORDS: algorithm, parse, log, height, alignment
def parse_algorithm_logs(directory_path, ceilometer_data, min_cluster_size=3, top_cluster_min=5):
    print(f"[LOG] Parsing algorithm logs in: {directory_path}")
    ceilo_times_ts = np.array([d.timestamp() for d, _h in ceilometer_data])
    ceilo_heights = np.array([h for _d, h in ceilometer_data]) if ceilometer_data else np.array([])
    best_match_data = []
    if not os.path.exists(directory_path):
        print(f"  - WARNING: Algorithm log directory not found: {directory_path}")
        return []
    file_pattern = re.compile(r"img-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.txt")
    cluster_pattern = re.compile(r"Size: (\d+) points\s*\n\s*-?\s*Height \(Distance\): ([\d.]+) meters")
    for filename in os.listdir(directory_path):
        file_match = file_pattern.match(filename)
        if not file_match:
            continue
        ts_str = file_match.group(1)
        try:
            ts = datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S")
            full_path = os.path.join(directory_path, filename)
            with open(full_path) as f:
                content = f.read()
            found_clusters = cluster_pattern.findall(content)
            if not found_clusters:
                continue
            all_clusters = [(int(size), float(height)) for size, height in found_clusters]
            clusters = [c for c in all_clusters if c[0] >= min_cluster_size]
            if not clusters:
                continue
            clusters.sort(key=lambda x: x[0], reverse=True)
            algo_ts = ts.timestamp()
            selected_height = None
            # Prefer top-tier clusters (N > top_cluster_min) for ceilometer matching
            top_clusters = [c for c in clusters if c[0] > top_cluster_min]
            match_pool = top_clusters if top_clusters else clusters
            if len(ceilo_times_ts) > 0:
                time_diffs = np.abs(ceilo_times_ts - algo_ts)
                nearest_idx = np.argmin(time_diffs)
                if time_diffs[nearest_idx] <= 10:
                    target = ceilo_heights[nearest_idx]
                    algo_heights = [h for _s, h in match_pool]
                    height_diffs = np.abs(np.array(algo_heights) - target)
                    selected_height = algo_heights[np.argmin(height_diffs)]
            if selected_height is None:
                selected_height = match_pool[0][1]
            best_match_data.append((ts, selected_height))
        except ValueError:
            continue
    print(f"  - Generated 'Best-Match' time series with {len(best_match_data)} data points.")
    return sorted(best_match_data, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

# PURPOSE: Apply moving-average smoothing to a time-height series.
# INPUTS: data (list of (datetime, float/int)), window_size (int)
# OUTPUTS: Smoothed list of (datetime, float) tuples. Returns original data if window < 2.
# KEYWORDS: smooth, moving_average, convolution, filter
def smooth_data(data, window_size):
    if not data or window_size < 2:
        return data
    times, heights = zip(*data)
    heights = np.array(heights, dtype=float)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(heights, kernel, mode="valid")
    offset = (window_size - 1) // 2
    smoothed_times = times[offset : offset + len(smoothed)]
    print(f"  - Smoothed data from {len(data)} to {len(smoothed)} points (window={window_size}).")
    return list(zip(smoothed_times, smoothed))


# ---------------------------------------------------------------------------
# Constant (global) shift search
# ---------------------------------------------------------------------------

# PURPOSE: Find the single best global time shift between algorithm and ceilometer via correlation maximization.
# INPUTS: algo_data, ceilo_data (lists of tuples), shift_range_s (int), tolerance_s (int)
# OUTPUTS: best_shift (int), max_corr (float), aligned_data (list of (dt, algo_h, ceilo_h))
# KEYWORDS: global_shift, correlation, time_alignment, optimization
def find_constant_shift(algo_data, ceilo_data, shift_range_s, tolerance_s):
    if len(algo_data) < 2 or not ceilo_data:
        return 0, 0, []
    ceilo_ts = np.array([d.timestamp() for d, _h in ceilo_data])
    ceilo_hs = np.array([h for _d, h in ceilo_data])
    best_shift, max_corr, best_aligned = 0, -1, []
    for shift in range(-shift_range_s, shift_range_s + 1):
        algo_ts_shifted = np.array([t.timestamp() + shift for t, _h in algo_data])
        algo_hs = np.array([h for _t, h in algo_data])
        aligned_pairs = _align_nearest(algo_ts_shifted, algo_hs, ceilo_ts, ceilo_hs, tolerance_s)
        if len(aligned_pairs) < 2:
            continue
        ceilo_vals, algo_vals = zip(*aligned_pairs)
        r = np.corrcoef(ceilo_vals, algo_vals)[0, 1]
        if r > max_corr:
            max_corr, best_shift = r, shift
            best_aligned = [
                (datetime.datetime.fromtimestamp(algo_ts_shifted[i]), algo_hs[i], ceilo_hs[idx])
                for i, idx in enumerate(_nearest_indices(algo_ts_shifted, ceilo_ts))
                if idx < len(ceilo_ts) and abs(ceilo_ts[idx] - algo_ts_shifted[i]) <= tolerance_s
            ]
    return best_shift, max_corr, best_aligned


def _nearest_indices(query, reference):
    idx = np.searchsorted(reference, query)
    idx = np.clip(idx, 0, len(reference) - 1)
    return idx


def _align_nearest(query_ts, query_hs, ref_ts, ref_hs, tol):
    idx = _nearest_indices(query_ts, ref_ts)
    valid = np.abs(ref_ts[idx] - query_ts) <= tol
    return [(ref_hs[i], query_hs[j]) for j, i in enumerate(idx) if valid[j]]


# ---------------------------------------------------------------------------
# Dynamic (sliding-window) shift search
# ---------------------------------------------------------------------------

# PURPOSE: Compute a time-varying shift profile using a sliding window over the time series.
# INPUTS: algo_data, ceilo_data (list), window_min (int), search_s (int), tolerance_s (int)
# OUTPUTS: dynamic_shifts (list of (center_time, shift)), aligned_data (list of (dt, algo_h, ceilo_h))
# KEYWORDS: dynamic_shift, sliding_window, wind_proxy, time_varying
def compute_dynamic_shift(algo_data, ceilo_data, window_min, search_s, tolerance_s):
    print(f"--- Computing Dynamic Shift (Window: {window_min} min, Search: +-{search_s}s) ---")
    algo_data.sort(key=lambda x: x[0])
    ceilo_data.sort(key=lambda x: x[0])
    t_a, h_a = zip(*algo_data) if algo_data else ([], [])
    t_c, h_c = zip(*ceilo_data) if ceilo_data else ([], [])
    if not t_a or not t_c:
        return [], []
    start = max(t_a[0], t_c[0])
    end = min(t_a[-1], t_c[-1])
    current = start
    delta = datetime.timedelta(minutes=window_min)
    dyn_shifts = []
    aligned_full = []
    while current < end:
        nxt = current + delta
        slice_a = [(t, h) for t, h in algo_data if current <= t < nxt]
        slice_c = [(t, h) for t, h in ceilo_data if current <= t < nxt]
        if len(slice_a) > 10 and len(slice_c) > 10:
            st_a, sh_a = zip(*slice_a)
            st_c, sh_c = zip(*slice_c)
            shift = _best_shift_for_slice(st_a, sh_a, st_c, sh_c, search_s, tolerance_s)
            center = current + (delta / 2)
            dyn_shifts.append((center, shift))
            # re-align
            ts_a = np.array([t.timestamp() for t in st_a]) + shift
            ts_c = np.array([t.timestamp() for t in st_c])
            for i, t_val in enumerate(ts_a):
                idx = np.searchsorted(ts_c, t_val)
                if idx < len(ts_c) and abs(ts_c[idx] - t_val) <= tolerance_s:
                    aligned_full.append((slice_c[idx][0], sh_a[i], slice_c[idx][1]))
        current = nxt
    if dyn_shifts:
        shift_vals = [s for _c, s in dyn_shifts]
        shift_range = max(shift_vals) - min(shift_vals)
        print(f"  - Produced {len(dyn_shifts)} shift windows, {len(aligned_full)} aligned points.")
        print(f"  - Shift range: {min(shift_vals)} to {max(shift_vals)}s (span={shift_range}s)")
    else:
        print(f"  - Produced 0 shift windows, {len(aligned_full)} aligned points.")
    return dyn_shifts, aligned_full


def _best_shift_for_slice(t_a, h_a, t_c, h_c, search_range, tolerance_s):
    best_shift, max_corr = 0, -1
    ts_a = np.array([t.timestamp() for t in t_a])
    ts_c = np.array([t.timestamp() for t in t_c])
    h_a = np.array(h_a)
    h_c = np.array(h_c)
    for shift in range(-search_range, search_range + 1, 2):
        shifted = ts_a + shift
        pairs = _align_nearest(shifted, h_a, ts_c, h_c, tolerance_s)
        if len(pairs) < 10:
            continue
        cv, av = zip(*pairs)
        r = np.corrcoef(cv, av)[0, 1]
        if r > max_corr:
            max_corr, best_shift = r, shift
    return best_shift


# ---------------------------------------------------------------------------
# DTW analysis
# ---------------------------------------------------------------------------

# PURPOSE: Perform DTW (Dynamic Time Warping) on aligned ceilo vs algo height series.
# INPUTS: aligned_data, cluster_name (str), window_size (int)
# OUTPUTS: Side effect – displays/saves a DTW alignment plot.
# KEYWORDS: dtw, alignment, normalized, warp_path
def analyze_and_plot_dtw(aligned_data, cluster_name, window_size, save_dir):
    if len(aligned_data) < 2:
        print("  - Not enough data for DTW analysis.")
        return
    from dtw import dtw as _dtw
    _t, algo_h, ceilo_h = zip(*aligned_data)
    algo_h = np.array(algo_h)
    ceilo_h = np.array(ceilo_h)
    if np.std(ceilo_h) == 0 or np.std(algo_h) == 0:
        print("  - Could not perform DTW: One series has zero variance.")
        return
    ceilo_n = (ceilo_h - np.mean(ceilo_h)) / np.std(ceilo_h)
    algo_n = (algo_h - np.mean(algo_h)) / np.std(algo_h)
    result = _dtw(algo_n, ceilo_n, keep_internals=True,
                  window_type="sakoechiba", window_args={"window_size": window_size})
    print(f"\n--- DTW Analysis Results for {cluster_name} ---")
    print(f"  DTW Normalized Distance: {result.distance:.4f}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"DTW Alignment: Ceilometer vs. {cluster_name}", fontsize=16)
    ax1.plot(np.asarray(result.reference).flatten(), "r-", label="Ceilometer (Normalized)")
    ax1.set_ylabel("Normalized Height"); ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.plot(np.asarray(result.query).flatten(), "b-", label="Algorithm (Normalized)")
    ax2.set_xlabel("Measurement Index"); ax2.set_ylabel("Normalized Height"); ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.6)
    step = max(1, len(result.index1) // 10)
    for i in range(0, len(result.index1), step):
        i1, i2 = result.index1[i], result.index2[i]
        con = patches.ConnectionPatch(
            xyA=(i1, np.asarray(result.query)[i1]), coordsA=ax2.transData,
            xyB=(i2, np.asarray(result.reference)[i2]), coordsB=ax1.transData,
            color="gray", linestyle="--", linewidth=0.8, alpha=0.8,
        )
        fig.add_artist(con)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_or_show(fig, os.path.join(save_dir, "dtw_alignment.png"))


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------

def _apply_bias(name, params, algo_heights):
    if name == "Linear":
        return params["a"] * algo_heights + params["b"]
    elif name == "Quadratic":
        return params["a"] * algo_heights**2 + params["b"] * algo_heights + params["c"]
    elif name == "Exponential":
        return params["a"] * np.exp(params["b"] * algo_heights) + params["c"]
    return np.zeros_like(algo_heights)


# PURPOSE: Brute-force search for the best height-dependent bias model minimizing MSE.
# INPUTS: aligned_data (list), param_ranges (dict)
# OUTPUTS: dict with name, mse, params, and corrected data.
# KEYWORDS: bias_correction, model_fitting, mse, linear, quadratic, exponential
def find_best_bias_model(aligned_data, param_ranges):
    LINEAR_RANGES = param_ranges.get("linear", {"a": (-2, 2, 10), "b": (-1000, 1000, 10)})
    QUADRATIC_RANGES = param_ranges.get("quadratic", {"a": (-1e-4, 1e-4, 10), "b": (-2, 2, 10), "c": (-1000, 1000, 10)})
    EXP_RANGES = param_ranges.get("exponential", {"a": (-1000, 1000, 10), "b": (-1e-3, 1e-3, 10), "c": (-1000, 1000, 10)})

    if len(aligned_data) < 2:
        return {"name": "None", "mse": float("inf"), "params": {}, "data": aligned_data}
    _t, algo_h, ceilo_h = zip(*aligned_data)
    algo_h = np.array(algo_h)
    ceilo_h = np.array(ceilo_h)
    best = {"name": "None", "mse": float("inf"), "params": {}, "data": aligned_data}

    a_rng = np.linspace(*LINEAR_RANGES["a"])
    b_rng = np.linspace(*LINEAR_RANGES["b"])
    for a in a_rng:
        for b in b_rng:
            corrected = algo_h + (a * algo_h + b)
            mse = np.mean((corrected - ceilo_h) ** 2)
            if mse < best["mse"]:
                best.update({"name": "Linear", "mse": mse, "params": {"a": a, "b": b}})

    a_rng = np.linspace(*QUADRATIC_RANGES["a"])
    b_rng = np.linspace(*QUADRATIC_RANGES["b"])
    c_rng = np.linspace(*QUADRATIC_RANGES["c"])
    for a in a_rng:
        for b in b_rng:
            for c in c_rng:
                corrected = algo_h + (a * algo_h**2 + b * algo_h + c)
                mse = np.mean((corrected - ceilo_h) ** 2)
                if mse < best["mse"]:
                    best.update({"name": "Quadratic", "mse": mse, "params": {"a": a, "b": b, "c": c}})

    a_rng = np.linspace(*EXP_RANGES["a"])
    b_rng = np.linspace(*EXP_RANGES["b"])
    c_rng = np.linspace(*EXP_RANGES["c"])
    for a in a_rng:
        for b in b_rng:
            for c in c_rng:
                corrected = algo_h + (a * np.exp(b * algo_h) + c)
                mse = np.mean((corrected - ceilo_h) ** 2)
                if mse < best["mse"]:
                    best.update({"name": "Exponential", "mse": mse, "params": {"a": a, "b": b, "c": c}})

    if best["name"] != "None":
        p = best["params"]
        bias_vec = _apply_bias(best["name"], p, algo_h)
        corrected_h = algo_h + bias_vec
        corrected_data = []
        times, _, ceilos = zip(*aligned_data)
        for i, t in enumerate(times):
            corrected_data.append((t, corrected_h[i], ceilos[i]))
        best["data"] = corrected_data
    return best


# PURPOSE: Plot the fitted bias model as a function of algorithm height.
# INPUTS: model_info (dict), original_heights (np.array), save_dir (str)
# OUTPUTS: Saved or displayed PNG.
# KEYWORDS: bias_plot, correction, model_curve
def plot_bias_model(model_info, original_heights, save_dir):
    if model_info["name"] == "None":
        return
    h_range = np.linspace(min(original_heights), max(original_heights), 200)
    p = model_info["params"]
    bias = _apply_bias(model_info["name"], p, h_range)
    if model_info["name"] == "Linear":
        param_str = f"a={p['a']:.3f}, b={p['b']:.1f}"
    elif model_info["name"] == "Quadratic":
        param_str = f"a={p['a']:.1e}, b={p['b']:.3f}, c={p['c']:.1f}"
    elif model_info["name"] == "Exponential":
        param_str = f"a={p['a']:.1f}, b={p['b']:.1e}, c={p['c']:.1f}"
    else:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_range, bias, "g-", lw=2)
    ax.set_title(f"Best Fit Bias Model: {model_info['name']}\n({param_str})", fontsize=14)
    ax.set_xlabel("Original Algorithm Height (meters)", fontsize=12)
    ax.set_ylabel("Calculated Bias (Correction) [m]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axhline(0, color="k", linestyle=":", lw=1)
    plt.tight_layout()
    _save_or_show(fig, os.path.join(save_dir, "bias_model.png"))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, path=None):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


# PURPOSE: Plot the time-aligned height series of ceilometer and algorithm.
# INPUTS: ceilo_data, aligned_data, shift_label (str), save_path (str|None)
# OUTPUTS: Saved or displayed temporal comparison plot.
# KEYWORDS: temporal_plot, timeseries, alignment, height_comparison
def plot_temporal(ceilo_data, aligned_data, shift_label, save_path=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    all_heights = []
    if ceilo_data:
        t, h = zip(*ceilo_data)
        ax.plot(t, h, "o-", color="red", label="Ceilometer", markersize=5, linewidth=1)
        all_heights.extend(h)
    if aligned_data:
        t, ah, _ch = zip(*aligned_data)
        ax.plot(t, ah, "x-", color="blue", label=f"Algorithm {shift_label}", markersize=5, mew=2, linewidth=1)
        all_heights.extend(ah)
    y_lower = max(0, min(all_heights) * 0.95) if all_heights else 0
    ax.set_title(f"Time Series Alignment: Ceilometer vs. Algorithm ({shift_label})", fontsize=16)
    ax.set_xlabel("Time"); ax.set_ylabel("Cloud Base Height (m)")
    ax.set_ylim(bottom=y_lower); ax.legend(); ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(); plt.tight_layout()
    _save_or_show(fig, save_path)


# PURPOSE: Plot the correlation scatter between algorithm and ceilometer heights.
# INPUTS: aligned_data, shift_label (str), correlation (float), save_path (str|None)
# OUTPUTS: Saved or displayed scatter plot.
# KEYWORDS: correlation_plot, scatter, pearson, agreement
def plot_correlation(aligned_data, shift_label, correlation, save_path=None):
    if not aligned_data:
        return
    _t, ah, ch = zip(*aligned_data)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ch, ah, alpha=0.7, label=f"n={len(aligned_data)}")
    lim = max(max(ch), max(ah)) * 1.05
    ymin = max(0, min(min(ch), min(ah)) * 0.95)
    ax.plot([ymin, lim], [ymin, lim], "r--", label="y=x")
    ax.set_title(f"Correlation ({shift_label})\nr={correlation:.3f}", fontsize=14)
    ax.set_xlabel("Ceilometer Height (m)"); ax.set_ylabel("Algorithm Height (m)")
    ax.set_xlim(ymin, lim); ax.set_ylim(ymin, lim); ax.grid(True); ax.legend()
    ax.set_aspect("equal", adjustable="box"); plt.tight_layout()
    _save_or_show(fig, save_path)


# PURPOSE: Plot the height error (algo - ceilo) over time.
# INPUTS: aligned_data, shift_label (str), save_path (str|None)
# OUTPUTS: Saved or displayed error plot.
# KEYWORDS: error_plot, residual, timeseries
def plot_error(aligned_data, shift_label, save_path=None):
    if not aligned_data:
        return
    t, ah, ch = zip(*aligned_data)
    errors = np.array(ah) - np.array(ch)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(t, errors, "x", color="purple", linestyle="None", markersize=8, mew=2)
    ax.axhline(0, color="r", linestyle="--", linewidth=2, label="No Error")
    ax.set_title(f"Height Error Over Time ({shift_label})", fontsize=16)
    ax.set_xlabel("Time"); ax.set_ylabel("Error (Algo - Ceilo) [m]")
    ax.legend(); ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(); plt.tight_layout()
    _save_or_show(fig, save_path)


# PURPOSE: Plot the dynamic shift profile with an optional wind-angle secondary axis.
# INPUTS: dynamic_shifts (list), distance_m (float), save_path (str|None)
# OUTPUTS: Saved or displayed shift profile plot.
# KEYWORDS: shift_profile, dynamic, wind_angle, proxy
def plot_shift_profile(dynamic_shifts, distance_m=50.0, save_path=None):
    if not dynamic_shifts:
        return
    times_s, shift_vals = zip(*dynamic_shifts)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times_s, shift_vals, "o-", markersize=4, color="purple")
    ax1.set_title("Dynamic Time Delay Profile")
    ax1.set_ylabel("Time Shift (s)"); ax1.set_xlabel("Time")
    ax1.grid(True)
    shifts_arr = np.array(shift_vals)
    if max(np.abs(shifts_arr)) < distance_m / 5.0:
        def forward(shift):
            val = np.clip((shift * 10.0) / distance_m, -1.0, 1.0)
            return np.degrees(np.arccos(val))
        def inverse(angle):
            return (distance_m / 10.0) * np.cos(np.radians(angle))
        secax = ax1.secondary_yaxis("right", functions=(forward, inverse))
        secax.set_ylabel("Estimated Angle (assume V=10 m/s)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(); plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Time range intersection
# ---------------------------------------------------------------------------

# PURPOSE: Trim two sorted time-series to their intersecting time window.
#          Removes data points from either side that fall outside the mutual range.
# INPUTS: series_a, series_b — lists of (datetime, value) tuples, sorted ascending.
# OUTPUTS: (trimmed_a, trimmed_b) — both filtered to [max(start), min(end)].
# KEYWORDS: intersect, trim, time_range, overlap
def intersect_time_ranges(series_a, series_b):
    if not series_a or not series_b:
        return [], []

    start = max(series_a[0][0], series_b[0][0])
    end = min(series_a[-1][0], series_b[-1][0])
    if start >= end:
        return [], []

    a_in = [(t, v) for t, v in series_a if start <= t <= end]
    b_in = [(t, v) for t, v in series_b if start <= t <= end]

    return a_in, b_in


# ---------------------------------------------------------------------------
# Time-interval filtering
# ---------------------------------------------------------------------------

# PURPOSE: Filter a sorted time-height series to a configurable analysis window.
#          Logs how many points were inside and outside the interval.
# INPUTS: data (list of (datetime, float)), time_start (str "HH:MM:SS"), time_end (str "HH:MM:SS"), label (str)
# OUTPUTS: Filtered list of (datetime, float). Returns empty if no points in range.
# KEYWORDS: time_filter, window, analysis_interval, skip
def filter_by_time_window(data, time_start_str, time_end_str, label="series"):
    if not data:
        print(f"[SKIP] {label}: no data to filter (empty series).")
        return []
    if time_start_str is None or time_end_str is None:
        return data
    try:
        t_start = datetime.datetime.strptime(time_start_str, "%H:%M:%S").time()
        t_end = datetime.datetime.strptime(time_end_str, "%H:%M:%S").time()
    except ValueError:
        print(f"[WARN] Invalid time format in analysis window config. Keeping all {len(data)} points.")
        return data
    original_count = len(data)
    filtered = [(dt, val) for dt, val in data if t_start <= dt.time() <= t_end]
    skipped = original_count - len(filtered)
    print(f"[FILTER] {label}: {len(filtered)}/{original_count} points within "
          f"{time_start_str}--{time_end_str} ({skipped} skipped outside window).")
    return filtered


# PURPOSE: Count how many algorithm timesteps have no corresponding ceilometer measurement
#          within tolerance, and report them for diagnostic purposes.
# INPUTS: algo_data, ceilo_data (list of tuples), tolerance_s (int)
# OUTPUTS: (count_skipped, skipped_timestamps) — number and list of timestamps without ceilometer match.
# KEYWORDS: missing_data, ceilometer_gap, diagnostics, skip
def count_unmatched_algo_points(algo_data, ceilo_data, tolerance_s):
    if not algo_data or not ceilo_data:
        return len(algo_data) if algo_data else 0, []
    ceilo_ts = np.array([d.timestamp() for d, _h in ceilo_data])
    skipped = []
    for t, h in algo_data:
        diffs = np.abs(ceilo_ts - t.timestamp())
        if np.min(diffs) > tolerance_s:
            skipped.append(t)
    if skipped:
        print(f"[SKIP] {len(skipped)} algorithm timesteps have NO ceilometer data "
              f"within +/-{tolerance_s}s tolerance.")
    else:
        print(f"[OK] All {len(algo_data)} algorithm timesteps have ceilometer data within tolerance.")
    return len(skipped), skipped


# PURPOSE: Create a scatter plot of ALL algorithm-ceilometer pairs (nearest-in-time, no shift search)
#          across the full analysis interval, as a raw baseline before any shift optimization.
# INPUTS: algo_data, ceilo_data (list of (datetime, float)), tolerance_s (int), label (str), save_path (str|None)
# OUTPUTS: Saved or displayed scatter plot with Pearson r.
# KEYWORDS: all_points, scatter, baseline, raw_correlation, full_interval
def plot_all_points_correlation(algo_data, ceilo_data, tolerance_s, label, save_path=None):
    if not algo_data or not ceilo_data:
        print(f"[SKIP] Cannot plot all-points correlation: missing data.")
        return

    ceilo_ts = np.array([d.timestamp() for d, _h in ceilo_data])
    ceilo_hs = np.array([h for _d, h in ceilo_data])
    algo_ts = np.array([t.timestamp() for t, _h in algo_data])
    algo_hs = np.array([h for _t, h in algo_data])

    # For each algorithm frame, find the nearest-in-time ceilometer measurement
    ceilo_idx = _nearest_indices(algo_ts, ceilo_ts)
    valid_mask = np.abs(ceilo_ts[ceilo_idx] - algo_ts) <= tolerance_s
    paired_algo_h = algo_hs[valid_mask]
    paired_ceilo_h = ceilo_hs[ceilo_idx[valid_mask]]

    n_skipped = np.sum(~valid_mask)
    n_total = len(algo_hs)
    print(f"[PAIR] All-points pairing: {len(paired_algo_h)}/{n_total} algorithm frames "
          f"matched to ceilometer ({n_skipped} skipped, no ceilometer within {tolerance_s}s).")

    if len(paired_algo_h) < 2:
        print(f"[SKIP] Too few paired points ({len(paired_algo_h)}) for all-points correlation scatter.")
        return

    r = np.corrcoef(paired_ceilo_h, paired_algo_h)[0, 1]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(paired_ceilo_h, paired_algo_h, alpha=0.6, s=20,
               label=f"n={len(paired_algo_h)} (skipped={n_skipped})")
    lim = max(np.max(paired_ceilo_h), np.max(paired_algo_h)) * 1.05
    ymin = max(0, min(np.min(paired_ceilo_h), np.min(paired_algo_h)) * 0.95)
    ax.plot([ymin, lim], [ymin, lim], "r--", linewidth=1.5, label="y = x")
    ax.set_title(f"All-Points Correlation ({label})\nPearson r = {r:.4f}",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Ceilometer Height (m)", fontsize=12)
    ax.set_ylabel("Algorithm Height (m)", fontsize=12)
    ax.set_xlim(ymin, lim)
    ax.set_ylim(ymin, lim)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return r, len(paired_algo_h), n_skipped


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# PURPOSE: Pair algorithm and ceilometer heights at a given fixed time shift.
# INPUTS: algo_data, ceilo_data (list of (datetime, value) tuples), shift_s (int), tolerance_s (int)
# OUTPUTS: list of (datetime, algo_height, ceilo_height) tuples aligned at the given shift
# KEYWORDS: align, nearest, pairing, time_shift
def _align_at_shift(algo_data, ceilo_data, shift_s, tolerance_s):
    if not algo_data or not ceilo_data:
        return []
    ceilo_ts = np.array([d.timestamp() for d, _h in ceilo_data])
    ceilo_hs = np.array([h for _d, h in ceilo_data])
    algo_ts = np.array([t.timestamp() + shift_s for t, _h in algo_data])
    algo_hs = np.array([h for _t, h in algo_data])
    idx = _nearest_indices(algo_ts, ceilo_ts)
    aligned = []
    for i, j in enumerate(idx):
        if j < len(ceilo_ts) and abs(ceilo_ts[j] - algo_ts[i]) <= tolerance_s:
            aligned.append((datetime.datetime.fromtimestamp(algo_ts[i]), algo_hs[i], ceilo_hs[j]))
    return aligned


def run_pipeline(cfg):
    os.makedirs(cfg["assets_dir"], exist_ok=True)
    ceilo = parse_ceilometer_logs(cfg["ceilo_path"])
    algo = parse_algorithm_logs(cfg["algo_dir"], ceilo, cfg.get("min_cluster_size", 3), cfg.get("top_cluster_min", 5))
    if not ceilo or len(algo) < 2:
        print(f"\n[SKIP] Insufficient data: ceilo={len(ceilo)} points, algo={len(algo)} points. Exiting.")
        return

    # Filter both time series to the configured analysis time window
    print(f"\n--- Time-Window Filtering ---")
    t_start = cfg.get("analysis_time_start", None)
    t_end = cfg.get("analysis_time_end", None)
    if t_start and t_end:
        ceilo = filter_by_time_window(ceilo, t_start, t_end, "ceilometer")
        algo = filter_by_time_window(algo, t_start, t_end, "algorithm")
        if not ceilo or not algo:
            print(f"\n[SKIP] No data remaining after time-window filter. Exiting.")
            return

    # Trim both datasets to their intersecting time interval
    print(f"\n--- Intersecting Time Ranges ---")
    print(f"  Ceilometer: {ceilo[0][0]} to {ceilo[-1][0]} ({len(ceilo)} points)")
    print(f"  Algorithm:  {algo[0][0]} to {algo[-1][0]} ({len(algo)} points)")
    ceilo, algo = intersect_time_ranges(ceilo, algo)
    if not ceilo or not algo:
        print(f"[SKIP] No overlapping time range between ceilometer and algorithm data. Exiting.")
        return
    print(f"  Intersection: {ceilo[0][0]} to {ceilo[-1][0]} ({len(ceilo)} ceilo + {len(algo)} algo)")

    # Report algorithm timesteps without ceilometer data
    print(f"\n--- Ceilometer Coverage Check ---")
    tolerance = cfg["time_tolerance"]
    count_unmatched_algo_points(algo, ceilo, tolerance)

    if cfg["smooth"]:
        print(f"\n--- Smoothing (window={cfg['smooth_window']}) ---")
        ceilo = smooth_data(ceilo, cfg["smooth_window"])
        algo = smooth_data(algo, cfg["smooth_window"])

    assets = cfg["assets_dir"]

    # Zero-shift baseline: direct nearest-in-time pairing without any time shift
    print(f"\n--- Zero-Shift Baseline ---")
    raw_aligned = _align_at_shift(algo, ceilo, 0, tolerance)
    if raw_aligned:
        _t, ah, ch = zip(*raw_aligned)
        raw_r = np.corrcoef(ch, ah)[0, 1]
    else:
        raw_r = 0
    print(f"  r={raw_r:.4f}, n={len(raw_aligned)}")

    # All-points correlation scatter (full time interval, raw pairing, no shift search)
    print(f"\n--- All-Points Correlation (full interval) ---")
    plot_all_points_correlation(
        algo, ceilo, tolerance, cfg.get("affine_label", "Unknown"),
        os.path.join(assets, "correlation_all_points.png"))

    # Pre-shift plots (raw, zero-shift alignment)
    plot_temporal(ceilo, raw_aligned, "No Shift",
                  os.path.join(assets, "timeseries_before.png"))
    plot_correlation(raw_aligned, "No Shift", raw_r,
                     os.path.join(assets, "correlation_before.png"))

    if cfg["mode"] == "constant":
        print(f"\n--- Constant Shift Search (+-{cfg['shift_range']}s) ---")
        best_shift, max_corr, aligned = find_constant_shift(
            algo, ceilo, cfg["shift_range"], cfg["time_tolerance"])
        print(f"  Best Shift: {best_shift}s, r={max_corr:.4f}, n={len(aligned)}")
        label = f"Const Shift {best_shift}s"
        final_data = aligned
        final_r = max_corr
        final_name = "Best-Match"
    else:  # dynamic
        shifts, aligned = compute_dynamic_shift(
            algo, ceilo, cfg["window_min"], cfg["shift_range"], cfg["time_tolerance"])
        plot_shift_profile(shifts, cfg["distance_m"],
                           os.path.join(assets, "dynamic_shift_profile.png"))
        if aligned:
            _t, ah, ch = zip(*aligned)
            final_r = np.corrcoef(ch, ah)[0, 1]
        else:
            final_r = 0
        label = "Dynamic Shift"
        final_data = aligned
        final_name = "Best-Match"
        best_shift = "dynamic"

    print(f"\n  Raw (zero-shift) r={raw_r:.4f}  |  After {label} r={final_r:.4f}")

    if cfg["bias_correction"] and final_data:
        print("\n--- Bias Correction ---")
        pr = {
            "linear": {"a": (-2, 2, 10), "b": (-1000, 1000, 10)},
            "quadratic": {"a": (-1e-4, 1e-4, 10), "b": (-2, 2, 10), "c": (-1000, 1000, 10)},
            "exponential": {"a": (-1000, 1000, 10), "b": (-1e-3, 1e-3, 10), "c": (-1000, 1000, 10)},
        }
        model = find_best_bias_model(final_data, pr)
        if model["name"] != "None":
            params_str = ", ".join(f"{k}={v:.4f}" for k, v in model["params"].items())
            print(f"  Best Model: {model['name']} | Params: {params_str} | MSE: {model['mse']:.2f}")
            final_data = model["data"]
            final_name += f" ({model['name']} Bias Corrected)"
            if final_data:
                _t, ah, ch = zip(*final_data)
                final_r = np.corrcoef(ch, ah)[0, 1]
                print(f"  Corrected r={final_r:.4f}")
                _t2, orig_ah, _ch2 = zip(*final_data)
                plot_bias_model(model, orig_ah, assets)
        else:
            print("  - Could not find better bias model.")

    # Post-shift/correction final plots
    if final_data:
        plot_temporal(ceilo, final_data, label, os.path.join(assets, "timeseries_final.png"))
        plot_correlation(final_data, final_name, final_r, os.path.join(assets, "correlation_final.png"))
        plot_error(final_data, final_name, os.path.join(assets, "error_timeseries.png"))
        if cfg["dtw"]:
            analyze_and_plot_dtw(final_data, final_name, cfg["dtw_window"], assets)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = {
        "ceilo_path": "/home/omega-luler/tasks/clouds/data/height data/Zve_Sci_250523.txt",
        "algo_dir": "/home/omega-luler/tasks/clouds/logs/250523/matchanything_eloftr",
        "assets_dir": "report_assets_ma",
        "mode": "dynamic",
        "shift_range": 300,
        "time_tolerance": 10,
        "smooth": True,
        "smooth_window": 5,
        "window_min": 5,
        "distance_m": 200.0,
        "bias_correction": False,
        "dtw": False,
        "dtw_window": 20,
        "affine_label": "Default",
    }
    # CLI override: first arg = algo_dir, second arg = assets_dir, third arg = affine_label
    if len(sys.argv) > 1 and sys.argv[1]:
        cfg["algo_dir"] = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        cfg["assets_dir"] = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]:
        cfg["affine_label"] = sys.argv[3]
    print(f"[CONFIG] affine_label={cfg['affine_label']}  |  mode={cfg['mode']}")
    run_pipeline(cfg)
