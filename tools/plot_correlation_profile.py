import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.find_time_shift import (
    parse_ceilometer_logs, 
    parse_algorithm_logs, 
    smooth_data
)

# --- CONFIG ---
LOG_DATE = "250523"
BASE_DIR = "/home/omega-luler/tasks/clouds_level_research"
CEILOMETER_PATH = os.path.join(BASE_DIR, f"data/height data/Zve_Sci_{LOG_DATE}.txt")
ALGO_LOG_DIR = os.path.join(BASE_DIR, f"logs/{LOG_DATE}")
ASSETS_DIR = os.path.join(BASE_DIR, "report_assets")

# Constants
SMOOTH_WINDOW = 10
SHIFT_SEARCH_RANGE = 60*5
TIME_TOLERANCE = 10
# Range of window sizes to test (in minutes)
WINDOW_TEST_RANGE = range(5, 25, 1) 

def get_best_shift_for_slice(t_algo, h_algo, t_ceilo, h_ceilo, search_range):
    """Finds best shift for a small slice of data."""
    best_shift = 0
    max_corr = -1.0 # Start with lowest possible correlation
    
    ts_algo = np.array([t.timestamp() for t in t_algo])
    ts_ceilo = np.array([t.timestamp() for t in t_ceilo])
    h_ceilo = np.array(h_ceilo)
    h_algo = np.array(h_algo)

    # Optimization: Step=2 for speed
    for shift in range(-search_range, search_range + 1, 2):
        shifted_algo = ts_algo + shift
        
        # Fast alignment via nearest neighbor
        indices = np.searchsorted(ts_ceilo, shifted_algo)
        
        h_a_align = []
        h_c_align = []
        
        # Vectorization is harder here due to the bounds check, standard loop is safer
        for i, idx in enumerate(indices):
            if idx < len(ts_ceilo):
                if abs(ts_ceilo[idx] - shifted_algo[i]) < TIME_TOLERANCE:
                    h_a_align.append(h_algo[i])
                    h_c_align.append(h_ceilo[idx])

        # Only calculate correlation if we have statistically significant data points
        if len(h_a_align) > 15:
            # Handle cases with constant input (variance=0) which causes NaN correlation
            if np.std(h_c_align) > 0 and np.std(h_a_align) > 0:
                curr_corr = np.corrcoef(h_c_align, h_a_align)[0, 1]
                if curr_corr > max_corr:
                    max_corr = curr_corr
                    best_shift = shift
                
    return best_shift

def evaluate_window_size(window_minutes, algo_data, ceilo_data):
    """
    Runs the dynamic shift logic for a specific window size 
    and returns the Global Correlation Coefficient.
    """
    # Sort by time
    algo_data.sort(key=lambda x: x[0])
    ceilo_data.sort(key=lambda x: x[0])
    
    t_algo_full, _ = zip(*algo_data)
    t_ceilo_full, _ = zip(*ceilo_data)
    
    start_time = max(t_algo_full[0], t_ceilo_full[0])
    end_time = min(t_algo_full[-1], t_ceilo_full[-1])
    
    current_time = start_time
    window_delta = datetime.timedelta(minutes=window_minutes)
    
    aligned_h_algo = []
    aligned_h_ceilo = []
    
    while current_time < end_time:
        next_time = current_time + window_delta
        
        # Slice data
        slice_algo = [(t, h) for t, h in algo_data if current_time <= t < next_time]
        slice_ceilo = [(t, h) for t, h in ceilo_data if current_time <= t < next_time]
        
        # Minimum points required to attempt alignment
        if len(slice_algo) > 15 and len(slice_ceilo) > 15:
            t_a, h_a = zip(*slice_algo)
            t_c, h_c = zip(*slice_ceilo)
            
            # Find best local shift
            shift = get_best_shift_for_slice(t_a, h_a, t_c, h_c, SHIFT_SEARCH_RANGE)
            
            # Re-align points for global metric
            ts_a = np.array([t.timestamp() for t in t_a]) + shift
            ts_c = np.array([t.timestamp() for t in t_c])
            
            for i, t_val in enumerate(ts_a):
                idx = np.searchsorted(ts_c, t_val)
                if idx < len(ts_c) and abs(ts_c[idx] - t_val) < TIME_TOLERANCE:
                    aligned_h_algo.append(h_a[i])
                    aligned_h_ceilo.append(slice_ceilo[idx][1])

        current_time = next_time

    if len(aligned_h_algo) < 50:
        return 0.0 # Not enough matched points to be valid

    return np.corrcoef(aligned_h_algo, aligned_h_ceilo)[0, 1]

def main():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    print("--- Loading & Smoothing Data ---")
    ceilo_data = parse_ceilometer_logs(CEILOMETER_PATH)
    algo_data = parse_algorithm_logs(ALGO_LOG_DIR, ceilo_data)

    c_smooth = smooth_data(ceilo_data, SMOOTH_WINDOW)
    a_smooth = smooth_data(algo_data, SMOOTH_WINDOW)
    
    results_window = []
    results_corr = []

    print(f"--- Starting Optimization (Range: {min(WINDOW_TEST_RANGE)} - {max(WINDOW_TEST_RANGE)} min) ---")
    
    best_corr = -1
    best_window = 0

    for w in WINDOW_TEST_RANGE:
        corr = evaluate_window_size(w, a_smooth, c_smooth)
        print(f"Window: {w} min | Correlation: {corr:.4f}")
        results_window.append(w)
        results_corr.append(corr)
        
        if corr > best_corr:
            best_corr = corr
            best_window = w

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_window, results_corr, 'o-', linewidth=2, markersize=6, label='Correlation')
    
    # Highlight max
    ax.axvline(best_window, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_window} min')
    ax.scatter([best_window], [best_corr], color='red', s=100, zorder=5)
    
    ax.set_title(f'Dynamic Alignment Optimization\nBest Window Size: {best_window} min (r={best_corr:.3f})')
    ax.set_xlabel('Window Size (minutes)')
    ax.set_ylabel('Global Correlation Coefficient')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()
    
    output_path = os.path.join(ASSETS_DIR, "window_size_optimization.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nOptimization Complete.")
    print(f"Best Window: {best_window} min")
    print(f"Best Correlation: {best_corr:.4f}")
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()