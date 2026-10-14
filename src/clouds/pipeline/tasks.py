# PURPOSE: The task-level entry points that operate on a whole acquisition:
#          process a directory, process a window into per-pair logs, analyse a
#          manual trail, sweep the calibration, or run the (unimplemented)
#          iterative refinement. Each task takes an explicit configuration and
#          analyzer rather than reaching into a shared runner object.
# INPUTS: cfg (clouds.config.Config), stereo (StereoCameraSystem), analyzer
#         (pipeline.analyze.StereoAnalyzer), plus per-task options.
# OUTPUTS: log files under the configured log directory, console reports.
# KEYWORDS: tasks, process_directory, time_series, trail, ablation
from __future__ import annotations

import datetime
import math
import os

import numpy as np

from .. import viz
from ..sources.algorithm_log import log_dir_for, write_pair_logs
from ..sources.images import find_image_pairs, find_pairs_in_window, image_clock_window
from .analyze import VIS_CONTROLS, StereoAnalyzer

def process_single_pair(analyzer, img1_path: str, img2_path: str,
                         affine_matrix=None) -> tuple[float | None, float | None]:
    """
    Processes one pair of images, prints results to console, and returns
    the distance and error for the largest cluster for backward compatibility.

    affine_matrix: optional 2x3 override used by the ablation sweep; when
    omitted, the calibrated matrix held by the runner is used.
    """
    result = analyzer.analyze_pair(img1_path, img2_path,
                                       affine_matrix=affine_matrix)
    
    if result is not None:
        analysis_results = result['clusters']
        if analysis_results:
            print(f"=================================\nResult for {os.path.basename(img1_path)}:")
            for i, res in enumerate(analysis_results):
                label = "Dominant Cluster" if i == 0 else f"Cluster {i+1}"
                print(f"  - {label} ({res['size']} points): Distance = {res['distance']:.2f} ± {res['error']:.2f} meters")
            print("=================================")
            dominant_cluster = analysis_results[0]
            return dominant_cluster['distance'], dominant_cluster['error']
    print(f"No valid clusters found for {os.path.basename(img1_path)}.")
    return None, None


def process_directory(cfg, stereo, analyzer):
    """Processes all image pairs in the configured directory."""
    image_pairs = find_image_pairs(cfg.dataset.images_dir)
    if not image_pairs:
        print(f"No valid image pairs found in {cfg.dataset.images_dir}")
        return
        
    print(f"Found {len(image_pairs)} image pairs to process.")
    for img1_path, img2_path in image_pairs:
        process_single_pair(analyzer, img1_path, img2_path)

# PURPOSE: Find and pair the two camera streams inside a wall-clock window,
#          matching each devID1 frame to the temporally nearest devID2 frame.
# INPUTS: start_str, end_str (str, "HH-MM-SS", inclusive window applied to the
#         filename timestamp); pair_tolerance_s (float, max allowed separation
#         between the two cameras' filename timestamps).
# OUTPUTS: Sorted list of (devID1_path, devID2_path) tuples. Empty list when the
#          window is malformed or no frame can be paired. Each devID2 frame is
#          consumed at most once.
# KEYWORDS: pairing, stereo, nearest_timestamp, time_window, devID


def process_time_series(cfg, stereo, analyzer, backend=None, affine_matrix=None):
    """
    Processes a time-ordered series of image pairs within a configured time window.
    
    Args:
        backend: Optional override for the matcher backend name. Used for log subdirectory.
        affine_matrix: Optional 2x3 numpy array overriding the calibrated matrix.
    """
    # The dataset window is expressed in the ceilometer clock; the filename
    # filter works in the image clock. Convert once, here.
    win_start, win_end = image_clock_window(cfg)
    image_pairs = find_pairs_in_window(cfg.dataset.images_dir, win_start, win_end,
                                       cfg.dataset.date, cfg.dataset.pair_tolerance_s)

    if not image_pairs:
        print(f"No valid image pairs found in {cfg.dataset.images_dir} between "
              f"{win_start} and {win_end} (image clock)")
        return

    log_dir = log_dir_for(cfg.output.logs_dir, backend or cfg.matching.backend)
    print(f"Logging results to: {log_dir}")

    stride = max(1, int(cfg.processing.stride))
    if stride > 1:
        image_pairs = image_pairs[::stride]

    print(f"Found {len(image_pairs)} time-filtered image pairs to process (stride={stride}).")
    
    skipped_no_clusters = 0
    skipped_no_height = 0
    for img1_path, img2_path in image_pairs:
        base_filename = os.path.basename(img1_path)
        log_name = base_filename.replace('devID1.jpg', '.txt')
        log_path = os.path.join(log_dir, log_name)
        
        if os.path.exists(log_path):
            print(f"\nLog for {base_filename} already exists. Skipping.")
            continue

        result = analyzer.analyze_pair(img1_path, img2_path, affine_matrix=affine_matrix)
        
        if result is not None:
            clusters = result['clusters']
            if clusters:
                print("  Analysis Results:")
                for i, res in enumerate(clusters):
                    label = "Dominant Cluster" if i == 0 else f"Cluster {i+1}"
                    print(f"    - {label} ({res['size']} points): Distance = {res['distance']:.2f} m")
            else:
                skipped_no_height += 1
                print(f"  [SKIP] No valid clusters (too few matches). Frames skipped so far: {skipped_no_height}")
        else:
            skipped_no_clusters += 1
            print(f"  [SKIP] Insufficient matches for clustering. Frames skipped so far: {skipped_no_clusters}")

        if cfg.processing.log_results and result is not None:
            write_pair_logs(result, img1_path, log_dir)

    total_skipped = skipped_no_clusters + skipped_no_height
    print(f"\n--- Time Series Complete ---")
    print(f"  Total frames processed: {len(image_pairs)}")
    print(f"  Skipped (no clusters/too few matches): {skipped_no_clusters}")
    print(f"  Skipped (clusters but no valid height): {skipped_no_height}")
    print(f"  Total skipped: {total_skipped}")
    print(f"  Valid frames with height: {len(image_pairs) - total_skipped}")


def analyze_trail(cfg, stereo):
    """
    Processes a single pair of images using manually selected points for an 
    aircraft trail, calculating a distance for each point to see the distribution.
    """
    # 1. Load the manually matched points
    try:
        data = np.load(cfg.dataset.manual_points)
        kp1 = data['kp1']
        kp2 = data['kp2']
        print(f"Loaded {len(kp1)} manually matched points from {cfg.dataset.manual_points}")
    except FileNotFoundError:
        print(f"Error: Manual points file not found at '{cfg.dataset.manual_points}'")
        print("Please run 'tools/trail_matcher.py' first to generate the points file.")
        return

    # 2. Calculate shifts (disparity) for each point
    shifts = kp1 - kp2
    
    # 3. Calculate distance for each point
    distances = []
    for shift_vector in shifts:
        disp_x = shift_vector[0]
        if abs(disp_x) > 1e-6: # Avoid division by zero
            distance, _ = stereo.compute_distance(disp_x, delta_px_error=0)
            distances.append(distance)
    
    if not distances:
        print("No valid distances could be calculated.")
        return

    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    print("\n" + "="*40)
    print("Aircraft Trail Distance Analysis")
    print(f"  - Points Analyzed: {len(distances)}")
    print(f"  - Mean Distance:    {mean_dist:.2f} meters")
    print(f"  - Std Deviation:    {std_dist:.2f} meters")
    print(f"  - Min Distance:     {np.min(distances):.2f} meters")
    print(f"  - Max Distance:     {np.max(distances):.2f} meters")
    print("="*40 + "\n")

    # 4. Visualize the results
    if cfg.processing.visualize:
        viz.plot_trail_distance_distribution(distances, kp1)


def run_homography_ablation(cfg, stereo, analyzer):
    """Runs an ablation study by injecting noise into the affine matrix rotation."""
    image_pairs = find_image_pairs(cfg.dataset.images_dir)
    if not image_pairs:
        print("No image pairs found for ablation study.")
        return

    img1_path, img2_path = image_pairs[0] # Use the first pair for the study
    print(f"Running ablation study on: {os.path.basename(img1_path)}")

    theta_vals, dist_vals, err_vals = [], [], []
    
    original_M = analyzer.affine.copy()

    for i in range(cfg.ablation.steps):
        theta = cfg.ablation.max_noise_theta * i / (cfg.ablation.steps - 1)
        print(f"\n--- Testing with theta noise: {theta:.2f} degrees ---")

        theta_rad = math.radians(theta)
        rotation_noise = np.array([
            [math.cos(theta_rad), -math.sin(theta_rad)],
            [math.sin(theta_rad), math.cos(theta_rad)]
        ])

        # Perturb a local copy only; the calibrated matrix is never mutated.
        M_perturbed = original_M.copy()
        M_perturbed[:, :2] = original_M[:, :2] @ rotation_noise

        distance, error = process_single_pair(analyzer, 
            img1_path, img2_path, affine_matrix=M_perturbed)

        if distance is not None:
            theta_vals.append(theta)
            dist_vals.append(distance)
            err_vals.append(error)

    if cfg.processing.visualize and theta_vals:
        viz.plot_ablation_results(theta_vals, dist_vals, err_vals)


def run_iterative_refinement():
    print("Iterative refinement pipeline is not fully implemented in this refactoring yet.")
    pass
