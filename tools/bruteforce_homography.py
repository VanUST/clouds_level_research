# /tools/bruteforce_homography.py
import os
import sys
import re
import datetime
import numpy as np
import cv2
from tqdm import tqdm
import math

# Add the project root to the Python path to allow importing 'config' and 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.camera import StereoCameraSystem
from src.pipeline import PipelineRunner

# =================================================================================
# 1. CONFIGURATION
# =================================================================================
# --- DATA PATHS ---
CEILOMETER_LOG_FILE = "/home/omega-luler/tasks/clouds_level_research/data/height data/Zve_Sci_250523.txt"

# --- BRUTE-FORCE SEARCH RANGES ---
# Define search spaces for perturbations around the original AFFINE_MATRIX from config.py
# Format: (start, stop, num_steps). More steps increase precision but also computation time.

# Rotation perturbation in degrees
THETA_RANGE = (-1, 1, 40)

# Translation perturbation in pixels for the X-axis
TX_RANGE = (-40, 40, 40)

# Translation perturbation in pixels for the Y-axis
TY_RANGE = (-40, 40, 40)

# --- ANALYSIS PARAMETERS ---
# Time tolerance in seconds to match an algorithm log with a ceilometer reading
TIME_ALIGNMENT_TOLERANCE_S = 30

# --- SMOOTHING ---
# Set to True to apply a moving average filter to the time series data before MSE calculation
ENABLE_SMOOTHING = True
SMOOTHING_WINDOW_SIZE = 5
# =================================================================================


def parse_ceilometer_logs(filepath: str) -> list[tuple[datetime.datetime, int]]:
    """Parses a ceilometer log file to extract (timestamp, height) pairs."""
    print(f"Parsing ceilometer log: {filepath}")
    data_points = []
    log_pattern = re.compile(r"^(\d{12})\s+MSK\s+H1\s+=\s+b'(\d+)'")

    if not os.path.exists(filepath):
        print(f"  - WARNING: Ceilometer log file not found. Skipping.")
        return []

    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.match(line)
            if match:
                timestamp_str, height_str = match.groups()
                height = int(height_str)
                if height == 9999:
                    continue
                try:
                    timestamp = datetime.datetime.strptime(timestamp_str, "%y%m%d%H%M%S")
                    data_points.append((timestamp, height))
                except ValueError:
                    continue
    
    print(f"  - Found {len(data_points)} valid ceilometer measurements.")
    return sorted(data_points, key=lambda x: x[0])


def smooth_data(data: list, window_size: int) -> list:
    """Applies a moving average filter to time series data."""
    if not data or window_size < 2:
        return data
    times, heights = zip(*data)
    heights = np.array(heights, dtype=float)
    smoothed_heights = np.convolve(heights, np.ones(window_size)/window_size, mode='valid')
    offset = (window_size - 1) // 2
    smoothed_times = times[offset : offset + len(smoothed_heights)]
    return list(zip(smoothed_times, smoothed_heights))


def create_perturbed_matrix(original_matrix, d_theta, d_tx, d_ty):
    """
    Applies rotation and translation perturbations to an original affine matrix.
    """
    # Create rotation matrix for the perturbation angle
    d_theta_rad = math.radians(d_theta)
    cos_t, sin_t = math.cos(d_theta_rad), math.sin(d_theta_rad)
    rotation_delta_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # Separate original matrix into rotation/scale and translation parts
    R_original = original_matrix[:, :2]
    t_original = original_matrix[:, 2]
    
    # Apply the rotation to the original rotation part
    R_new = R_original @ rotation_delta_matrix
    
    # Apply the translation to the original translation part
    t_new = t_original + np.array([d_tx, d_ty])
    
    # Combine back into a 2x3 affine matrix
    new_matrix = np.hstack([R_new, t_new.reshape(2, 1)])
    return new_matrix


def main():
    """
    Main function to run the brute-force homography calibration.
    """
    # 1. Initialize core components from the main application
    stereo_system = StereoCameraSystem(
        base=config.STEREO_BASE,
        angle_of_view=config.ANGLE_OF_VIEW,
        image_width=config.IMAGE_WIDTH,
        model=config.CAMERA_MODEL
    )
    runner = PipelineRunner(stereo_system, config)

    # 2. Load and prepare data
    ceilometer_data = parse_ceilometer_logs(CEILOMETER_LOG_FILE)
    image_pairs = runner._get_time_filtered_pairs(
        config.TIME_SERIES_START_TIME, 
        config.TIME_SERIES_END_TIME
    )

    if not ceilometer_data or not image_pairs:
        print("Insufficient data. Check paths and time ranges in config.py.")
        return

    if ENABLE_SMOOTHING:
        print(f"\nApplying smoothing with window size {SMOOTHING_WINDOW_SIZE}...")
        ceilometer_data = smooth_data(ceilometer_data, SMOOTHING_WINDOW_SIZE)

    # Create a quick lookup map for ceilometer data
    ceilo_map = {t.timestamp(): h for t, h in ceilometer_data}
    ceilo_timestamps = np.array(list(ceilo_map.keys()))

    # 3. Setup the brute-force search space
    original_matrix = config.AFFINE_MATRIX.copy()
    theta_space = np.linspace(*THETA_RANGE)
    tx_space = np.linspace(*TX_RANGE)
    ty_space = np.linspace(*TY_RANGE)
    
    total_iterations = len(theta_space) * len(tx_space) * len(ty_space)
    print(f"\nStarting brute-force search with {total_iterations} total iterations.")

    best_mse = float('inf')
    best_params = {'theta': 0, 'tx': 0, 'ty': 0}
    best_matrix = original_matrix

    # 4. Run the main brute-force loop
    with tqdm(total=total_iterations, desc="Brute-forcing Matrix") as pbar:
        for d_theta in theta_space:
            for d_tx in tx_space:
                for d_ty in ty_space:
                    pbar.update(1)
                    
                    # Create and set the candidate matrix for this iteration
                    candidate_matrix = create_perturbed_matrix(original_matrix, d_theta, d_tx, d_ty)
                    runner.config.AFFINE_MATRIX = candidate_matrix
                    
                    aligned_algo_heights = []
                    aligned_ceilo_heights = []

                    # Process all image pairs with the current candidate matrix
                    for img1_path, img2_path in image_pairs:
                        # Find corresponding ceilometer height
                        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', img1_path)
                        if not match: continue
                        
                        img_time = datetime.datetime.strptime(match.group(1), '%Y-%m-%dT%H-%M-%S')
                        img_ts = img_time.timestamp()

                        time_diffs = np.abs(ceilo_timestamps - img_ts)
                        nearest_idx = np.argmin(time_diffs)
                        
                        if time_diffs[nearest_idx] > TIME_ALIGNMENT_TOLERANCE_S:
                            continue
                        
                        target_ceilo_height = ceilo_map[ceilo_timestamps[nearest_idx]]
                        
                        # Run analysis
                        analysis_results = runner._analyze_stereo_pair(img1_path, img2_path)
                        if not analysis_results:
                            continue

                        # Find the best matching cluster
                        cluster_heights = [r['distance'] for r in analysis_results]
                        best_height = min(cluster_heights, key=lambda h: abs(h - target_ceilo_height))
                        
                        aligned_algo_heights.append(best_height)
                        aligned_ceilo_heights.append(target_ceilo_height)

                    # Calculate MSE if enough points were found
                    if len(aligned_algo_heights) > 2:
                        mse = np.mean((np.array(aligned_algo_heights) - np.array(aligned_ceilo_heights))**2)
                        if mse < best_mse:
                            best_mse = mse
                            best_params = {'theta': d_theta, 'tx': d_tx, 'ty': d_ty}
                            best_matrix = candidate_matrix
                            pbar.set_postfix({
                                'Best MSE': f'{math.sqrt(best_mse):.2f}m',
                                'd_theta': f'{d_theta:.2f}°',
                                'd_tx': f'{d_tx:.1f}px',
                                'd_ty': f'{d_ty:.1f}px'
                            })

    # 5. Print final results
    print("\n" + "="*50)
    print("Brute-force search complete!")
    if np.isinf(best_mse):
        print("No valid matrix configurations found that produced results.")
    else:
        print(f"Original Matrix:\n{original_matrix}")
        print("\n--- Best Found Parameters ---")
        print(f"  - MSE: {best_mse:.2f} (RMSE: {math.sqrt(best_mse):.2f} meters)")
        print(f"  - Perturbations:")
        print(f"    - Delta Theta: {best_params['theta']:.3f} degrees")
        print(f"    - Delta Tx:    {best_params['tx']:.2f} pixels")
        print(f"    - Delta Ty:    {best_params['ty']:.2f} pixels")
        print("\n--- New Optimal Matrix ---")
        print("Copy the following numpy array into your config.py as AFFINE_MATRIX:")
        print(f"\n{repr(best_matrix)}\n")
    print("="*50)


if __name__ == "__main__":
    main()