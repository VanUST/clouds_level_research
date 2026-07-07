# /config.py
import numpy as np
from enum import Enum

class CameraModel(Enum):
    """Enum to define the camera projection model."""
    PINHOLE = 'pinhole'
    FISHEYE = 'fisheye'

# =================================================================================
# 1. RUN CONFIGURATION
# =================================================================================
# Define which task to run when executing main.py
# Options: 'PROCESS_DIR', 'ITERATIVE_REFINEMENT', 'ABLATE_HOMOGRAPHY', 'PROCESS_TRAIL', 'PROCESS_TIME_SERIES'
RUN_TASK = 'PROCESS_TIME_SERIES'

# Parameters for the time series processing task
TIME_SERIES_START_TIME = "12-00-00" # HH-MM-SS format
TIME_SERIES_END_TIME = "17-00-00"   # HH-MM-SS format

# Stride for time series processing. Processes every Nth image pair.
# stride=1: every pair (default), stride=2: every other pair, etc.
# Use higher values to speed up tests on long intervals.
STRIDE = 1

# =================================================================================
# 2. PATHS CONFIGURATION
# =================================================================================
# Set the directory containing the stereo image pairs (e.g., 'img-2025-05-23T12-26-02devID1.jpg')
IMAGE_DIR = "/home/omega-luler/Data/clouds/fisheye_250523_part2"

MANUAL_POINTS_PATH = "/home/omega-luler/tasks/clouds_level_research/data/trail/trail_points.npz"

# Directory to save log files from the time series analysis
LOG_DIR = "/home/omega-luler/tasks/clouds/logs/250523"


# =================================================================================
# 3. CAMERA & STEREO SYSTEM CONFIGURATION
# =================================================================================
# Specify the camera model to use for distance calculations
CAMERA_MODEL = CameraModel.FISHEYE

# Stereo baseline (distance between the two cameras in meters)
STEREO_BASE = 45.0  # meters

# Camera's angle of view in degrees (e.g., 180 for fisheye, 62 for pinhole)
ANGLE_OF_VIEW = 180.0  # degrees

# Original image width in pixels. Used in distance calculation.
IMAGE_WIDTH = 1920

# Affine transformation matrix for fisheye
AFFINE_MATRIX = np.array([[0.0014207881638163794, -0.9999989906799874, 1972.73673196935],
                          [0.9999989906799874, 0.0014207881638163794, 3.6708577729732]])


# M = np.array([[9.97682421e-01, 1.15133317e-02, 2.36960989e+02],
#              [1.06785471e-02, 9.97667996e-01, 2.48774002e+01]])
# For non fisheye

# =================================================================================
# 4. IMAGE PROCESSING CONFIGURATION
# =================================================================================
# The original resolution of the images before any processing
ORIGINAL_SIZE = (1920, 1920)

# The target size for images after initial cropping, before feature matching
# Downsizing can speed up processing significantly.
# Example: (ORIGINAL_SIZE[0] // 8, ORIGINAL_SIZE[1] // 8)
TARGET_SIZE = (ORIGINAL_SIZE[0] // 6, ORIGINAL_SIZE[1] // 6)

# =================================================================================
# 5. ALGORITHM HYPERPARAMETERS
# =================================================================================
# Feature matcher backend selection.
# Options: 'kornia_loftr' (default, uses kornia.feature.LoFTR),
#          'matchanything_eloftr' (Efficient LoFTR from MatchAnything)
MATCHER_BACKEND = 'kornia_loftr'

# Extra keyword arguments passed to the matcher backend factory.
# After running `python tools/setup_matchanything.py`, uncomment and use:
# For 'matchanything_eloftr': set 'third_party_dir' to the directory containing
# the MatchAnything source (i.e. the parent of MatchAnything/).
MATCHER_BACKEND_KWARGS = {
    'third_party_dir': 'third_party',
}
# LoFTR feature matcher confidence threshold
LOFTR_MATCH_THRESHOLD = 0.6

# HDBSCAN clustering configuration
HDBSCAN_CONFIG = {
    'min_cluster_size': 3,
    'min_samples': 3,
    'min_epsilon': 0.0,
    'cluster_selection_epsilon': 0.00,
    'allow_single_cluster': True,
}

# Post-match filtering configuration. Applied after the matcher returns keypoints.
# Set to None to disable filtering entirely (recommended default).
# For noisy data or longer time windows, enable lenient filtering:
#   {'y_thresh_px': 10.0, 'conf_thresh': None, 'mad_n': 3.0, 'cycle_consistency': False}
# Filter keys:
#   y_thresh_px (float): Max y-disparity in pixels (epipolar scanline constraint)
#   conf_thresh (float|None): Min match confidence score (matcher-specific ranges differ)
#   mad_n (float|None): Disparity MAD multiplier for statistical outlier removal
#   cycle_consistency (bool): Enable expensive bidirectional backward-forward check
#   cycle_pixel_thresh (float): Pixel tolerance for cycle-consistency reprojection check
MATCH_FILTER_CONFIG = None

# Parameters for the iterative refinement pipeline
ITERATIVE_REFINEMENT_PARAMS = {
    'iterations': 3
}

# Parameters for the homography ablation study
ABLATION_PARAMS = {
    'max_noise_theta': 15.0, # Max rotation noise in degrees
    'steps': 10
}

# =================================================================================
# 6. VISUALIZATION & LOGGING CONFIGURATION
# =================================================================================
# Master switch to enable/disable all visualizations (e.g., images, plots)
VISUALIZE = False

# Master switch to enable/disable text file logging for the time series task
LOG_RESULTS = True

# Granular control over specific visualizations (only active if VISUALIZE is True)
# For time-series, it's recommended to turn most of these off to avoid window spam
VISUALIZATION_CONTROLS = {
    'show_loftr_matches': False,
    'show_cluster_vectors': True,
    'show_clustered_matches_on_images': True,
    'plot_distance_histograms': False,
    'run_normality_tests': False, # Plots sub-histograms with normality stats
}