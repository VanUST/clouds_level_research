# /src/pipeline.py
import os
import cv2
import numpy as np
import math
import re
import datetime
from collections import Counter

from .camera import StereoCameraSystem
from .feature_processing import get_corresponding_feature_points, cluster_shifts
from .image_utils import center_crop
from . import visualization as viz

class PipelineRunner:
    """
    Orchestrates the different processing pipelines for stereo images.
    """
    def __init__(self, stereo_system: StereoCameraSystem, config):
        self.stereo_system = stereo_system
        self.config = config

    def _get_image_pairs(self):
        """Finds and pairs images ending in 1.jpg and 2.jpg in the directory."""
        image_pairs = {}
        directory = self.config.IMAGE_DIR
        for filename in os.listdir(directory):
            if filename.lower().endswith((".jpg", ".jpeg")):
                base_name, ext = os.path.splitext(filename)
                if base_name.endswith(('_1', '_2')):
                    pair_id = base_name[:-2]
                    img_num = int(base_name[-1]) - 1 # 0 for _1, 1 for _2
                    
                    if pair_id not in image_pairs:
                        image_pairs[pair_id] = [None, None]
                    
                    image_pairs[pair_id][img_num] = os.path.join(directory, filename)
        
        # Filter out incomplete pairs and return a sorted list
        valid_pairs = [tuple(paths) for paths in image_pairs.values() if all(paths)]
        return sorted(valid_pairs)

    def _analyze_stereo_pair(self, img1_path: str, img2_path: str, affine_matrix=None) -> dict | None:
        """
        Analyzes a single stereo pair and returns cluster info plus raw match data.

        Args:
            img1_path, img2_path: Paths to the stereo pair.
            affine_matrix: Optional 2x3 numpy array overriding config.AFFINE_MATRIX.

        Returns:
            dict with keys:
              - 'clusters': list[dict]  (cluster summaries as before)
              - 'kp1': np.ndarray      (N,2) matched keypoints in image1
              - 'kp2': np.ndarray      (N,2) matched keypoints in image2
              - 'shifts': np.ndarray   (N,2) disparity vectors
              - 'labels': np.ndarray   (N,)  cluster label per match (-1=noise)
              - 'centers': np.ndarray  (M,2) cluster centroids in shift space
            or None if insufficient data.
        """
        cfg = self.config
        
        # Load images
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        if img1_orig is None or img2_orig is None:
            print(f"Warning: Could not load image pair: {img1_path}, {img2_path}")
            return None

        # Preprocess: Apply affine transform and crop
        _affine = affine_matrix if affine_matrix is not None else cfg.AFFINE_MATRIX
        img1 = cv2.warpAffine(img1_orig, _affine, dsize=cfg.ORIGINAL_SIZE)
        img1 = center_crop(img1, cfg.TARGET_SIZE[0], cfg.TARGET_SIZE[1])
        img2 = center_crop(img2_orig, cfg.TARGET_SIZE[0], cfg.TARGET_SIZE[1])

        # Feature Matching
        backend_kwargs = getattr(cfg, 'MATCHER_BACKEND_KWARGS', {})
        filter_cfg = getattr(cfg, 'MATCH_FILTER_CONFIG', None)
        kp1, kp2 = get_corresponding_feature_points(
            img1, img2,
            visualize=cfg.VISUALIZE and cfg.VISUALIZATION_CONTROLS['show_loftr_matches'],
            match_thresh=cfg.LOFTR_MATCH_THRESHOLD,
            backend=cfg.MATCHER_BACKEND,
            filter_config=filter_cfg,
            **backend_kwargs
        )
        if len(kp1) < cfg.HDBSCAN_CONFIG['min_cluster_size']:
            return None

        # Calculate shifts and cluster them
        shifts = kp1 - kp2
        centers, std_devs, labels, colors = cluster_shifts(
            shift_arr=shifts,
            hdbscan_config=cfg.HDBSCAN_CONFIG,
            show_clusters=cfg.VISUALIZE and cfg.VISUALIZATION_CONTROLS['show_cluster_vectors']
        )
        
        # Optional Visualizations
        if cfg.VISUALIZE:
            if cfg.VISUALIZATION_CONTROLS['show_clustered_matches_on_images']:
                viz.visualize_clusters_on_images(img1, img2, kp1, kp2, labels, colors)
            if cfg.VISUALIZATION_CONTROLS['plot_distance_histograms']:
                viz.plot_distance_histogram_with_normality(
                    shifts, labels, colors, self.stereo_system,
                    run_normality_test=cfg.VISUALIZATION_CONTROLS['run_normality_tests']
                )

        # Build cluster summaries and raw data bundle
        cluster_list = []
        if len(centers) > 0:
            cluster_sizes = Counter(labels)
            valid_clusters = {label: count for label, count in cluster_sizes.items() if label >= 0}
            sorted_cluster_ids = sorted(valid_clusters.keys(), key=lambda k: valid_clusters[k], reverse=True)

            for i, cluster_id in enumerate(sorted_cluster_ids):
                disp_x = centers[i][0]
                std_dev_x = std_devs[i][0] if i < len(std_devs) else 0.0
                distance, error = self.stereo_system.compute_distance(disp_x, std_dev_x)
                mask = (labels == cluster_id)
                cluster_points_uv = kp1[mask]
                mean_u, mean_v = np.mean(cluster_points_uv, axis=0)
                cluster_list.append({
                    "cluster_id": int(cluster_id),
                    "size": valid_clusters[cluster_id],
                    "distance": distance,
                    "error": error,
                    "mean_u": mean_u,
                    "mean_v": mean_v,
                    "disp_x": disp_x
                })

        return {
            'clusters': cluster_list,
            'kp1': kp1,
            'kp2': kp2,
            'shifts': shifts,
            'labels': labels,
            'centers': np.array(centers) if len(centers) > 0 else np.empty((0, 2)),
        }

    def _process_single_pair(self, img1_path: str, img2_path: str) -> tuple[float | None, float | None]:
        """
        Processes one pair of images, prints results to console, and returns
        the distance and error for the largest cluster for backward compatibility.
        """
        result = self._analyze_stereo_pair(img1_path, img2_path)
        
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

    def process_directory(self):
        """Processes all image pairs in the configured directory."""
        image_pairs = self._get_image_pairs()
        if not image_pairs:
            print(f"No valid image pairs found in {self.config.IMAGE_DIR}")
            return
            
        print(f"Found {len(image_pairs)} image pairs to process.")
        for img1_path, img2_path in image_pairs:
            self._process_single_pair(img1_path, img2_path)
    
    def _get_time_filtered_pairs(self, start_str: str, end_str: str) -> list[tuple[str, str]]:
        """
        Finds and pairs images within a specific time window from the configured directory.
        """
        try:
            start_time = datetime.datetime.strptime(start_str, '%H-%M-%S').time()
            end_time = datetime.datetime.strptime(end_str, '%H-%M-%S').time()
        except ValueError:
            print(f"Error: Invalid time format in config.py. Please use HH-MM-SS.")
            return []

        directory = self.config.IMAGE_DIR
        pattern = re.compile(r'img-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})devID([12])')
        
        image_files = {}
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                timestamp_str, dev_id = match.groups()
                try:
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H-%M-%S')
                    if start_time <= timestamp.time() <= end_time:
                        base_name = f"img-{timestamp_str}"
                        if base_name not in image_files:
                            image_files[base_name] = [None, None]
                        
                        full_path = os.path.join(directory, filename)
                        image_files[base_name][int(dev_id) - 1] = full_path
                except ValueError:
                    continue 

        valid_pairs = [tuple(paths) for paths in image_files.values() if all(paths)]
        return sorted(valid_pairs)

    def _write_log_for_pair(self, result: dict, img1_path: str, log_dir: str):
        """Writes both a .txt summary and a .npz with raw keypoints/matches/clusters."""
        clusters = result.get('clusters', [])
        if not clusters:
            return

        base_filename = os.path.basename(img1_path)
        txt_name = base_filename.replace('devID1.jpg', '.txt')
        npz_name = base_filename.replace('devID1.jpg', '.npz')
        txt_path = os.path.join(log_dir, txt_name)
        npz_path = os.path.join(log_dir, npz_name)

        # --- Text log (summary, backward compatible) ---
        with open(txt_path, 'w') as f:
            f.write(f"Analysis for: {base_filename}\n")
            f.write(f"Total matches: {len(result['kp1'])}\n")
            f.write("-" * 35 + "\n")
            for i, res in enumerate(clusters):
                f.write(f"Cluster {res['cluster_id']} ({'Dominant' if i == 0 else ''}):\n")
                f.write(f"  Size: {res['size']} points\n")
                f.write(f"  Height (Distance): {res['distance']:.2f} meters\n")
                f.write(f"  Centroid UV: {res['mean_u']:.2f}, {res['mean_v']:.2f}\n")
                f.write(f"  Disparity X: {res['disp_x']:.4f}\n")
            f.write("-" * 35 + "\n")

        # --- NPZ log (raw data for reproducibility) ---
        np.savez_compressed(
            npz_path,
            kp1=result['kp1'],
            kp2=result['kp2'],
            shifts=result['shifts'],
            labels=result['labels'],
            centers=result['centers'],
        )
        print(f"  -> Logs saved: {txt_path}, {npz_path}")

    def _get_log_dir(self, backend: str = None) -> str:
        """Return the effective log directory, optionally with backend subdirectory."""
        base = self.config.LOG_DIR
        sub = backend or getattr(self.config, 'MATCHER_BACKEND', 'default')
        log_dir = os.path.join(base, sub)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def process_time_series(self, backend: str = None, affine_matrix=None):
        """
        Processes a time-ordered series of image pairs within a configured time window.
        
        Args:
            backend: Optional override for the matcher backend name. Used for log subdirectory.
            affine_matrix: Optional 2x3 numpy array overriding config.AFFINE_MATRIX.
        """
        cfg = self.config
        image_pairs = self._get_time_filtered_pairs(
            cfg.TIME_SERIES_START_TIME,
            cfg.TIME_SERIES_END_TIME
        )

        if not image_pairs:
            print(f"No valid image pairs found in {cfg.IMAGE_DIR} between "
                  f"{cfg.TIME_SERIES_START_TIME} and {cfg.TIME_SERIES_END_TIME}")
            return

        log_dir = self._get_log_dir(backend)
        print(f"Logging results to: {log_dir}")

        stride = max(1, int(getattr(cfg, 'STRIDE', 1)))
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

            result = self._analyze_stereo_pair(img1_path, img2_path, affine_matrix=affine_matrix)
            
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

            if cfg.LOG_RESULTS and result is not None:
                self._write_log_for_pair(result, img1_path, log_dir)

        total_skipped = skipped_no_clusters + skipped_no_height
        print(f"\n--- Time Series Complete ---")
        print(f"  Total frames processed: {len(image_pairs)}")
        print(f"  Skipped (no clusters/too few matches): {skipped_no_clusters}")
        print(f"  Skipped (clusters but no valid height): {skipped_no_height}")
        print(f"  Total skipped: {total_skipped}")
        print(f"  Valid frames with height: {len(image_pairs) - total_skipped}")

    def _process_trail_pair(self):
        """
        Processes a single pair of images using manually selected points for an 
        aircraft trail, calculating a distance for each point to see the distribution.
        """
        cfg = self.config
        
        # 1. Load the manually matched points
        try:
            data = np.load(cfg.MANUAL_POINTS_PATH)
            kp1 = data['kp1']
            kp2 = data['kp2']
            print(f"Loaded {len(kp1)} manually matched points from {cfg.MANUAL_POINTS_PATH}")
        except FileNotFoundError:
            print(f"Error: Manual points file not found at '{cfg.MANUAL_POINTS_PATH}'")
            print("Please run 'tools/trail_matcher.py' first to generate the points file.")
            return

        # 2. Calculate shifts (disparity) for each point
        shifts = kp1 - kp2
        
        # 3. Calculate distance for each point
        distances = []
        for shift_vector in shifts:
            disp_x = shift_vector[0]
            if abs(disp_x) > 1e-6: # Avoid division by zero
                distance, _ = self.stereo_system.compute_distance(disp_x, delta_px_error=0)
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
        if cfg.VISUALIZE:
            viz.plot_trail_distance_distribution(distances, kp1)
    
    def run_homography_ablation(self):
        """Runs an ablation study by injecting noise into the affine matrix rotation."""
        cfg = self.config
        image_pairs = self._get_image_pairs()
        if not image_pairs:
            print("No image pairs found for ablation study.")
            return

        img1_path, img2_path = image_pairs[0] # Use the first pair for the study
        print(f"Running ablation study on: {os.path.basename(img1_path)}")

        theta_vals, dist_vals, err_vals = [], [], []
        
        original_M = cfg.AFFINE_MATRIX.copy()

        for i in range(cfg.ABLATION_PARAMS['steps']):
            theta = cfg.ABLATION_PARAMS['max_noise_theta'] * i / (cfg.ABLATION_PARAMS['steps']-1)
            print(f"\n--- Testing with theta noise: {theta:.2f} degrees ---")

            theta_rad = math.radians(theta)
            rotation_noise = np.array([
                [math.cos(theta_rad), -math.sin(theta_rad)],
                [math.sin(theta_rad), math.cos(theta_rad)]
            ])

            M_rot = original_M[:, :2] @ rotation_noise
            cfg.AFFINE_MATRIX[:, :2] = M_rot
            
            distance, error = self._process_single_pair(img1_path, img2_path)

            if distance is not None:
                theta_vals.append(theta)
                dist_vals.append(distance)
                err_vals.append(error)

        cfg.AFFINE_MATRIX = original_M
        
        if cfg.VISUALIZE and theta_vals:
            viz.plot_ablation_results(theta_vals, dist_vals, err_vals)
            
    def run_iterative_refinement(self):
        print("Iterative refinement pipeline is not fully implemented in this refactoring yet.")
        pass