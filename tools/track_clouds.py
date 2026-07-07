# /tools/track_clouds.py
import os
import sys
import re
import math
import numpy as np
import datetime
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D

# Add the project root to the Python path to allow importing 'config' and 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.image_utils import center_crop

# --- CONFIG ---
LOG_DIR = "/home/omega-luler/tasks/clouds_level_research/logs/250523"
GIF_OUTPUT_PATH = "/home/omega-luler/tasks/clouds_level_research/cloud_dynamics.gif"

# --- NEW FILTERING CONFIG ---
MIN_POINTS_PER_CLUSTER = 30          # Do not track clusters with fewer points than this
MAX_SPEED_METERS_PER_MINUTE = 100.0 # Max meters a cloud can move per minute

# --- DYNAMIC VISUALIZATION CONFIG ---
M_TOP_TRACKS = 5           # The number of most consistent tracks to show at any given time
TRAIL_LENGTH_FRAMES = 10   # How many historical frames to show behind the cloud (comet tail). Set to 0 for infinite.
GIF_FPS = 2                # Frames per second for the output GIF
MAX_FRAMES_PER_GIF = 120    # Maximum number of frames per GIF file. Will split into multiple parts if exceeded.

# --- TRACKING CONFIG ---
MAX_TRACK_AGE = 5  # Frames to remember a lost cloud before killing the track
MIN_TRACK_LENGTH = 5  # Minimum frames a track must exist to be plotted

# Sync camera parameters from config
TARGET_WIDTH = config.TARGET_SIZE[0]
TARGET_HEIGHT = config.TARGET_SIZE[1]
AOV_DEGREES = config.ANGLE_OF_VIEW

def pixel_to_3d(u, v, z):
    """
    Approximates physical X, Y coordinates from pixel u, v and depth Z.
    Using a simplified spherical projection for Fisheye.
    """
    cx, cy = TARGET_WIDTH / 2, TARGET_HEIGHT / 2
    r_px = math.sqrt((u - cx)**2 + (v - cy)**2)
    
    if r_px == 0:
        return 0.0, 0.0, z
        
    f_px = TARGET_WIDTH / math.radians(AOV_DEGREES)
    theta = r_px / f_px
    r_physical = z * math.tan(theta)
    
    x = r_physical * ((u - cx) / r_px)
    y = r_physical * ((v - cy) / r_px)
    return x, y, z

class Track:
    # Now storing u and v to plot on the 2D image
    def __init__(self, track_id, timestamp, x, y, z, size, u, v):
        self.track_id = track_id
        self.history = [(timestamp, x, y, z, size, u, v)]
        self.age = 0  # Frames since last update
    
    def update(self, timestamp, x, y, z, size, u, v):
        self.history.append((timestamp, x, y, z, size, u, v))
        self.age = 0
        
    def get_last_pos(self):
        return self.history[-1][1:4]
        
    def get_last_timestamp(self):
        return self.history[-1][0]

def parse_enriched_logs(log_dir):
    """Parses logs including the new UV centroid data."""
    frames_data = {}
    file_pattern = re.compile(r'img-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.txt')
    
    # Matches the exact format printed by your updated _write_log_for_pair
    cluster_pattern = re.compile(
        r"Size: (\d+) points\n"
        r"\s*- Height \(Distance\): ([\d.]+) meters\n"
        r"\s*- Centroid UV: ([\d.]+), ([\d.]+)"
    )

    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist.")
        return frames_data

    for filename in os.listdir(log_dir):
        file_match = file_pattern.match(filename)
        if not file_match:
            continue
            
        timestamp_str = file_match.group(1)
        try:
            timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
        except ValueError:
            continue

        with open(os.path.join(log_dir, filename), 'r') as f:
            content = f.read()

        clusters = []
        for match in cluster_pattern.finditer(content):
            clusters.append({
                'size': int(match.group(1)),
                'z': float(match.group(2)),
                'u': float(match.group(3)),
                'v': float(match.group(4))
            })

        if clusters:
            frames_data[timestamp] = clusters

    return frames_data

def run_tracker(frames_data):
    """Tracks clouds across chronological frames."""
    tracks = []
    next_track_id = 0
    
    for timestamp, clusters in sorted(frames_data.items()):
        current_measurements = []
        for c in clusters:
            # 1. Filter out minor clusters based on point count
            if c['size'] < MIN_POINTS_PER_CLUSTER:
                continue
                
            x, y, z = pixel_to_3d(c['u'], c['v'], c['z'])
            current_measurements.append({'x': x, 'y': y, 'z': z, 'size': c['size'], 'u': c['u'], 'v': c['v']})
            
        active_tracks = [t for t in tracks if t.age <= MAX_TRACK_AGE]
        
        if not active_tracks:
            # First frame or all tracks lost, initialize new tracks
            for m in current_measurements:
                tracks.append(Track(next_track_id, timestamp, m['x'], m['y'], m['z'], m['size'], m['u'], m['v']))
                next_track_id += 1
            continue
            
        # Build cost matrix (Euclidean 3D distance)
        cost_matrix = np.zeros((len(active_tracks), len(current_measurements)))
        for i, track in enumerate(active_tracks):
            tx, ty, tz = track.get_last_pos()
            last_time = track.get_last_timestamp()
            
            # 2. Calculate dynamic distance threshold
            time_diff_minutes = (timestamp - last_time).total_seconds() / 60.0
            
            # Edge case safeguard: Handle exact same timestamps or negative time leaps
            if time_diff_minutes <= 0:
                time_diff_minutes = 1e-5 
                
            dynamic_max_dist = MAX_SPEED_METERS_PER_MINUTE * time_diff_minutes
            
            for j, m in enumerate(current_measurements):
                dist = math.sqrt((tx - m['x'])**2 + (ty - m['y'])**2 + (tz - m['z'])**2)
                
                # Reject matches that violate the maximum speed limit
                if dist > dynamic_max_dist:
                    cost_matrix[i, j] = 1e9  # Effectively infinite cost
                else:
                    cost_matrix[i, j] = dist
                
        # Assign using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_measurements = set()
        for i, j in zip(row_ind, col_ind):
            # Ensure the assignment was within our dynamic speed limit threshold
            if cost_matrix[i, j] < 1e9:
                m = current_measurements[j]
                active_tracks[i].update(timestamp, m['x'], m['y'], m['z'], m['size'], m['u'], m['v'])
                assigned_measurements.add(j)
                
        # Age unassigned tracks
        for i, track in enumerate(active_tracks):
            if i not in row_ind or cost_matrix[i, col_ind[list(row_ind).index(i)]] >= 1e9:
                track.age += 1
                
        # Create new tracks for unassigned measurements
        for j, m in enumerate(current_measurements):
            if j not in assigned_measurements:
                tracks.append(Track(next_track_id, timestamp, m['x'], m['y'], m['z'], m['size'], m['u'], m['v']))
                next_track_id += 1
                
    return [t for t in tracks if len(t.history) >= MIN_TRACK_LENGTH]

def generate_animated_gif(tracks, frames_data, output_path):
    """
    Generates side-by-side GIFs visualizing the Top M tracks in 3D and overlaid on the source image.
    Splits into multiple parts if the frame count exceeds MAX_FRAMES_PER_GIF.
    """
    if not tracks:
        print("No tracks available to animate.")
        return

    # Extract global timestamps
    timestamps = sorted(list(frames_data.keys()))
    total_frames = len(timestamps)
    
    # Establish fixed bounds for the 3D plot so the camera doesn't jump
    all_x, all_y, all_z = [], [], []
    for t in tracks:
        for _, x, y, z, _, _, _ in t.history:
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            
    # Edge case: If valid tracks exist but coordinates are perfectly identical (zero variance)
    pad = 100 
    min_x, max_x = min(all_x) - pad, max(all_x) + pad
    min_y, max_y = min(all_y) - pad, max(all_y) + pad
    min_z, max_z = min(all_z) - pad, max(all_z) + pad

    # Use a consistent color map for tracks so colors don't swap dynamically
    colormap = plt.get_cmap('tab20')

    # Determine chunking strategy
    chunk_size = MAX_FRAMES_PER_GIF if MAX_FRAMES_PER_GIF > 0 else total_frames
    num_chunks = math.ceil(total_frames / chunk_size)
    
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    ext = os.path.splitext(output_path)[1]
    
    os.makedirs(base_dir, exist_ok=True)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_timestamps = timestamps[start_idx:end_idx]
        
        chunk_filename = f"{base_name}_part_{chunk_idx + 1:02d}{ext}"
        chunk_path = os.path.join(base_dir, chunk_filename)
        
        # Set up the figure and axis for side-by-side plotting
        fig = plt.figure(figsize=(16, 8))
        ax_img = fig.add_subplot(121)
        ax_3d = fig.add_subplot(122, projection='3d')
        
        def update(frame_idx):
            ax_img.clear()
            ax_3d.clear()
            
            current_time = chunk_timestamps[frame_idx]
            
            # --- 1. Set up the 2D Image Subplot ---
            img_name = f"img-{current_time.strftime('%Y-%m-%dT%H-%M-%S')}devID1.jpg"
            img_path = os.path.join(config.IMAGE_DIR, img_name)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Apply the exact same preprocessing pipeline from your code
                    img = cv2.warpAffine(img, config.AFFINE_MATRIX, dsize=config.ORIGINAL_SIZE)
                    img = center_crop(img, config.TARGET_SIZE[0], config.TARGET_SIZE[1])
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax_img.imshow(img_rgb)
            
            ax_img.set_title(f"Stereo Source (devID1)\nTime: {current_time.strftime('%H:%M:%S')}", fontsize=14)
            ax_img.axis('off')
            
            # --- 2. Set up the 3D Tracking Subplot ---
            ax_3d.set_xlim(min_x, max_x)
            ax_3d.set_ylim(min_y, max_y)
            ax_3d.set_zlim(min_z, max_z)
            
            ax_3d.set_xlabel('X (meters)')
            ax_3d.set_ylabel('Y (meters)')
            ax_3d.set_zlabel('Z / Height (meters)')
            ax_3d.set_title(f"Dynamic Top {M_TOP_TRACKS} Tracks", fontsize=14)
            
            # --- 3. Gather Active Tracks ---
            active_tracks = []
            for t in tracks:
                start_time = t.history[0][0]
                end_time = t.history[-1][0]
                if start_time <= current_time <= end_time:
                    active_tracks.append(t)
                    
            # Sort by consistency (total length of the track's life)
            active_tracks.sort(key=lambda t: len(t.history), reverse=True)
            
            # Take Top M
            top_m = active_tracks[:M_TOP_TRACKS]
            
            # --- 4. Plot ---
            for t in top_m:
                # Extract history strictly up to the current frame time (seamless across chunks)
                past_history = [pt for pt in t.history if pt[0] <= current_time]
                if not past_history:
                    continue
                    
                color = colormap(t.track_id % 20)
                
                # -> 4A. Plot on 2D Image if the track was explicitly observed in this exact frame
                current_pt = next((pt for pt in past_history if pt[0] == current_time), None)
                if current_pt:
                    u, v = current_pt[5], current_pt[6]
                    ax_img.scatter(u, v, color=color, s=120, edgecolors='white', linewidth=2)
                    ax_img.text(u + 10, v - 10, f'ID:{t.track_id}', color='white', fontsize=10, weight='bold', 
                                bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=2))

                # -> 4B. Plot on 3D axes (with trailing comet tail)
                if TRAIL_LENGTH_FRAMES > 0:
                    past_history = past_history[-TRAIL_LENGTH_FRAMES:]
                    
                xs = [pt[1] for pt in past_history]
                ys = [pt[2] for pt in past_history]
                zs = [pt[3] for pt in past_history]
                
                # Plot the comet tail
                ax_3d.plot(xs, ys, zs, color=color, alpha=0.7, linewidth=2, label=f'Track {t.track_id}')
                # Plot the head of the comet
                ax_3d.scatter(xs[-1], ys[-1], zs[-1], color=color, s=50, edgecolors='k')

            if top_m:
                ax_3d.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        print(f"Generating GIF Part {chunk_idx + 1}/{num_chunks} ({len(chunk_timestamps)} frames)...")
        
        ani = animation.FuncAnimation(fig, update, frames=len(chunk_timestamps), interval=1000/GIF_FPS)
        writer = animation.PillowWriter(fps=GIF_FPS)
        
        ani.save(chunk_path, writer=writer)
        print(f"Successfully saved to: {chunk_path}")
        
        # Clear figure memory before next loop iteration
        plt.close(fig)

# --- Execution Block ---
if __name__ == "__main__":
    print(f"Parsing enriched logs from: {LOG_DIR}")
    frames = parse_enriched_logs(LOG_DIR)
    
    if not frames:
        print("No valid log data found. Did you re-run the pipeline with the enriched logging?")
    else:
        print(f"Loaded data for {len(frames)} frames. Running Hungarian Tracker...")
        active_tracks = run_tracker(frames)
        
        print(f"Tracking complete. Found {len(active_tracks)} valid cloud trajectories.")
        
        # Generate chunked animated GIFs
        generate_animated_gif(active_tracks, frames, GIF_OUTPUT_PATH)