# /src/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from .camera import StereoCameraSystem

def visualize_loftr_matches(img1, img2, kp1, kp2, max_matches_to_draw=200):
    """Draws lines between matched keypoints on a combined image."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1
    combined_image[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2

    # Draw a random subset of matches if there are too many
    indices = np.random.choice(len(kp1), min(len(kp1), max_matches_to_draw), replace=False)
    for i in indices:
        pt1 = tuple(map(int, kp1[i]))
        pt2 = tuple(map(int, (kp2[i][0] + w1, kp2[i][1])))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(combined_image, pt1, pt2, color, 1)
        cv2.circle(combined_image, pt1, 2, color, -1)
        cv2.circle(combined_image, pt2, 2, color, -1)
    
    cv2.namedWindow('LoFTR Feature Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('LoFTR Feature Matches', combined_image)
    cv2.waitKey(1) # Use waitKey(1) for non-blocking display

def visualize_clusters_on_images(img1, img2, kp1, kp2, labels, colors):
    """Visualizes the spatial location of clustered feature points on the images."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1
    combined_image[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2
    
    for idx, (pt1, pt2) in enumerate(zip(kp1, kp2)):
        label = labels[idx]
        color_rgb_01 = colors.get(label, (0.5, 0.5, 0.5))
        color_bgr_255 = tuple(int(c * 255) for c in color_rgb_01[::-1]) # Convert to BGR for OpenCV
        
        cv2.circle(combined_image, tuple(map(int, pt1)), 3, color_bgr_255, -1)
        pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
        cv2.circle(combined_image, pt2_shifted, 3, color_bgr_255, -1)
        
    cv2.namedWindow('Clustered Feature Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Clustered Feature Matches', combined_image)
    cv2.waitKey(1)

def plot_shift_vectors(X, labels, probabilities, colors, params):
    """Plots the 2D shift vectors colored by cluster label."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    for label in set(labels):
        mask = (labels == label)
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[label]], label=f'Cluster {label}' if label != -1 else 'Noise', s=30, alpha=0.7)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title(f"Shift Vector Clustering (Estimated clusters: {n_clusters})\nParams: {params}")
    plt.xlabel("X Displacement (pixels)")
    plt.ylabel("Y Displacement (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_distance_histogram_with_normality(shift_arr, labels, colors, stereo_system: StereoCameraSystem, bins=30, run_normality_test=True):
    """Plots distance histograms for each cluster with optional normality analysis."""
    distances = np.array([stereo_system.compute_distance(s[0], 0)[0] for s in shift_arr if s[0] != 0])
    labels_filtered = np.array([labels[i] for i, s in enumerate(shift_arr) if s[0] != 0])
    
    unique_labels = np.unique(labels_filtered)
    
    # Combined Histogram
    plt.figure(figsize=(12, 7))
    for label in unique_labels:
        cluster_distances = distances[labels_filtered == label]
        label_name = f'Cluster {label}' if label != -1 else 'Noise'
        plt.hist(cluster_distances, bins=bins, alpha=0.6, color=colors.get(label), label=f'{label_name} (n={len(cluster_distances)})')
    plt.xlabel("Calculated Distance (meters)")
    plt.ylabel("Frequency")
    plt.title("Combined Distance Histogram from All Feature Matches")
    plt.legend()
    plt.show()

    # Per-cluster subplots with normality test
    if run_normality_test:
        valid_labels = [l for l in unique_labels if l != -1]
        if not valid_labels: return
        
        n_plots = len(valid_labels)
        fig, axs = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), squeeze=False)
        
        for i, label in enumerate(valid_labels):
            ax = axs[i, 0]
            cluster_distances = distances[labels_filtered == label]
            mu, std = np.mean(cluster_distances), np.std(cluster_distances)
            
            ax.hist(cluster_distances, bins=bins, density=True, alpha=0.6, color=colors.get(label), label=f'Cluster {label}')
            
            # Fit and plot normal distribution
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2, label='Normal Fit')
            
            # Shapiro-Wilk normality test
            test_result = "Not enough data"
            if len(cluster_distances) >= 3:
                stat, p_value = stats.shapiro(cluster_distances)
                test_result = f"Shapiro-Wilk: W={stat:.3f}, p={p_value:.3f}"
            
            ax.set_title(f"Cluster {label} | Mean={mu:.2f}, Std={std:.2f}\n{test_result}")
            ax.set_xlabel("Distance (meters)")
            ax.set_ylabel("Density")
            ax.legend()
        
        plt.tight_layout()
        plt.show()

def plot_ablation_results(theta_vals, dist_vals, err_vals):
    """Plots the results of the homography ablation study."""
    plt.figure(figsize=(10, 6))
    plt.errorbar(theta_vals, dist_vals, yerr=err_vals, fmt='-o', capsize=5)
    plt.xlabel("Rotation Noise Angle θ (°)")
    plt.ylabel("Calculated Distance (meters)")
    plt.title("Calculated Distance vs. Injected Homography Noise")
    plt.grid(True)
    plt.show()

def plot_distance_convergence(refinement_results):
    """Plots distance convergence from iterative refinement."""
    plt.figure(figsize=(12, 7))
    for idx, results in enumerate(refinement_results):
        iterations = list(range(1, len(results) + 1))
        distances = [r[3] for r in results]
        errors = [r[4] for r in results]
        plt.errorbar(iterations, distances, yerr=errors, marker='o', linestyle='-', label=f'Pair {idx+1}', capsize=4)

    plt.xlabel('Iteration')
    plt.ylabel('Distance (meters)')
    plt.title('Convergence of Stereo Distance with Error Bars')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_trail_distance_distribution(distances: np.ndarray, keypoints: np.ndarray):
    """
    Plots the distribution of calculated distances for a contrail.

    Args:
        distances (np.ndarray): Array of calculated distances.
        keypoints (np.ndarray): The source keypoints (kp1) used for the calculation.
    """
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Histogram of distances
    ax1.hist(distances, bins=30, alpha=0.75, color='skyblue', edgecolor='black')
    ax1.axvline(mean_dist, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dist:.2f}m')
    ax1.set_title(f'Distance Distribution for Aircraft Trail\n(Mean: {mean_dist:.2f}m, Std: {std_dist:.2f}m)')
    ax1.set_xlabel('Calculated Distance (meters)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Scatter plot of distance vs. vertical image position
    # This helps check for systematic errors (e.g., due to lens distortion)
    y_positions = keypoints[:, 1]
    ax2.scatter(y_positions, distances, alpha=0.6)
    ax2.set_title('Distance vs. Vertical Pixel Position')
    ax2.set_xlabel('Y-coordinate in Image 1 (pixels)')
    ax2.set_ylabel('Calculated Distance (meters)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
