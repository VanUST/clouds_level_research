import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
import collections
import torch
import kornia as K

from romatch import roma_outdoor

def get_corresponding_feature_points_roma(img1_path, img2_path,target_size):
    roma_model = roma_outdoor(device='cpu')
    # Match
    warp, certainty = roma_model.match(img1_path, img2_path, device='cpu')
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
    H_A = H_B = target_size[1]
    W_A = W_B = target_size[0]
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    return kptsA, kptsB

def get_corresponding_feature_points_loftr(img1, img2, vizualize = False,match_tresh = 0.5):
    
    torch.cuda.empty_cache()
    
    default_cfg = {
    "backbone_type": "ResNetFPN",
    "resolution": (8, 2),
    "fine_window_size": 5,
    "fine_concat_coarse_feat": True,
    "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
    "coarse": {
        "d_model": 256,
        "d_ffn": 256,
        "nhead": 8,
        "layer_names": ["self", "cross", "self", "cross", "self", "cross", "self", "cross"],
        "attention": "linear",
        "temp_bug_fix": False,
    },
    "match_coarse": {
        "thr": match_tresh,
        "border_rm": 2,
        "match_type": "dual_softmax",
        "dsmax_temperature": 0.1,
        "skh_iters": 3,
        "skh_init_bin_score": 1.0,
        "skh_prefilter": True,
        "train_coarse_percent": 0.4,
        "train_pad_num_gt_min": 200,
    },
    "fine": {"d_model": 128, "d_ffn": 128, "nhead": 8, "layer_names": ["self", "cross"], "attention": "linear"},
    }
    
    # Convert images to grayscale if they are not already
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    # Convert images to tensors and normalize pixel values
    img1_tensor = K.image_to_tensor(img1_gray, False).float() / 255.0  # Shape: [1, 1, H, W]
    img2_tensor = K.image_to_tensor(img2_gray, False).float() / 255.0

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    # Initialize LoFTR matcher with pretrained weights ('outdoor' or 'indoor')
    matcher = K.feature.LoFTR(pretrained='outdoor',config=default_cfg).to(device)

    

    # Prepare input dictionary
    input_dict = {"image0": img1_tensor, "image1": img2_tensor}

    # Perform matching
    with torch.no_grad():
        correspondences = matcher(input_dict)

    # Extract matched keypoints coordinates
    keypoints0 = correspondences['keypoints0'].cpu().numpy()
    keypoints1 = correspondences['keypoints1'].cpu().numpy()
    
    
    if vizualize:
        # Visualize matches
        # Convert images back to BGR for visualization if necessary
        if len(img1.shape) == 2:
            img1_viz = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_viz = img1.copy()
        if len(img2.shape) == 2:
            img2_viz = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_viz = img2.copy()

        # Create a combined image for visualization
        h1, w1 = img1_viz.shape[:2]
        h2, w2 = img2_viz.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        combined_image = np.zeros((height, width, 3), dtype=np.uint8)
        combined_image[:h1, :w1] = img1_viz
        combined_image[:h2, w1:w1+w2] = img2_viz

        # Draw matches
        # Randomly select a subset of matches to display if there are too many
        num_matches = keypoints0.shape[0]
        max_matches_to_draw = 500
        if num_matches > max_matches_to_draw:
            indices = np.random.choice(num_matches, max_matches_to_draw, replace=False)
            keypoints0_to_draw = keypoints0[indices]
            keypoints1_to_draw = keypoints1[indices]
        else:
            keypoints0_to_draw = keypoints0
            keypoints1_to_draw = keypoints1

        # Draw lines between matching keypoints
        for (pt1, pt2) in zip(keypoints0_to_draw, keypoints1_to_draw):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0] + w1), int(pt2[1])  # Offset x2 by width of img1
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(combined_image, (x1, y1), 2, color, -1)
            cv2.circle(combined_image, (x2, y2), 2, color, -1)
            cv2.line(combined_image, (x1, y1), (x2, y2), color, 1)

        # Display the image
        cv2.imshow('LoFTR Feature Matches', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return keypoints0, keypoints1

def get_corresponding_feature_points(img1, img2):
     
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints in both images
    keypoints1 = orb.detect(img1, None)
    keypoints2 = orb.detect(img2, None)

    # Initialize TEBLID descriptor extractor
    teblid = cv2.xfeatures2d.TEBLID_create(scale_factor = 1.)

    # Compute TEBLID descriptors based on the keypoints detected by ORB
    keypoints1, descriptors1 = teblid.compute(img1, keypoints1)
    keypoints2, descriptors2 = teblid.compute(img2, keypoints2)

    # Use BFMatcher with Hamming distance for binary descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Find initial matches using k-NN with k=2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply the ratio test to filter good matches
    ratio_thresh = 0.75
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Get image sizes (width, height)
    size1 = (img1.shape[1], img1.shape[0])
    size2 = (img2.shape[1], img2.shape[0])

    # Apply GMS matcher to refine matches
    matchesGMS = cv2.xfeatures2d.matchGMS(
        size1, size2, keypoints1, keypoints2, good_matches, withRotation=False, withScale=False
    )

    # Extract matched keypoints coordinates
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matchesGMS])
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matchesGMS])

    return pts1, pts2

def apply_affine_matrix(kp_array, matrix):
     processed_kp = np.zeros_like(kp_array)
     for kp,kp_processed in zip(kp_array,processed_kp):
        kp_temp = np.array([kp[0],kp[1], 1])
        kp_temp = matrix@kp_temp
        #print(kp_temp)
        processed_kp[0] = kp_temp[0]
        processed_kp[1] = kp_temp[1]
     return processed_kp

def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None, colors=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    labels = labels if labels is not None else np.ones(X.shape[0])
    probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])

    # Use the provided colors or generate them if not provided
    if colors is None:
        colors = get_cluster_colors(labels)

    # Map probabilities for marker sizes
    proba_map = {idx: probabilities[idx] for idx in range(len(labels))}

    for label in set(labels):
        if label == -1:
            col = (0, 0, 0)
        else:
            col = colors[label]
        class_index = np.where(labels == label)[0]
        for ci in class_index:
            ax.plot(
                X[ci, 0],
                X[ci, 1],
                "x" if label == -1 else "o",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=4 if label == -1 else 1 + 5 * proba_map[ci],
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "True" if ground_truth else "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f" | {parameters_str}"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def cluster_vecs(shift_arr, hdbscan_config, show_clusters=False):
    # Initialize the HDBSCAN object with the provided configurations
    hdbscan = HDBSCAN(
        min_cluster_size=hdbscan_config["min_cluster_size"],
        min_samples=hdbscan_config["min_samples"],
        cluster_selection_epsilon=hdbscan_config["epsilon"],
        allow_single_cluster=True,
        store_centers= 'centroid'
    )

    # Handle case when shift_arr is too small
    if len(shift_arr) < hdbscan_config["min_samples"]:
        labels = np.full(len(shift_arr), -1, dtype=int)
        colors = get_cluster_colors(labels)
        return [], [], labels, colors  # Empty centroids, std_devs, labels, colors

    # Fit the model
    hdbscan.fit(shift_arr)

    # Get the labels
    labels = hdbscan.labels_

    # Generate colors
    colors = get_cluster_colors(labels)

    # Plot the clusters if requested
    if show_clusters:
        plot(shift_arr, labels, hdbscan.probabilities_, colors=colors)

    # Count the number of points in each cluster
    power_of_cluster = collections.Counter(labels)

    # Filter out the noise cluster, which has label -1
    valid_cluster_indices = [key for key in power_of_cluster.keys() if int(key) >= 0]

    if len(valid_cluster_indices) == 0:
        return [], [], labels, colors  # Return empty centroids and std_devs, but return labels and colors

    # Extract the centroids of valid clusters
    centroids = hdbscan.centroids_

    # Gather centroids and their corresponding cluster sizes
    good_centroids = []
    for ind in valid_cluster_indices:
        good_centroids.append([power_of_cluster[ind], centroids[ind]])

    # Sort centroids by the size of the cluster, in descending order
    good_sorted = sorted(good_centroids, key=lambda x: x[0], reverse=True)
    sorted_centroids = [x[1] for x in good_sorted]

    # Calculate the standard deviation for each cluster
    cluster_standard_deviations = []
    for cluster_label in valid_cluster_indices:
        cluster_points = shift_arr[labels == cluster_label]
        cluster_std_dev = np.std(cluster_points, axis=0)
        cluster_standard_deviations.append(cluster_std_dev)

    return sorted_centroids, cluster_standard_deviations, labels, colors

def visualize_clusters_on_images(img1, img2, kp1, kp2, labels, colors):
    # Convert images to color if they are grayscale
    if len(img1.shape) == 2:
        img1_viz = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_viz = img1.copy()
    if len(img2.shape) == 2:
        img2_viz = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_viz = img2.copy()

    # Create a combined image for visualization
    h1, w1 = img1_viz.shape[:2]
    h2, w2 = img2_viz.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = img1_viz
    combined_image[:h2, w1:w1 + w2] = img2_viz

    # Draw matches
    for idx, (pt1, pt2) in enumerate(zip(kp1, kp2)):
        label = labels[idx]
        color_rgb = colors.get(label, (0, 0, 0))  # RGB in 0-1 range
        # Convert RGB to BGR 0-255 for OpenCV
        color_bgr = [int(255 * color_rgb[2]), int(255 * color_rgb[1]), int(255 * color_rgb[0])]
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0] + w1), int(pt2[1])  # Offset x2 by width of img1
        color = tuple(color_bgr)
        cv2.circle(combined_image, (x1, y1), 2, color, -1)
        cv2.circle(combined_image, (x2, y2), 2, color, -1)
        # cv2.line(combined_image, (x1, y1), (x2, y2), color, 1)

    # Display the image
    cv2.imshow('Clustered Feature Matches', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def get_cluster_colors(labels):
    unique_labels = set(labels)
    unique_labels_list = sorted(unique_labels)
    n_clusters = len(unique_labels_list)
    colors = {}
    colormap = plt.cm.get_cmap('rainbow', n_clusters)
    for idx, label in enumerate(unique_labels_list):
        if label == -1:
            colors[label] = (0, 0, 0)  # Black for noise
        else:
            rgb = colormap(idx)[:3]  # RGB in 0-1 range
            colors[label] = rgb
    return colors