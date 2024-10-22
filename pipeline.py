import os
import math
import numpy as np
import cv2
from feature_matching import get_corresponding_feature_points_loftr, cluster_vecs, apply_affine_matrix, visualize_clusters_on_images

def show_image(img):
    cv2.namedWindow("image", flags=cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def center_crop(image, crop_width, crop_height):
    height, width = image.shape[:2]
    start_x = max(0, (width - crop_width) // 2)
    start_y = max(0, (height - crop_height) // 2)
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_image

def count_dist_error(base, angle_of_view, img_width, delta_px, delta_px_error):
    # Convert angle_of_view to radians
    angle_in_radians = angle_of_view * 2 * math.pi / (2 * 360)
    
    # Calculate the original distance
    count_dist_value = base * img_width / (2 * math.tan(angle_in_radians) * abs(delta_px))
    
    # Calculate the error in distance
    error = (base * img_width)* delta_px_error / (2 * math.tan(angle_in_radians) * delta_px**2) 
    
    return count_dist_value, error

def count_dist(base,angle_of_view,img_width,delta_px):
    return base*img_width/(2*math.tan(angle_of_view*2*math.pi/(2*360))*abs(delta_px))

def find_distance_by_image_pair(img1_path, img2_path, M, base, angle_of_view, orig_size, target_size, hdbscan_config, match_tresh=0.5):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Apply affine transformation to img1
    img1 = cv2.warpAffine(img1, M, dsize=orig_size)

    # Center crop both images
    img1 = center_crop(img1, target_size[0], target_size[1])
    img2 = center_crop(img2, target_size[0], target_size[1])

    # Get corresponding feature points
    kp1, kp2 = get_corresponding_feature_points_loftr(img1, img2, vizualize=False, match_tresh=match_tresh)

    # Calculate shift array and cluster centers
    shift_arr = kp1 - kp2
    centers, std_devs, labels, colors = cluster_vecs(shift_arr=shift_arr, hdbscan_config=hdbscan_config, show_clusters=True)

    # Visualize clusters on images
    visualize_clusters_on_images(img1, img2, kp1, kp2, labels, colors)

    # Calculate and print distances for each cluster center
    for center, std_dev in zip(centers, std_devs):
        distance, error = count_dist_error(base, angle_of_view, width, center[0], std_dev)
        print(f"Distance for {img1_path}: {distance} +- {error[0]}")
        
def process_image_pairs(directory, base, angle_of_view, orig_size, target_size, M, hdbscan_config, match_tresh):
    image_pairs = {}

    # Collect image pairs based on their filenames
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            name, ext = os.path.splitext(filename)
            base_name = name[:-2]
            pair_number = name[-1]
            if base_name not in image_pairs:
                image_pairs[base_name] = [None, None]
            image_pairs[base_name][int(pair_number) - 1] = os.path.join(directory, filename)

    scale_coeff = target_size[0] / orig_size[0]

    for base_name, pair in image_pairs.items():
        if None in pair:
            continue  # Skip incomplete pairs

        img1_path, img2_path = pair

        find_distance_by_image_pair(img1_path,img2_path,M,base,angle_of_view,orig_size,target_size,hdbscan_config, match_tresh)

# Parameters
directory = r"D:\clouds_research\data\test_images\20170901_1300"
base = 17.0
angle_of_view = 62.0
width = 3072.0
orig_size = (3072,2304)
target_size = (int(orig_size[0]/2), int(orig_size[1]/2))
scale_coeff = target_size[0]/orig_size[0]
M = np.array([[9.97682421e-01, 1.15133317e-02, 2.36960989e+02],
                [1.06785471e-02, 9.97667996e-01, 2.48774002e+01]])

config = {'min_cluster_size': 3, 'min_samples': 3, 'epsilon': 2}

# Process image pairs
process_image_pairs(directory, base, angle_of_view, orig_size, target_size, M, config, 0.7)
