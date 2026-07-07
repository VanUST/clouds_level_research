# /tools/trail_matcher.py
import cv2
import numpy as np
import os
import sys

# Add the project root to the Python path to allow importing 'config' and 'src'
sys.path.append(os.getcwd())
import config
from src.camera import StereoCameraSystem # <-- Import StereoCameraSystem

def select_trail_points(img1_path, img2_path, output_path):
    """
    Manually select corresponding points along a trail in two images and save them.
    Includes a visual guide and live distance calculation for each point pair.
    """
    img1_orig = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1_orig is None or img2 is None:
        print("Error: Could not load one or both images.")
        return

    # --- Initialize the Stereo Camera System for live calculations ---
    stereo_system = StereoCameraSystem(
        base=config.STEREO_BASE,
        angle_of_view=config.ANGLE_OF_VIEW,
        image_width=config.IMAGE_WIDTH,
        model=config.CAMERA_MODEL
    )
    print("Stereo system initialized for live distance feedback.")

    # Apply the affine matrix to the first image
    print("Applying affine transformation to the first image...")
    img1_corrected = cv2.warpAffine(img1_orig, config.AFFINE_MATRIX, dsize=config.ORIGINAL_SIZE)
    
    # Stack the corrected image 1 and original image 2
    composite_img = np.hstack((img1_corrected, img2))
    window_name = "Manual Trail Matcher - 's' to save, 'q' to quit"
    
    pts_src = []
    pts_dst = []
    click_state = 0  # 0: expecting click on left image, 1: on right image

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_state
        if event == cv2.EVENT_LBUTTONDOWN:
            img1_width = img1_corrected.shape[1]
            if click_state == 0:
                if x < img1_width:
                    pts_src.append([x, y])
                    # Draw confirmed green circle on the persistent composite image
                    cv2.circle(composite_img, (x, y), 5, (0, 255, 0), -1)
                    
                    # --- ENHANCEMENT 1: Draw temporary guide on a copy ---
                    display_copy = composite_img.copy()
                    guide_x = x + img1_width
                    cv2.drawMarker(display_copy, (guide_x, y), (0, 0, 255), 
                                   markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)
                    cv2.imshow(window_name, display_copy)
                    # ----------------------------------------------------
                    
                    click_state = 1
                else:
                    print("Please click on the trail in the LEFT image first.")
            
            elif click_state == 1:
                if x >= img1_width:
                    x_adj = x - img1_width
                    pts_dst.append([x_adj, y])
                    
                    # Draw confirmed point and connecting line on the persistent image
                    cv2.circle(composite_img, (x, y), 5, (0, 255, 0), -1)
                    last_pt_src = (pts_src[-1][0], pts_src[-1][1])
                    cv2.line(composite_img, last_pt_src, (x, y), (0, 255, 255), 1)
                    
                    # Show the final persistent image (removes the red guide cross)
                    cv2.imshow(window_name, composite_img)
                    
                    # --- ENHANCEMENT 2: Live distance calculation ---
                    last_kp1 = pts_src[-1]
                    last_kp2 = pts_dst[-1]
                    disp_x = last_kp1[0] - last_kp2[0]
                    
                    if abs(disp_x) > 1e-6:
                        distance, _ = stereo_system.compute_distance(disp_x, delta_px_error=0)
                        print(f"  > Point #{len(pts_src)}: Disparity = {disp_x:.2f} px, Distance = {distance:.2f} meters")
                    else:
                        print(f"  > Point #{len(pts_src)}: Disparity is near zero. Cannot calculate distance.")
                    # ------------------------------------------------
                    
                    click_state = 0 # Reset for the next pair of points
                else:
                    print("Please click on the corresponding point in the RIGHT image.")

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\nClick along the aircraft trail: Left image first, then the corresponding point on the right.")
    
    while True:
        # Initial display or updates are handled by the callback
        if 'display_copy' not in locals(): # Show initial image
             cv2.imshow(window_name, composite_img)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            if len(pts_src) != len(pts_dst) or len(pts_src) == 0:
                print("\nError: Point pairs are incomplete. Cannot save.")
                continue
            
            matched_points = {'kp1': np.array(pts_src), 'kp2': np.array(pts_dst)}
            np.savez(output_path, **matched_points)
            print(f"\nSuccessfully saved {len(pts_src)} point pairs to {output_path}")
            break
            
        elif key == ord('q'):
            print("\nQuit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- IMPORTANT: CONFIGURE YOUR FILES HERE ---
    IMG1_NAME = "img-2024-10-12T09-42-44devID1.jpg"
    IMG2_NAME = "img-2024-10-12T09-42-44devID2.jpg"
    OUTPUT_NAME = "trail_points.npz"
    
    # Paths are now built using the IMAGE_DIR from your main config file
    img1_path = os.path.join(config.IMAGE_DIR, IMG1_NAME)
    img2_path = os.path.join(config.IMAGE_DIR, IMG2_NAME)
    output_path = os.path.join(config.IMAGE_DIR, OUTPUT_NAME)

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: One or both images not found.")
        print(f"Checked for: {img1_path}")
        print(f"Checked for: {img2_path}")
    else:
        select_trail_points(img1_path, img2_path, output_path)