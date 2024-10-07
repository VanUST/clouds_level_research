import cv2
import numpy as np
import math

def resize(img,target_size ):
     H_target, W_target = target_size
     # Resize images to the target size (H, W)
     img_resized = cv2.resize(img, (W_target, H_target))
     return img_resized

def affine_calibration(image_pairs):
    """
    Perform affine calibration using manually selected corresponding points across image pairs.

    Parameters:
    - image_pairs: List of tuples containing pairs of image file paths.
    - target_size: Tuple (H, W) specifying the height and width to which images should be resized.

    Returns:
    - M: The computed affine transformation matrix using all the selected points.
    """
    
    all_pts_src = []
    all_pts_dst = []

    for idx, (img1_path, img2_path) in enumerate(image_pairs):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None:
            print(f"Error loading image {img1_path}")
            continue
        if img2 is None:
            print(f"Error loading image {img2_path}")
            continue

        # Stack images horizontally
        composite_img = np.hstack((img1, img2))

        # Lists to store points for this image pair
        pts_src = []
        pts_dst = []

        # Variables to keep track of clicks
        click_state = [0]  # 0: expecting click on first image, 1: on second image
        point_counter = [0]

        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Determine whether click is on first or second image
                img1_width = img1.shape[1]
                if click_state[0] == 0:
                    if x < img1_width:
                        # Click is on first image
                        pts_src.append([x, y])
                        # Draw a circle on the point
                        cv2.circle(composite_img, (x, y), 5, (0, 0, 255), -1)
                        cv2.imshow(f"Image Pair {idx+1}", composite_img)
                        click_state[0] = 1  # Now expect click on second image
                    else:
                        print("Please click on the first image (left)")
                elif click_state[0] == 1:
                    if x >= img1_width:
                        # Adjust x coordinate relative to second image
                        x_adj = x - img1_width
                        pts_dst.append([x_adj, y])
                        # Draw a circle on the point
                        cv2.circle(composite_img, (x, y), 5, (255, 0, 0), -1)
                        cv2.imshow(f"Image Pair {idx+1}", composite_img)
                        click_state[0] = 0  # Now expect click on first image
                        point_counter[0] += 1
                        print(f"Point pair {point_counter[0]} selected")
                    else:
                        print("Please click on the second image (right)")
                        
        cv2.namedWindow(f"Image Pair {idx+1}",cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback(f"Image Pair {idx+1}", mouse_callback)

        print(f"\nSelect corresponding points for image pair {idx+1}")
        print("Click on a point in the first image (left), then the corresponding point in the second image (right).")
        print("Press 'q' when done with this image pair.")

        while True:
            cv2.imshow(f"Image Pair {idx+1}", composite_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow(f"Image Pair {idx+1}")

        # Convert points to numpy arrays
        pts_src = np.array(pts_src, dtype=np.float32)
        pts_dst = np.array(pts_dst, dtype=np.float32)

        # Append points to the global lists
        all_pts_src.append(pts_src)
        all_pts_dst.append(pts_dst)

    if len(all_pts_src) == 0 or len(all_pts_dst) == 0:
        print("No points were selected.")
        return None

    # Concatenate all points
    all_pts_src = np.vstack(all_pts_src)
    all_pts_dst = np.vstack(all_pts_dst)

    if len(all_pts_src) < 3:
        print("At least 3 point pairs are required to compute an affine transformation.")
        return None

    # Now perform affine transformation calibration on all these points
    M, inliers = cv2.estimateAffine2D(all_pts_src, all_pts_dst, method=cv2.RANSAC)

    if M is not None:
        # Convert to a 3x3 matrix by adding [0,0,1] to the bottom
        M_affine = np.vstack([M, [0, 0, 1]])
        print("\nAffine transformation matrix computed using all selected points:")
        print(M_affine)
    else:
        print("Affine transformation could not be computed.")
        return None

    return M_affine

if __name__ == "__main__":
    image_pairs = [
        (r'D://clouds_research//data//stars//Stars-2017-09//20170901-205833-593-1.jpeg', r'D://clouds_research//data//stars//Stars-2017-09//20170901-205833-593-2.jpeg'),
        (r'D://clouds_research//data//stars//Stars-2017-09//20170901-215833-593-1.jpeg', r'D://clouds_research//data//stars//Stars-2017-09//20170901-215833-593-2.jpeg'),
        (r'D://clouds_research//data//stars//Stars-2017-09//20170901-225833-609-1.jpeg',r'D://clouds_research//data//stars//Stars-2017-09//20170901-225833-609-2.jpeg'),
        (r'D://clouds_research//data//stars//Stars-2017-09//20170901-235833-625-1.jpeg',r'D://clouds_research//data//stars//Stars-2017-09//20170901-235833-625-2.jpeg'),
        (r'D://clouds_research//data//stars//Stars-2017-09//20170902-005833-671-1.jpeg',r'D://clouds_research//data//stars//Stars-2017-09//20170902-005833-671-2.jpeg')
    ]

    M = affine_calibration(image_pairs)
    if M is not None:
        # Use the affine transformation matrix M as needed
        pass