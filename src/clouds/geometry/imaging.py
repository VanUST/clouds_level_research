# /src/image_utils.py
import math

import cv2
import numpy as np


# PURPOSE: Build the 2x3 affine that rotates an image about a chosen point.
#          Used to apply the measured residual roll between the stereo cameras:
#          for a fisheye whose image radius is proportional to the angle from the
#          optical axis, rotating the image about the principal point is exactly
#          equivalent to rotating the viewing directions, so this corrects the
#          geometry without touching the lens model or the calibration matrix.
# INPUTS: centre (cx, cy) in pixels; angle_deg (float), counter-clockwise in
#         image coordinates (visually clockwise, because image y points down).
# OUTPUTS: (2,3) float array for cv2.warpAffine. A zero angle returns the
#          identity, so callers need no special case.
# KEYWORDS: rotation, roll, rectification, affine, calibration_correction
def rotation_about(centre, angle_deg: float) -> np.ndarray:
    """
    Composes translate(-centre) * rotate(angle) * translate(+centre) into a single
    2x3 matrix, so warpAffine needs one pass and creates no intermediate image.
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = float(centre[0]), float(centre[1])
    return np.array([[c, -s, cx - c * cx + s * cy],
                     [s, c, cy - s * cx - c * cy]], dtype=float)


def center_crop(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """
    Crops an image to the specified size from the center.

    Args:
        image (np.ndarray): The input image.
        crop_width (int): The desired width of the cropped image.
        crop_height (int): The desired height of the cropped image.

    Returns:
        np.ndarray: The center-cropped image.
    """
    height, width = image.shape[:2]
    start_x = max(0, (width - crop_width) // 2)
    start_y = max(0, (height - crop_height) // 2)
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def show_image(img: np.ndarray, window_name: str = "Image"):
    """
    Displays an image in a fullscreen window until a key is pressed.

    Args:
        img (np.ndarray): The image to display.
        window_name (str): The name of the display window.
    """
    cv2.namedWindow(window_name, flags=cv2.WND_PROP_FULLSCREEN)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()