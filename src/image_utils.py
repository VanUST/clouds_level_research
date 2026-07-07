# /src/image_utils.py
import cv2
import numpy as np

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