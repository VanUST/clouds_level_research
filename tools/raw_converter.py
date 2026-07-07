import os
import rawpy
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def process_image_with_threshold(rgb_image, min_value):
    # Apply denoising using Gaussian filter
    denoised_image = gaussian_filter(rgb_image, sigma=1)

    # Cut the lower part of the spectrum (enhance contrast)
    denoised_image = np.clip(denoised_image, min_value, 255)
    denoised_image = (denoised_image - min_value) * (255.0 / (255.0 - min_value))
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

    return denoised_image

def convert_crw_to_jpeg(directory):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        if filename.lower().endswith(".crw"):
            crw_path = os.path.join(directory, filename)
            jpeg_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.jpeg")
            
            try:
                # Load the CRW file
                with rawpy.imread(crw_path) as raw:
                    # Process raw data to RGB image
                    rgb_image = raw.postprocess()

                min_value = 50  # Initial threshold value
                while True:
                    # Process the image with the current threshold
                    processed_image = process_image_with_threshold(rgb_image, min_value)

                    # Show the processed image using OpenCV
                    cv2.imshow(f"Threshold: {min_value}", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                    
                    # Wait for user input
                    key = cv2.waitKey(0)

                    if key == ord('y'):  # If user presses 'y', accept the image
                        break
                    elif key == ord('n'):  # If user presses 'n', allow changing the threshold
                        try:
                            new_threshold = input("Enter new threshold value (0-255): ")
                            min_value = int(new_threshold)
                        except ValueError:
                            print("Invalid threshold value. Please enter an integer between 0 and 255.")
                    else:
                        print("Invalid input. Press 'y' to accept, 'n' to change the threshold.")

                # Convert to a PIL image and save as JPEG
                img = Image.fromarray(processed_image)
                img.save(jpeg_path, "JPEG")
                print(f"Converted: {filename} -> {jpeg_path}")

                # Destroy the OpenCV window for the current image
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    directory = input("Enter the directory containing CRW images: ")
    convert_crw_to_jpeg(directory)
