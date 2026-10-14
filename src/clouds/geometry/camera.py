# /src/camera.py
import math
import numpy as np
from enum import Enum


# PURPOSE: Enumerate the supported camera projection models so distance
#          calculations can dispatch on an explicit, validated value rather than
#          a bare string.
# INPUTS: none.
# OUTPUTS: CameraModel.FISHEYE or CameraModel.PINHOLE.
# KEYWORDS: camera, projection, fisheye, pinhole, enum
class CameraModel(Enum):
    """Supported camera projection models."""
    PINHOLE = 'pinhole'
    FISHEYE = 'fisheye'

class StereoCameraSystem:
    """
    Represents the stereo camera setup and handles distance calculations.
    """
    def __init__(self, base: float, angle_of_view: float, image_width: int, model: CameraModel):
        """
        Initializes the stereo camera system.

        Args:
            base (float): The baseline distance between the cameras.
            angle_of_view (float): The camera's field of view in degrees.
            image_width (int): The width of the images in pixels.
            model (CameraModel): The camera projection model (PINHOLE or FISHEYE).
        """
        if not isinstance(model, CameraModel):
            raise TypeError("model must be an instance of CameraModel Enum")
            
        self.base = base
        self.angle_of_view = angle_of_view
        self.image_width = image_width
        self.model = model

    def compute_distance(self, delta_px: float, delta_px_error: float) -> tuple[float, float]:
        """
        Calculates the distance to an object based on pixel disparity.

        This method acts as a dispatcher, calling the appropriate formula
        based on the camera model specified during initialization.

        Args:
            delta_px (float): The horizontal disparity in pixels.
            delta_px_error (float): The standard deviation of the disparity.

        Returns:
            A tuple containing:
            - The calculated distance.
            - The associated error in the distance calculation.
        """
        # Avoid division by zero for disparity
        if abs(delta_px) < 1e-6:
            delta_px = 1e-6

        if self.model == CameraModel.PINHOLE:
            return self._compute_dist_pinhole(delta_px, delta_px_error)
        elif self.model == CameraModel.FISHEYE:
            return self._compute_dist_fisheye(delta_px, delta_px_error)
        else:
            raise ValueError(f"Unsupported camera model: {self.model}")

    def _compute_dist_pinhole(self, delta_px: float, delta_px_error: float) -> tuple[float, float]:
        """Distance and error calculation for a pinhole camera model."""
        angle_in_radians = math.radians(self.angle_of_view / 2)
        
        # Common factor to simplify expressions
        denominator_factor = 2 * math.tan(angle_in_radians)
        
        distance = (self.base * self.image_width) / (denominator_factor * abs(delta_px))
        error = (self.base * self.image_width * delta_px_error) / (denominator_factor * delta_px**2)
        
        return distance, error

    def _compute_dist_fisheye(self, delta_px: float, delta_px_error: float) -> tuple[float, float]:
        """Distance and error calculation for a fisheye camera model."""
        distance = (self.image_width * self.base) / (abs(delta_px) * math.pi)
        error = (self.base * self.image_width * delta_px_error) / (math.pi * delta_px**2)
        
        return distance, error