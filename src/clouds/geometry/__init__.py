"""PURPOSE: Geometry — camera model, projection, the height<->disparity relation
and the image operations used to prepare a stereo pair.

KEYWORDS: package, geometry, camera, imaging
"""
from .camera import CameraModel, StereoCameraSystem  # noqa: F401
from .imaging import center_crop  # noqa: F401
