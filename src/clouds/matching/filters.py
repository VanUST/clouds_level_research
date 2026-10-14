# PURPOSE: Correspondence extraction between a stereo pair — the callable the
#          pipeline invokes — together with the match filters that clean its
#          output. Kept separate from the matcher backends so a backend can be
#          replaced without touching filter policy.
# INPUTS: two images; a matcher backend name; a filter configuration.
# OUTPUTS: (keypoints0, keypoints1) arrays, optionally filtered.
# KEYWORDS: matching, correspondence, filters, epipolar, cycle_consistency
from __future__ import annotations

import numpy as np
import torch

# Global cache for the active matcher instance
_matcher_instance = None
_matcher_backend = None


# PURPOSE: Create or retrieve the cached feature matcher function for the active backend.
# INPUTS: backend_name (str), config (dict)
# OUTPUTS: callable (img0, img1, visualize=False) -> (np.ndarray, np.ndarray)
# KEYWORDS: lazy_init, matcher, singleton, cache, backend
def _get_matcher(backend_name, config):
    global _matcher_instance, _matcher_backend
    if _matcher_instance is not None and _matcher_backend == backend_name:
        return _matcher_instance
    from .matchers import create_matcher
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _matcher_instance = create_matcher(backend_name, config, device)
    _matcher_backend = backend_name
    return _matcher_instance


# PURPOSE: Expose the cached matcher object itself, so a caller that owns several
#          image pairs can batch them into few forward passes instead of paying one
#          forward pass per pair. The returned callable may expose `match_batch`.
# INPUTS: backend_name (str), config (dict).
# OUTPUTS: the cached matcher callable.
# KEYWORDS: batching, matcher, accessor
def get_matcher(backend_name, config):
    return _get_matcher(backend_name, config)


# PURPOSE: Find corresponding feature points between two images using the configured matcher backend,
#          with optional match filtering (epipolar, confidence, disparity-MAD, cycle-consistency).
# INPUTS: img1 (np.ndarray), img2 (np.ndarray), visualize (bool), match_thresh (float),
#         backend (str) – one of 'kornia_loftr', 'matchanything_eloftr',
#         filter_config (dict|None) – controls post-match filtering
# OUTPUTS: Tuple[np.ndarray, np.ndarray] keypoints from image1 and image2.
# KEYWORDS: feature_matching, matcher, loftr, eloftr, stereo, correspondences, filtering
def get_corresponding_feature_points(img1, img2, visualize=False, match_thresh=0.5,
                                     backend="kornia_loftr", filter_config=None, **kwargs):
    """
    Finds corresponding feature points between two images.

    Args:
        img1 (np.ndarray): The first image (BGR or grayscale).
        img2 (np.ndarray): The second image (BGR or grayscale).
        visualize (bool): If True, displays the matches.
        match_thresh (float): Confidence threshold for the matcher.
        backend (str): Matcher backend: 'kornia_loftr' or 'matchanything_eloftr'.
        filter_config (dict|None): Match filtering configuration with keys:
            - 'y_thresh_px' (float): Max allowed y-disparity in pixels (epipolar).
            - 'conf_thresh' (float): Min match confidence score.
            - 'mad_n' (float): Disparity MAD multiplier for outlier removal.
            - 'cycle_consistency' (bool): Enable bidirectional cycle check.
            - 'cycle_pixel_thresh' (float): Pixel tolerance for cycle consistency.
        **kwargs: Additional backend-specific config.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Keypoints from image 1, keypoints from image 2.
    """

    config = {"match_threshold": match_thresh, **kwargs}
    matcher = _get_matcher(backend, config)

    fc = filter_config or {}

    # Cycle-consistency: runs matcher bidirectionally, takes priority
    if fc.get("cycle_consistency"):
        pixel_thresh = fc.get("cycle_pixel_thresh", 2.0)
        kp1, kp2 = filter_matches_cycle_consistency(
            matcher, img1, img2, pixel_thresh=pixel_thresh
        )
        # Apply additional filters on the cycle-consistent result
        masks = []
        if fc.get("y_thresh_px") is not None:
            masks.append(filter_mask_epipolar(kp1, kp2, fc["y_thresh_px"]))
        if fc.get("mad_n") is not None and len(kp1) > 0:
            masks.append(filter_mask_disparity_mad(kp1, kp2, n_mad=fc["mad_n"]))
        kp1, kp2 = apply_filter_masks(kp1, kp2, masks)
        return kp1, kp2

    # Standard matching with optional confidence return
    result = matcher(img1, img2, visualize=visualize, return_conf=fc.get("conf_thresh") is not None)
    if isinstance(result, tuple) and len(result) == 3:
        kp1, kp2, conf = result
    else:
        kp1, kp2 = result
        conf = None

    # Build filter masks
    masks = []
    if fc.get("y_thresh_px") is not None and len(kp1) > 0:
        masks.append(filter_mask_epipolar(kp1, kp2, fc["y_thresh_px"]))
    if fc.get("conf_thresh") is not None and conf is not None:
        masks.append(filter_mask_confidence(conf, fc["conf_thresh"]))
    if fc.get("mad_n") is not None and len(kp1) > 0:
        masks.append(filter_mask_disparity_mad(kp1, kp2, n_mad=fc["mad_n"]))

    kp1, kp2 = apply_filter_masks(kp1, kp2, masks)
    return kp1, kp2


# PURPOSE: Backward-compatible wrapper that delegates to the Kornia LoFTR backend.
# INPUTS: Same as get_corresponding_feature_points with backend='kornia_loftr'.
# OUTPUTS: Tuple[np.ndarray, np.ndarray]
# KEYWORDS: legacy, loftr, kornia, backward_compat
def get_corresponding_feature_points_loftr(img1, img2, visualize=False, match_thresh=0.5):
    """
    Legacy wrapper around Kornia LoFTR matcher (backward compatible).

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.
        visualize (bool): If True, displays the matches.
        match_thresh (float): The confidence threshold for LoFTR matches.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Keypoints from image 1, keypoints from image 2.
    """
    return get_corresponding_feature_points(
        img1, img2, visualize=visualize, match_thresh=match_thresh,
        backend="kornia_loftr"
    )


# ---------------------------------------------------------------------------
# Match filtering utilities
# ---------------------------------------------------------------------------

# PURPOSE: Filter matched keypoints by epipolar constraint — corresponding points
#          in a rectified stereo pair must lie on the same horizontal scanline.
# INPUTS: kp0, kp1 (np.ndarray Nx2), y_thresh_px (float)
# OUTPUTS: mask (np.ndarray bool of length N)
# KEYWORDS: epipolar, filter, y_disparity, rectification, scanline
def filter_mask_epipolar(kp0, kp1, y_thresh_px):
    y_diff = np.abs(kp0[:, 1] - kp1[:, 1])
    return y_diff <= y_thresh_px


# PURPOSE: Filter matched keypoints by confidence score threshold.
# INPUTS: conf (np.ndarray length N), conf_thresh (float)
# OUTPUTS: mask (np.ndarray bool of length N)
# KEYWORDS: confidence, filter, threshold, match_quality
def filter_mask_confidence(conf, conf_thresh):
    return conf >= conf_thresh


# PURPOSE: Filter matched keypoints by statistical disparity consistency.
#          Removes matches whose disparity deviates from the median by more than
#          n_mad * MAD (Median Absolute Deviation). This removes outlier matches
#          that likely correspond to physically implausible depths.
# INPUTS: kp0, kp1 (np.ndarray Nx2), n_mad (float), min_matches (int)
# OUTPUTS: mask (np.ndarray bool of length N)
# KEYWORDS: disparity, outlier, MAD, statistical_filter, robust
def filter_mask_disparity_mad(kp0, kp1, n_mad=3.0, min_matches=10):
    dx = kp0[:, 0] - kp1[:, 0]
    median = np.median(dx)
    mad = np.median(np.abs(dx - median))
    if mad < 1e-8:
        return np.ones(len(kp0), dtype=bool)
    threshold = n_mad * mad
    return np.abs(dx - median) <= threshold


# PURPOSE: Filter matches by bidirectional cycle-consistency (forward-backward check).
#          Runs the matcher in both directions and keeps only matches where a point
#          in img0 maps to a point in img1 that maps back to within pixel_thresh of
#          the original point in img0.
# INPUTS: matcher (callable), img1, img2 (np.ndarray), pixel_thresh (float)
# OUTPUTS: (kp0_filt, kp1_filt) tuple of filtered np.ndarrays
# KEYWORDS: cycle_consistency, backward_forward, mutual_check, bidirectional
def filter_matches_cycle_consistency(matcher, img1, img2, pixel_thresh=2.0):
    from scipy.spatial import cKDTree

    kp0_fwd, kp1_fwd = matcher(img1, img2, visualize=False)
    kp1_bwd, kp0_bwd = matcher(img2, img1, visualize=False)

    if len(kp0_fwd) == 0 or len(kp0_bwd) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    tree_bwd_kp1 = cKDTree(kp1_bwd)
    dists_1, idx_1 = tree_bwd_kp1.query(kp1_fwd, distance_upper_bound=pixel_thresh)

    valid_forward = np.isfinite(dists_1)
    if not np.any(valid_forward):
        return np.empty((0, 2)), np.empty((0, 2))

    matched_kp0_bwd = kp0_bwd[idx_1[valid_forward]]
    dists_back = np.linalg.norm(kp0_fwd[valid_forward] - matched_kp0_bwd, axis=1)
    valid_back = dists_back <= pixel_thresh

    mask_keep = np.zeros(len(kp0_fwd), dtype=bool)
    mask_keep[valid_forward] = valid_back

    return kp0_fwd[mask_keep], kp1_fwd[mask_keep]


# PURPOSE: Apply a combined mask of multiple filters (logical AND) and return filtered keypoints.
# INPUTS: kp0, kp1 (np.ndarray Nx2), masks (list of bool np.ndarray)
# OUTPUTS: (kp0_filt, kp1_filt) tuple of filtered np.ndarrays
# KEYWORDS: combine_filters, AND, mask, keypoints
def apply_filter_masks(kp0, kp1, masks):
    if not masks:
        return kp0, kp1
    combined = np.ones(len(kp0), dtype=bool)
    for m in masks:
        if m is not None:
            combined &= m
    return kp0[combined], kp1[combined]
