# PURPOSE: Turn one stereo pair into candidate cloud layers: rectify the first
#          frame with the calibrated affine matrix, extract dense correspondences
#          with the configured matcher, cluster them in disparity space and convert
#          each cluster to a height with its uncertainty.
#          This class holds only the per-pair measurement step; pairing, logging
#          and the higher-level tasks live in sibling modules.
# INPUTS: a validated configuration, a stereo camera system, two image paths.
# OUTPUTS: a bundle of cluster summaries plus the raw keypoints, disparities,
#          labels and cluster centres, or None when there is too little to cluster.
# KEYWORDS: stereo, analysis, loftr, hdbscan, disparity, cloud_layers
from __future__ import annotations

import math
import os
from collections import Counter

import cv2
import numpy as np
from scipy.signal import find_peaks

from ..clustering.disparity import cluster_shifts
from ..geometry.camera import StereoCameraSystem
from ..geometry.imaging import center_crop, rotation_about
from ..geometry.triangulation import heights_from_matches


# PURPOSE: Estimate cloud base height as the LOWEST PROMINENT MODE of the per-match
#          height distribution, rather than as the minimum of a cluster fragment.
#
#          The shipped chain runs HDBSCAN(min_cluster_size=3) over the 2-D disparity
#          vectors and then takes the lowest size>=8 fragment. That reports an extreme
#          low-tail order statistic of the measurement noise: at min_cluster_size=3 a
#          single layer is shredded into ~232 micro-clusters of median size 5, so the
#          "lowest with >=8 points" is a handful of outliers. It also makes the value
#          depend on the match COUNT, because a larger sample from a heavy-tailed
#          distribution has a lower minimum - which is the mechanism behind the inverted
#          regime response.
#
#          A mode has neither problem: it is a property of where the data concentrates,
#          not of how many points there are, and it is robust to a thin low tail.
#
# INPUTS: heights (N,) per-match triangulated heights in metres; weights (N,) >= 0,
#         typically the inverse ray-intersection residual; min_weight_share (float)
#         the fraction of total weight a mode must carry to count as prominent;
#         min_matches (int) an absolute floor on the same.
# OUTPUTS: (height, info) where height is the lowest prominent mode's location in
#          metres, or None when no mode qualifies. `info` records every mode found and
#          their weights, so the choice is auditable rather than opaque. Returns
#          (None, info) for fewer than min_matches finite points, for all-zero weights,
#          and when every candidate mode falls below the prominence floor.
# KEYWORDS: mode, KDE, robust, selection, order_statistic, R1
def _mode_cluster(hmode, size, disp_x, disp_std, u, v, error):
    """Present the mode estimate in the SAME dict schema the clustering path emits.

    The downstream consumers - to_measurements, the 300 m merge and the cluster dump -
    read a fixed set of keys, so the mode path adapts to that schema rather than
    introducing a second kind of object. The position fields are weight-weighted means
    over the matches, so a dump stays meaningful.
    """
    return {
        "cluster_id": 0,
        "size": int(size),
        "distance": float(hmode),
        "error": float(error),
        "mean_u": float(u),
        "mean_v": float(v),
        "disp_x": float(disp_x),
        "disp_std": float(disp_std),
    }


def lowest_prominent_mode(heights, weights=None, min_weight_share=0.10,
                          min_matches=50, bandwidth_m=None):
    h = np.asarray(heights, float)
    ok = np.isfinite(h)
    h = h[ok]
    if weights is None:
        w = np.ones_like(h)
    else:
        w = np.asarray(weights, float)[ok]
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    if len(h) < min_matches or w.sum() <= 0:
        return None, {"n": int(len(h)), "modes": []}
    # Grid the support and evaluate a weighted Gaussian KDE on it. A grid is used
    # rather than scipy.stats.gaussian_kde because it is O(n * grid) with no
    # per-point matrix, and the grid resolution is set well below the bandwidth.
    lo, hi = float(np.percentile(h, 0.5)), float(np.percentile(h, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, {"n": int(len(h)), "modes": []}
    grid = np.linspace(lo, hi, 512)
    if bandwidth_m is None:
        # Silverman's rule on the weighted sample, floored so a very tight
        # distribution is still smoothed enough to have one mode rather than many.
        sd = float(np.sqrt(np.average((h - np.average(h, weights=w)) ** 2, weights=w)))
        n_eff = max(w.sum() ** 2 / max((w ** 2).sum(), 1e-12), 1.0)
        bandwidth_m = max(1.06 * sd * n_eff ** (-0.2), 25.0)
    bw = float(bandwidth_m)
    # KDE on the grid, in chunks to bound memory
    dens = np.zeros_like(grid)
    for i in range(0, len(h), 4096):
        hh = h[i:i + 4096]; ww = w[i:i + 4096]
        z = (grid[:, None] - hh[None, :]) / bw
        dens += (ww[None, :] * np.exp(-0.5 * z * z)).sum(axis=1)
    dens /= (w.sum() * bw * math.sqrt(2 * math.pi))
    peaks, props = find_peaks(dens, prominence=dens.max() * 0.10)
    modes = []
    if len(peaks):
        # weight under each mode: integrate the density over the basin, which the
        # prominence splitter reports directly as left/right bases
        for k, pi in enumerate(peaks):
            left = int(props["left_bases"][k]); right = int(props["right_bases"][k])
            share = float(np.trapezoid(dens[left:right + 1], grid[left:right + 1]))
            n_in = int(((h >= grid[left]) & (h <= grid[right])).sum())
            modes.append({"h": float(grid[pi]), "share": share, "n": n_in})
    modes.sort(key=lambda m: m["h"])
    info = {"n": int(len(h)), "bandwidth_m": bw, "modes": modes}
    for m in modes:
        if m["share"] >= min_weight_share and m["n"] >= min_matches:
            return m["h"], info
    return None, info
from ..matching.filters import get_corresponding_feature_points, get_matcher
from ..matching.matchers import supports_batching
from .. import viz


# PURPOSE: Visualisation switches. Presentation only; all off, and they never
#          influence a numerical result.
# KEYWORDS: visualisation, switches
VIS_CONTROLS = {
    "show_loftr_matches": False,
    "show_cluster_vectors": False,
    "show_clustered_matches_on_images": False,
    "plot_distance_histograms": False,
    "run_normality_tests": False,
}


# PURPOSE: Measure one stereo pair and describe its cloud layers.
# INPUTS: cfg, stereo system.
# OUTPUTS: a StereoAnalyzer whose analyze_pair() returns the per-pair bundle.
# KEYWORDS: analyzer, stereo, clusters
class StereoAnalyzer:
    """PURPOSE: Per-pair stereo measurement.
    INPUTS: cfg (clouds.config.Config), stereo (StereoCameraSystem).
    OUTPUTS: cluster bundles via analyze_pair().
    KEYWORDS: stereo, analyze, clusters
    """

    def __init__(self, cfg, stereo):
        self.cfg = cfg
        self.stereo = stereo
        # Resolved once from the calibration registry so no code path can mutate
        # a process-wide calibration matrix.
        self.affine = np.array(cfg.calibration_record.matrix_2x3, dtype=float)
        self.visualize = bool(cfg.processing.visualize)
        # f is a geometry model constant, not part of the fixed affine calibration;
        # see docs/plan/w1-calibration.md.
        # The focal constant: configured when measured, otherwise the historical
        # assumption of a 180-degree field inscribed in the frame. See
        # docs/plan/w1-calibration.md W1.8 for the five measurements.
        _f_cfg = float(getattr(cfg.camera, "f_px_per_rad", 0.0) or 0.0)
        self.f_px_per_rad = _f_cfg if _f_cfg > 0 else float(cfg.camera.image_width_px) / math.pi
        self.exact_triangulation = bool(getattr(cfg.camera, "exact_triangulation", False))
        self.hdbscan_config = {
            "min_cluster_size": cfg.clustering.min_cluster_size,
            "min_samples": cfg.clustering.min_samples,
            "allow_single_cluster": cfg.clustering.allow_single_cluster,
        }

    # PURPOSE: Load, rectify and crop one stereo pair into the exact grayscale
    #          crops the matcher will see. Split out from analyze_pair so a caller
    #          can prepare many pairs on the CPU and then match them in few GPU
    #          forward passes.
    # INPUTS: img1_path, img2_path (str), optional affine override.
    # OUTPUTS: (crop1_bgr, crop2_bgr, gray1, gray2), or None if a file is unreadable.
    #          The roll correction is applied to camera 2 because that is the camera
    #          the correction was measured against (see the class docstring and
    #          docs/plan/w1-calibration.md).
    # KEYWORDS: preprocessing, rectification, batch
    def prepare_pair(self, img1_path: str, img2_path: str, affine_matrix=None):
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        if img1_orig is None or img2_orig is None:
            print(f"Warning: Could not load image pair: {img1_path}, {img2_path}")
            return None
        _affine = affine_matrix if affine_matrix is not None else self.affine
        roll = float(getattr(self.cfg.camera, "roll_correction_deg", 0.0) or 0.0)
        if roll:
            cxy = (self.cfg.camera.original_size[0] / 2.0,
                   self.cfg.camera.original_size[1] / 2.0)
            img2_orig = cv2.warpAffine(img2_orig, rotation_about(cxy, roll),
                                       dsize=self.cfg.camera.original_size)
        img1 = cv2.warpAffine(img1_orig, _affine, dsize=self.cfg.camera.original_size)
        img1 = center_crop(img1, self.cfg.camera.target_size[0], self.cfg.camera.target_size[1])
        img2 = center_crop(img2_orig, self.cfg.camera.target_size[0], self.cfg.camera.target_size[1])
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return img1, img2, g1, g2

    # PURPOSE: Turn a set of correspondences into the cluster bundle, with no
    #          matching involved. Shared by the single-pair and batched paths so
    #          both produce identical output.
    # INPUTS: kp1, kp2 (N,2) in the cropped frames; optional colour crops for
    #         visualisation.
    # OUTPUTS: the same dict analyze_pair returns, or None when there are too few
    #          matches to attempt clustering.
    # KEYWORDS: clustering, packaging, batch
    # PURPOSE: Drop correspondences that cannot be cloud, before they reach the
    #          clustering. Two independent reasons a match cannot be cloud: it lies
    #          outside the lens's circular field, or it lies in the near-field
    #          annulus where foliage, structures and the ground are.
    # INPUTS: kp1, kp2 (N,2) in the cropped frame; the analyzer's mask settings.
    # OUTPUTS: (kp1, kp2) filtered, possibly empty. With both masks disabled this is
    #          the identity, so every recorded result is preserved exactly.
    # KEYWORDS: mask, field, foreground, validation, geometry
    def _apply_field_masks(self, kp1, kp2):
        cfg = self.cfg.camera
        keep = np.ones(len(kp1), dtype=bool)
        # The masks are specified in ORIGINAL-frame pixels; the keys are in the crop.
        off = (cfg.original_size[0] - cfg.target_size[0]) / 2.0
        cxy = cfg.original_size[0] / 2.0
        r = np.hypot(kp1[:, 0] + off - cxy, kp1[:, 1] + off - cxy)
        if cfg.field_radius_px and cfg.field_radius_px > 0:
            keep &= r <= cfg.field_radius_px
        if cfg.mask_foreground and cfg.foreground_radius_px > 0:
            keep &= r <= cfg.foreground_radius_px
        return kp1[keep], kp2[keep]

    def package_matches(self, kp1, kp2, img1=None, img2=None):
        kp1, kp2 = self._apply_field_masks(np.asarray(kp1), np.asarray(kp2))
        if len(kp1) < self.hdbscan_config['min_cluster_size']:
            return None
        # The mode path bypasses clustering entirely: it works on the per-match
        # heights directly, so the HDBSCAN fragmentation and the 300 m merge never
        # enter. Emitted as a single cluster so the rest of the pipeline is unchanged.
        if str(self.cfg.temporal.cbh_strategy) == "lowest_mode" and self.exact_triangulation:
            tri = heights_from_matches(
                kp1, kp2, self.f_px_per_rad, self.cfg.camera.stereo_base_m,
                centre=(self.cfg.camera.target_size[0] / 2.0,
                        self.cfg.camera.target_size[1] / 2.0),
                distortion_a=self.cfg.camera.distortion_a)
            hv, rv = tri["heights"], tri["residual"]
            # weight by the inverse ray-intersection residual: a match whose two rays
            # nearly meet is a better estimate of where the cloud surface is
            w = 1.0 / (1.0 + np.where(np.isfinite(rv), rv, np.inf))
            hmode, minfo = lowest_prominent_mode(hv, w)
            self.last_mode_info = minfo
            if hmode is None:
                return None
            du = (kp1 - kp2)[:, 0]
            md = float(np.average(du, weights=w))
            # the error is the mode basin's half-width, so it carries the same meaning
            # as a cluster's spread in the clustering path
            hw = float(np.sqrt(np.average((hv[np.isfinite(hv)] - hmode) ** 2,
                                          weights=w[np.isfinite(hv)])))
            return {"clusters": [_mode_cluster(
                        hmode, len(hv), md,
                        float(np.sqrt(np.average((du - md) ** 2, weights=w))),
                        float(np.average(kp1[:, 0], weights=w)),
                        float(np.average(kp1[:, 1], weights=w)), hw)],
                    "kp1": kp1, "kp2": kp2, "labels": None, "shifts": kp1 - kp2}
        shifts = kp1 - kp2
        centers, std_devs, labels, colors = cluster_shifts(
            shift_arr=shifts,
            hdbscan_config=self.hdbscan_config,
            show_clusters=self.visualize and VIS_CONTROLS['show_cluster_vectors']
        )
        if self.visualize and img1 is not None:
            if VIS_CONTROLS['show_clustered_matches_on_images']:
                viz.visualize_clusters_on_images(img1, img2, kp1, kp2, labels, colors)
            if VIS_CONTROLS['plot_distance_histograms']:
                viz.plot_distance_histogram_with_normality(
                    shifts, labels, colors, self.stereo,
                    run_normality_test=VIS_CONTROLS['run_normality_tests']
                )
        cluster_list = []
        if len(centers) > 0:
            cluster_sizes = Counter(labels)
            valid_clusters = {label: count for label, count in cluster_sizes.items() if label >= 0}
            sorted_cluster_ids = sorted(valid_clusters.keys(), key=lambda k: valid_clusters[k], reverse=True)
            # Exact triangulation, when enabled: per-match heights from the ray
            # intersection, with the measured lens distortion applied, and the
            # cluster's own height distribution summarised. This replaces the
            # paraxial D = f*B/mean-disparity, which is biased off-axis and cannot
            # express the distortion at all.
            exact_h = None
            if self.exact_triangulation:
                exact_h = heights_from_matches(
                    kp1, kp2, self.f_px_per_rad, self.cfg.camera.stereo_base_m,
                    centre=(self.cfg.camera.target_size[0] / 2.0,
                            self.cfg.camera.target_size[1] / 2.0),
                    distortion_a=self.cfg.camera.distortion_a)["heights"]

            for i, cluster_id in enumerate(sorted_cluster_ids):
                disp_x = centers[i][0]
                std_dev_x = std_devs[i][0] if i < len(std_devs) else 0.0
                mask = (labels == cluster_id)
                if exact_h is not None:
                    hv = exact_h[mask]
                    hv = hv[np.isfinite(hv)]
                    if len(hv) == 0:
                        continue
                    distance = float(np.median(hv))
                    # the cluster's own interquartile half-range, propagated as the
                    # error: this is the spread of real matches, not the paraxial
                    # propagation of a mean disparity's standard deviation
                    error = float(0.5 * (np.percentile(hv, 75) - np.percentile(hv, 25)))
                else:
                    distance, error = self.stereo.compute_distance(disp_x, std_dev_x)
                cluster_points_uv = kp1[mask]
                mean_u, mean_v = np.mean(cluster_points_uv, axis=0)
                cluster_list.append({
                    "cluster_id": int(cluster_id),
                    "size": valid_clusters[cluster_id],
                    "distance": distance,
                    "error": error,
                    "mean_u": mean_u,
                    "mean_v": mean_v,
                    "disp_x": disp_x,
                    "disp_std": float(std_dev_x),
                })
        return {
            'clusters': cluster_list,
            'kp1': kp1,
            'kp2': kp2,
            'shifts': shifts,
            'labels': labels,
            'centers': centers,
        }

    # PURPOSE: Run only the matcher on already-prepared crops, so the matcher can
    #          execute on a worker thread while the consumer clusters the previous
    #          pair. Split out of analyze_prepared without changing its behaviour.
    # INPUTS: gray1, gray2 (2-D uint8).
    # OUTPUTS: (kp1, kp2) correspondence arrays.
    # KEYWORDS: match, split, overlap, threading
    def match_prepared(self, gray1, gray2):
        return get_corresponding_feature_points(
            gray1, gray2,
            visualize=self.visualize and VIS_CONTROLS['show_loftr_matches'],
            match_thresh=self.cfg.matching.match_threshold,
            backend=self.cfg.matching.backend,
            filter_config=self.cfg.matching.filter,
            **(self.cfg.matching.params or {}),
        )

    # PURPOSE: Match already-prepared grayscale crops and package the result.
    # INPUTS: gray1, gray2 (2-D uint8); optional colour crops for visualisation.
    # OUTPUTS: the cluster bundle, or None.
    # KEYWORDS: single_pair, match, analyze
    def analyze_prepared(self, gray1, gray2, img1=None, img2=None):
        kp1, kp2 = self.match_prepared(gray1, gray2)
        return self.package_matches(kp1, kp2, img1, img2)

    # PURPOSE: Analyze many pairs with few GPU forward passes. The evaluation loop
    #          spends 41 % of its time in the matcher and the GPU is largely idle
    #          at one pair per pass, so this is a throughput measure; it yields the
    #          same correspondences as analyze_pair up to floating-point batching.
    # INPUTS: pairs (list of (path1, path2)); batch_size (int).
    # OUTPUTS: list of cluster bundles in input order, with None where a pair is
    #          unreadable or yields too few matches.
    # KEYWORDS: batching, throughput, evaluate
    def analyze_batch(self, pairs, batch_size: int = 6):
        prepped = [self.prepare_pair(p1, p2, affine_matrix=self.affine) for p1, p2 in pairs]
        grays, idx = [], []
        for i, prep in enumerate(prepped):
            if prep is not None:
                grays.append((prep[2], prep[3]))
                idx.append(i)
        results = [None] * len(pairs)
        if not grays:
            return results
        matcher = get_matcher(self.cfg.matching.backend,
                              {"match_threshold": self.cfg.matching.match_threshold,
                               **(self.cfg.matching.params or {})})
        if supports_batching(matcher):
            matches = matcher.match_batch(grays, batch_size=batch_size)
        else:
            matches = [matcher(a, b) for a, b in grays]
        for j, i in enumerate(idx):
            kp1, kp2 = matches[j]
            prep = prepped[i]
            results[i] = self.package_matches(kp1, kp2, prep[0], prep[1])
        return results

    def analyze_pair(self, img1_path: str, img2_path: str, affine_matrix=None) -> dict | None:
        """
        Analyzes a single stereo pair and returns cluster info plus raw match data.

        Args:
            img1_path, img2_path: Paths to the stereo pair.
            affine_matrix: Optional 2x3 numpy array overriding config.AFFINE_MATRIX.

        Returns:
            dict with keys:
              - 'clusters': list[dict]  (cluster summaries as before)
              - 'kp1': np.ndarray      (N,2) matched keypoints in image1
              - 'kp2': np.ndarray      (N,2) matched keypoints in image2
              - 'shifts': np.ndarray   (N,2) disparity vectors
              - 'labels': np.ndarray   (N,)  cluster label per match (-1=noise)
              - 'centers': np.ndarray  (M,2) cluster centroids in shift space
            or None if insufficient data.
        """
        prep = self.prepare_pair(img1_path, img2_path, affine_matrix)
        if prep is None:
            return None
        return self.analyze_prepared(prep[2], prep[3], prep[0], prep[1])
