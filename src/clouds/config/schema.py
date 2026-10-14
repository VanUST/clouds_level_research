# PURPOSE: Load, validate and hash the project's YAML configuration, and resolve
#          calibration identities from the calibration registry. This replaces
#          the previous mutable config.py module: configuration is now data that
#          is read once, validated, frozen, and hashed into every run manifest,
#          so a result can always be tied back to the exact parameters that made
#          it.
# INPUTS: path to a YAML config file (see configs/default.yaml for the schema);
#         optionally the id of a calibration entry under configs/calib/.
# OUTPUTS: a frozen Config dataclass tree; a Calibration record; a sha256 over the
#          raw config bytes. Raises ConfigError on any schema violation, so a
#          malformed config fails loudly instead of silently defaulting.
# KEYWORDS: config, yaml, dataclass, validation, hash, calibration_registry
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml




class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed or inconsistent."""




# ---------------------------------------------------------------------------
# Typed config tree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectCfg:
    name: str
    purpose: str


@dataclass(frozen=True)
class DatasetCfg:
    images_dir: str
    date: str
    window_start: str
    window_end: str
    image_clock: str
    ceilometer_clock: str
    clock_offset_h: float
    ceilometer_log: str
    pair_tolerance_s: float
    manual_points: str = ""


@dataclass(frozen=True)
class CameraCfg:
    model: str
    stereo_base_m: float
    angle_of_view_deg: float
    image_width_px: int
    original_size: tuple
    target_size: tuple
    calibration: str
    # Measured residual roll of the second camera about the principal point, in
    # degrees, applied before matching. Zero (the default) reproduces the
    # historical behaviour exactly, so the correction is always an explicit,
    # recorded choice rather than a silent change. The value is measured from the
    # images themselves (experiments/w1/epipolar_check.py), never from ground
    # truth. See docs/plan/w1-calibration.md W1.12-W1.13.
    roll_correction_deg: float = 0.0
    # Cubic radial lens distortion in theta = (r/f)(1 + a (r/f)^2). Measured from
    # the sky at -0.30 (experiments/layers/distortion_fit.py); 0.0 is the ideal
    # equidistant law and reproduces the historical behaviour exactly.
    distortion_a: float = 0.0
    # When true, per-match heights come from exact unit-sphere triangulation with
    # the distortion above, instead of the paraxial D = f*B/disparity applied to a
    # cluster's mean disparity. Off by default so every recorded result stands.
    exact_triangulation: bool = False
    # Focal constant in px/rad. 0.0 means "derive it from image_width_px / pi",
    # which is the historical assumption of a 180-degree field inscribed in the
    # frame. Five independent measurements put the true value near 654 px/rad
    # (docs/plan/w1-calibration.md W1.8): the assumed 611 px/rad makes every height
    # about 7 % low. Setting this explicitly is a recorded choice, never a default
    # change.
    f_px_per_rad: float = 0.0
    # Radial field mask, in pixels from the image centre of the ORIGINAL frame.
    # 0.0 disables it. The natural field of a circular fisheye image is a circle;
    # a square crop reaches sqrt(2) times further in its corners and admits
    # near-field terrain. Measured effect of the corners is in
    # docs/plan/research-workplan.md R12.
    field_radius_px: float = 0.0
    # When true, drop matches in the outer annulus where the ground, foliage and
    # structures are, which cannot be cloud.
    mask_foreground: bool = False
    # Radius, in ORIGINAL-frame pixels from the image centre, inside which the
    # foreground mask keeps matches. Only used when mask_foreground is true.
    foreground_radius_px: float = 0.0


@dataclass(frozen=True)
class MatchingCfg:
    backend: str
    match_threshold: float
    filter: Optional[dict]
    # Pairs matched per GPU forward pass. 1 (the default) reproduces the exact
    # historical numerics; larger values are a throughput measure for development
    # and differ from 1 by ~0.1 % of matches and ~1 % of median height, because
    # batching changes floating-point reduction order.
    batch_size: int = 1
    # Pairs whose decode/rectify runs one thread ahead of the consumer. Has no
    # numerical effect (identical prepared arrays, only produced earlier), so it
    # is on by default; set 0 to disable and reproduce the strictly serial loop.
    prefetch: int = 1
    # When True the prefetch worker also runs the matcher, so the consumer thread
    # only clusters/filters while the GPU works ahead. No numerical effect; it
    # changes which thread executes matching. Measured 4.10 fps vs the
    # prepare-only prefetch's 3.71 fps over 150 frames of 2025-05-23
    # (configs/_ab_overlap1.yaml vs configs/_ab_batch1.yaml with overlap off).
    # Requires prefetch >= 1 and batch_size == 1.
    overlap_matching: bool = True
    # Backend-specific keyword arguments forwarded verbatim to the matcher factory
    # for the selected `backend` (for example the dense backends' grid_stride and
    # certainty_threshold). None means "use the backend's own defaults", so the
    # historical kornia_loftr path is unchanged when this key is absent.
    params: Optional[dict] = None


@dataclass(frozen=True)
class ClusteringCfg:
    min_cluster_size: int
    min_samples: int
    allow_single_cluster: bool




@dataclass(frozen=True)
class TemporalCfg:
    mode: str
    cbh_strategy: str
    sigma_h_accel: float
    measurement_noise_scale: float
    cluster_merge_height_tol_m: float
    min_cluster_size_for_tracking: int
    # When false, the Kalman filter is bypassed and the delivered series is the
    # per-frame selection. Default true preserves every recorded result. Added
    # 2026-10-13 on review: the filter was measured to remove 48 % of the raw
    # variation of a signal that is already ~4x less responsive than the reference,
    # so its contribution has to be separable from the retrieval's own.
    filter_enabled: bool = True


@dataclass(frozen=True)
class ProcessingCfg:
    task: str
    stride: int
    visualize: bool
    log_results: bool


@dataclass(frozen=True)
class AblationCfg:
    max_noise_theta: float
    steps: int


@dataclass(frozen=True)
class MetricsCfg:
    window_min: int
    hop_min: int
    shift_search_s: int
    tolerance_s: float


@dataclass(frozen=True)
class OutputCfg:
    logs_dir: str
    assets_dir: str


@dataclass(frozen=True)
class Config:
    project: ProjectCfg
    dataset: DatasetCfg
    camera: CameraCfg
    matching: MatchingCfg
    clustering: ClusteringCfg
    temporal: TemporalCfg
    processing: ProcessingCfg
    ablation: AblationCfg
    metrics: MetricsCfg
    output: OutputCfg
    calibration_dir: str = field(repr=False, default="")
    source_path: str = field(repr=False, default="")
    sha256: str = field(repr=False, default="")
    resolved: dict = field(repr=False, default_factory=dict)
    calibration_record: Optional[Calibration] = field(repr=False, default=None)
