# PURPOSE: Switchable feature matcher backends for stereo image matching.
#          Provides a unified interface with multiple backend options (Kornia LoFTR, MatchAnything Eloftr).
# INPUTS: Backend name (str), two images (np.ndarray), config dict.
# OUTPUTS: Tuple of (keypoints0, keypoints1) as np.ndarrays.
# KEYWORDS: feature_matcher, loftr, eloftr, matchanything, switchable, backend, stereo
import sys
import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY = {}


def register_backend(name):
    """Decorator to register a matcher backend factory."""
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_available_backends():
    """Return list of registered backend names."""
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Kornia LoFTR (default)
# ---------------------------------------------------------------------------

# PURPOSE: Create a Kornia LoFTR matcher instance and return a callable that accepts two images
#          and returns (kp0, kp1) keypoint arrays.
# INPUTS: config (dict with match_thresh key), device (str)
# OUTPUTS: callable (img0, img1, visualize=False) -> (np.ndarray, np.ndarray)
# KEYWORDS: kornia, loftr, factory, matcher
@register_backend("kornia_loftr")
def _make_kornia_loftr(config, device="cpu"):
    import cv2
    import torch
    import kornia as K
    import numpy as np

    default_cfg = {
        "backbone_type": "ResNetFPN",
        "resolution": (8, 2),
        "fine_window_size": 5,
        "fine_concat_coarse_feat": True,
        "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
        "coarse": {
            "d_model": 256, "d_ffn": 256, "nhead": 8,
            "layer_names": ["self", "cross", "self", "cross", "self", "cross", "self", "cross"],
            "attention": "linear", "temp_bug_fix": False,
        },
        "match_coarse": {
            "thr": config.get("match_threshold", 0.5),
            "border_rm": 2, "match_type": "dual_softmax",
            "dsmax_temperature": 0.1, "skh_iters": 3,
            "skh_init_bin_score": 1.0, "skh_prefilter": True,
            "train_coarse_percent": 0.4, "train_pad_num_gt_min": 200,
        },
        "fine": {
            "d_model": 128, "d_ffn": 128, "nhead": 8,
            "layer_names": ["self", "cross"], "attention": "linear",
        },
    }

    matcher = K.feature.LoFTR(pretrained="outdoor", config=default_cfg).to(device)
    matcher.eval()

    def _match(img1, img2, visualize=False, return_conf=False):
        torch.cuda.empty_cache()
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        t1 = K.image_to_tensor(img1_gray, False).float() / 255.0
        t2 = K.image_to_tensor(img2_gray, False).float() / 255.0
        t1, t2 = t1.to(device), t2.to(device)
        with torch.no_grad():
            corr = matcher({"image0": t1, "image1": t2})
        kp0 = corr["keypoints0"].cpu().numpy()
        kp1 = corr["keypoints1"].cpu().numpy()
        conf = corr.get("confidence", None)
        if conf is not None:
            conf = conf.cpu().numpy()
        if visualize:
            from .visualization import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, kp0, kp1)
        if return_conf and conf is not None:
            return kp0, kp1, conf
        return kp0, kp1

    return _match


# ---------------------------------------------------------------------------
# MatchAnything Eloftr
# ---------------------------------------------------------------------------

# PURPOSE: Create a MatchAnything Eloftr matcher instance and return a callable that accepts two images
#          and returns (kp0, kp1) keypoint arrays. Requires third_party/MatchAnything to be set up.
# INPUTS: config (dict with match_threshold, img_resize, model_path, third_party_dir keys), device (str)
# OUTPUTS: callable (img0, img1, visualize=False) -> (np.ndarray, np.ndarray)
# KEYWORDS: matchanything, eloftr, efficient_loftr, factory, matcher
@register_backend("matchanything_eloftr")
def _make_matchanything_eloftr(config, device="cpu"):
    third_party_dir = config.get("third_party_dir", os.path.join(os.path.dirname(__file__), "..", "third_party"))
    ma_path = Path(third_party_dir) / "MatchAnything"
    if not ma_path.exists():
        raise FileNotFoundError(
            f"MatchAnything third-party code not found at {ma_path}. "
            "Run: python tools/setup_matchanything.py"
        )

    # Inject MatchAnything root and all its sub-dependencies into sys.path.
    # Needed because MatchAnything's internal code uses absolute imports from
    # third-party packages (ROMA, EfficientLoFTR, etc.) under its own tree.
    ma_root = str(ma_path.resolve())
    ma_parent = str(ma_path.parent.resolve())
    roma_path = os.path.join(ma_root, "third_party", "ROMA")
    lofts_paths = [
        os.path.join(ma_root, "third_party", "EfficientLoFTR"),
    ]
    for p in [ma_root, ma_parent, roma_path] + lofts_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import cv2
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F

    # Stash our project's 'src' module so MatchAnything can import its own 'src'.
    _our_src = sys.modules.pop("src", None)
    _our_src_sub = {k: v for k, v in sys.modules.items() if k.startswith("src.")}
    for k in _our_src_sub:
        del sys.modules[k]

    try:
        from MatchAnything.src.lightning.lightning_loftr import PL_LoFTR
        from MatchAnything.src.config.default import get_cfg_defaults

        base_cfg = get_cfg_defaults()
        config_path = str(ma_path / "configs" / "models" / "eloftr_model.py")
        base_cfg.merge_from_file(config_path)
    finally:
        if _our_src is not None:
            sys.modules["src"] = _our_src
        for k, v in _our_src_sub.items():
            sys.modules[k] = v

    base_cfg.METHOD = "matchanything_eloftr"
    base_cfg.LOFTR.MATCH_COARSE.THR = config.get("match_threshold", 0.001)
    if base_cfg.LOFTR.COARSE.ROPE:
        assert base_cfg.DATASET.NPE_NAME is not None
        base_cfg.LOFTR.COARSE.NPE = [832, 832, config.get("img_resize", 832), config.get("img_resize", 832)]

    model_path = config.get("model_path", str(ma_path / "weights" / "matchanything_eloftr.ckpt"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Download from https://huggingface.co/LittleFrog/MatchAnything_checkpoints"
        )

    pl_model = PL_LoFTR(base_cfg, pretrained_ckpt=model_path, test_mode=True)
    net = pl_model.matcher.eval().to(device)
    dfactor = 32

    def _resize_and_pad(img_gray, df=32):
        h, w = img_gray.shape
        w_new = int(w // df * df)
        h_new = int(h // df * df)
        resized = cv2.resize(img_gray.astype(np.float32), (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)
        pad_size = max(h_new, w_new)
        padded = np.zeros((pad_size, pad_size), dtype=np.float32)
        padded[:h_new, :w_new] = resized
        mask = np.zeros((pad_size, pad_size), dtype=bool)
        mask[:h_new, :w_new] = True
        h_scale = h / h_new
        w_scale = w / w_new
        return padded, mask, np.array([h_scale, w_scale])

    def _dict_to_device(d, dev):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(dev)
            elif isinstance(v, dict):
                _dict_to_device(v, dev)
            elif isinstance(v, list):
                d[k] = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in v]
        return d

    def _match(img1, img2, visualize=False, return_conf=False):
        torch.cuda.empty_cache()
        orig_size0 = np.array(img1.shape[:2])
        orig_size1 = np.array(img2.shape[:2])

        img1_gray = np.array(Image.fromarray(img1).convert("L")).astype(np.float32)
        img2_gray = np.array(Image.fromarray(img2).convert("L")).astype(np.float32)

        (p0, mask0, hw0), (p1, mask1, hw1) = (
            _resize_and_pad(img1_gray, dfactor),
            _resize_and_pad(img2_gray, dfactor),
        )

        t0 = torch.from_numpy(p0)[None][None] / 255.0
        t1 = torch.from_numpy(p1)[None][None] / 255.0

        batch = {
            "image0": t0, "image1": t1,
            "image0_rgb_origin": t0, "image1_rgb_origin": t1,
            "origin_img_size0": torch.from_numpy(orig_size0)[None],
            "origin_img_size1": torch.from_numpy(orig_size1)[None],
        }

        if mask0 is not None:
            m0 = torch.from_numpy(mask0).to(device)
            m1 = torch.from_numpy(mask1).to(device)
            stacked = torch.stack([m0, m1], dim=0)[None].float()
            ts_mask = F.interpolate(
                stacked,
                scale_factor=0.125, mode="nearest",
                recompute_scale_factor=False,
            )[0].bool()
            batch["mask0"] = ts_mask[0:1]
            batch["mask1"] = ts_mask[1:2]

        batch = _dict_to_device(batch, device)
        net(batch)

        mkpts0 = batch["mkpts0_f"].cpu()
        mkpts1 = batch["mkpts1_f"].cpu()
        mconf = batch.get("mconf", None)

        mkpts0 = mkpts0 * torch.tensor(hw0)[[1, 0]]
        mkpts1 = mkpts1 * torch.tensor(hw1)[[1, 0]]

        kp0 = mkpts0.detach().numpy()
        kp1 = mkpts1.detach().numpy()

        if mconf is not None:
            mconf = mconf.cpu().numpy()

        if visualize and len(kp0) > 0:
            from .visualization import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, kp0, kp1)

        if return_conf and mconf is not None:
            return kp0, kp1, mconf
        return kp0, kp1

    return _match


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# PURPOSE: Create and return a feature matcher function for the given backend name.
# INPUTS: backend_name (str), config (dict), device (str)
# OUTPUTS: callable with signature (img0: np.ndarray, img1: np.ndarray, visualize: bool) -> (kp0, kp1)
# KEYWORDS: factory, matcher, create, backend
def create_matcher(backend_name, config=None, device=None):
    """
    Create a feature matcher function for the specified backend.

    Args:
        backend_name: One of 'kornia_loftr' or 'matchanything_eloftr'.
        config: Configuration dict passed to the backend factory.
        device: Torch device string. Auto-detected if None.

    Returns:
        A callable: match(img0, img1, visualize=False) -> (np.ndarray, np.ndarray)
    """
    if backend_name not in _REGISTRY:
        raise ValueError(
            f"Unknown matcher backend '{backend_name}'. "
            f"Available: {get_available_backends()}"
        )
    if config is None:
        config = {}
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    factory = _REGISTRY[backend_name]
    return factory(config, device)


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
