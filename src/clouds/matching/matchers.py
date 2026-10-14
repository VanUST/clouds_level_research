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
        # No torch.cuda.empty_cache() here: it forces a device-wide cache flush on
        # every pair (measured 19 ms/frame in the cProfile of a 40-frame
        # evaluation) and buys nothing — the caching allocator already reuses the
        # blocks this model frees, and the peak stays well inside 24 GB.
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
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, kp0, kp1)
        if return_conf and conf is not None:
            return kp0, kp1, conf
        return kp0, kp1

    # PURPOSE: Match many pairs in few forward passes. The pipeline runs one
    #          640x640 pair per forward pass while the GPU sits near idle, so this
    #          is purely a throughput measure for development and batch evaluation;
    #          it produces the same correspondences as _match, one pair at a time.
    # INPUTS: gray_pairs (list of (gray1, gray2) 2-D uint8 arrays), batch_size.
    # OUTPUTS: list of (kp0, kp1) in the same order. kornia pads a batch to the
    #          The batch output is a flat match list plus a batch_indexes array, so
    #          each item's matches are recovered by selecting on that index.
    # KEYWORDS: batching, throughput, loftr
    def _match_batch(gray_pairs, batch_size=8):
        results = []
        for start in range(0, len(gray_pairs), batch_size):
            chunk = gray_pairs[start:start + batch_size]
            # image_to_tensor yields (1, 1, H, W) for a 2-D grayscale array, i.e.
            # an already-batched single channel image, so concatenating along dim 0
            # gives the (B, 1, H, W) the backbone wants.
            t1 = torch.cat([K.image_to_tensor(a, False).float() / 255.0
                            for a, _ in chunk], 0).to(device)
            t2 = torch.cat([K.image_to_tensor(b, False).float() / 255.0
                            for _, b in chunk], 0).to(device)
            with torch.no_grad():
                corr = matcher({"image0": t1, "image1": t2})
            # For a batch, kornia returns a FLAT list of matches plus a parallel
            # batch_indexes array saying which pair each match came from. There is
            # no padding to strip, so no compute is wasted on a rich frame.
            k0 = corr["keypoints0"].detach().cpu().numpy()
            k1 = corr["keypoints1"].detach().cpu().numpy()
            idx = corr.get("batch_indexes")
            idx = (idx.detach().cpu().numpy().astype(int) if idx is not None
                   else np.zeros(len(k0), dtype=int))
            for i in range(len(chunk)):
                take = idx == i
                results.append((k0[take], k1[take]))
        return results

    _match.match_batch = _match_batch
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
        # No torch.cuda.empty_cache(): see the kornia backend's _match.
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
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, kp0, kp1)

        if return_conf and mconf is not None:
            return kp0, kp1, mconf
        return kp0, kp1

    return _match


# ---------------------------------------------------------------------------
# EfficientLoFTR (zju3dv) — detector-free, dense coarse + two-stage fine
# ---------------------------------------------------------------------------

# PURPOSE: Install the two compatibility shims the 2024 EfficientLoFTR reference
#          code needs on the modern stack this project runs (kornia 0.8.3, torch
#          2.14): kornia moved create_meshgrid out of kornia.utils.grid, and the
#          repo's utils.misc imports pytorch_lightning.utilities.rank_zero_only at
#          module import even for inference. Both are aliases only — no numerics
#          change — and neither is installed when the real module is importable.
# INPUTS: none.
# OUTPUTS: sys.modules has kornia.utils.grid and, when pytorch_lightning is absent,
#          a stub pytorch_lightning package exposing rank_zero_only.
# KEYWORDS: compatibility, shim, kornia, pytorch_lightning, efficient_loftr
def _install_efficient_loftr_shims():
    import sys
    import types

    try:
        import kornia.utils.grid  # noqa: F401
    except Exception:
        import kornia.utils as _ku
        mod = types.ModuleType("kornia.utils.grid")
        try:
            from kornia.geometry import create_meshgrid as _cmg
        except Exception:
            _cmg = _ku.create_meshgrid
        mod.create_meshgrid = _cmg
        sys.modules["kornia.utils.grid"] = mod

    try:
        import pytorch_lightning  # noqa: F401
    except Exception:
        pl = types.ModuleType("pytorch_lightning")
        pl.__path__ = []
        util = types.ModuleType("pytorch_lightning.utilities")

        def rank_zero_only(fn=None, *a, **k):
            return fn if fn is not None else (lambda g: g)

        util.rank_zero_only = rank_zero_only
        pl.utilities = util
        sys.modules["pytorch_lightning"] = pl
        sys.modules["pytorch_lightning.utilities"] = util


# PURPOSE: Create an EfficientLoFTR matcher (detector-free dense matcher whose
#          two-stage fine correlation is documented to remove the spatial variance
#          of LoFTR's fine soft-argmax), and return the callable the pipeline uses.
# INPUTS: config with optional repo_dir, weights, coarse_threshold keys; device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1), with
#          `match_batch` attached for throughput runs. Raises FileNotFoundError
#          with the exact remedy when the repo or weights are absent.
# KEYWORDS: efficient_loftr, detector_free, dense, subpixel, factory
@register_backend("efficient_loftr")
def _make_efficient_loftr(config, device="cpu"):
    import cv2
    import torch

    repo = config.get("repo_dir") or os.environ.get("EFFLOFTR_DIR", "/home/iustimov/effloftr")
    weights = config.get("weights") or os.path.join(repo, "weights", "eloftr_outdoor_state_dict.pt")
    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"EfficientLoFTR reference code not found at {repo}. "
            "Clone https://github.com/zju3dv/EfficientLoFTR and set EFFLOFTR_DIR."
        )
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"EfficientLoFTR weights not found at {weights}. Download "
            "https://huggingface.co/kornia/Efficient_LOFTR/resolve/main/eloftr_outdoor.ckpt "
            "and save its ['state_dict'] with torch.save."
        )

    _install_efficient_loftr_shims()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from copy import deepcopy
    from src.loftr import LoFTR as _ELoFTR, full_default_cfg, reparameter

    net = _ELoFTR(config=deepcopy(full_default_cfg))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net = reparameter(net).eval().to(device)

    def _run(g1, g2):
        h0, w0 = g1.shape[:2]
        h, w = h0 // 32 * 32, w0 // 32 * 32
        if (h, w) != (h0, w0):
            g1 = cv2.resize(g1, (w, h))
            g2 = cv2.resize(g2, (w, h))
        t1 = torch.from_numpy(np.ascontiguousarray(g1))[None][None].to(device).float() / 255.0
        t2 = torch.from_numpy(np.ascontiguousarray(g2))[None][None].to(device).float() / 255.0
        batch = {"image0": t1, "image1": t2}
        with torch.no_grad():
            net(batch)
        k0 = batch["mkpts0_f"].detach().cpu().numpy()
        k1 = batch["mkpts1_f"].detach().cpu().numpy()
        conf = batch.get("mconf")
        if conf is not None:
            conf = conf.detach().cpu().numpy()
        sx, sy = w0 / float(w), h0 / float(h)
        k0 = k0 * np.array([sx, sy])
        k1 = k1 * np.array([sx, sy])
        return k0, k1, conf

    def _match(img1, img2, visualize=False, return_conf=False):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        k0, k1, conf = _run(np.ascontiguousarray(g1), np.ascontiguousarray(g2))
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf and conf is not None:
            return k0, k1, conf
        return k0, k1

    def _match_batch(gray_pairs, batch_size=8):
        return [_match(a, b)[:2] for a, b in gray_pairs]

    _match.match_batch = _match_batch
    return _match


# ---------------------------------------------------------------------------
# RoMa (romatch) — dense, DINOv2-based, zero-shot
# ---------------------------------------------------------------------------

# PURPOSE: Create a RoMa (v2, `romatch`) dense matcher and return the callable the
#          pipeline uses. RoMa predicts a dense correspondence field on a regular
#          grid with a per-pixel certainty; a stride subsamples that field to the
#          correspondence arrays the rest of the project expects, and the certainty
#          threshold keeps only the matches RoMa itself trusts.
# INPUTS: config with optional coarse_res, upsample_res, grid_stride,
#         certainty_threshold, use_custom_corr keys; device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1).
# KEYWORDS: roma, romatch, dense, dinov2, zero_shot, factory
@register_backend("roma")
def _make_roma(config, device="cpu"):
    import cv2
    import torch
    from PIL import Image
    from romatch import roma_outdoor

    torch_device = torch.device(device)
    amp = torch.float16 if torch_device.type == "cuda" else torch.float32
    model = roma_outdoor(
        device=torch_device,
        amp_dtype=amp,
        coarse_res=int(config.get("coarse_res", 560)),
        upsample_res=int(config.get("upsample_res", 864)),
        symmetric=True,
        use_custom_corr=bool(config.get("use_custom_corr", False)),
        upsample_preds=bool(config.get("upsample_preds", True)),
    )
    stride = int(config.get("grid_stride", 4))
    cert_thresh = float(config.get("certainty_threshold", 0.5))

    def _match(img1, img2, visualize=False, return_conf=False):
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        h, w = rgb1.shape[:2]
        with torch.no_grad():
            warp, certainty = model.match(Image.fromarray(rgb1), Image.fromarray(rgb2))
        # match() returns (warp, certainty) on the upsampled grid; for symmetric
        # matching the width is doubled and the first half is the A->B direction.
        # In that half the four channels are ABSOLUTE coordinates, not a flow:
        # (x_A, y_A, x_B, y_B), each normalised to [-1, 1] over the match grid.
        # This was verified on the real frames with a known-shift test — with B
        # translated by +5 px the third channel moves by +5 px, while on an
        # image paired with itself channels 2,3 equal channels 0,1 to 0.02 px
        # (experiments/matchers/_diag_roma4.py).
        warp = warp[0].detach().cpu().numpy()          # (H, 2W, 4)
        certainty = certainty[0].detach().cpu().numpy()  # (H, 2W)
        hw = warp.shape[1] // 2
        ab = warp[:, :hw]                              # (x_A, y_A, x_B, y_B)
        cert = certainty[:, :hw]
        yy, xx = np.mgrid[0:ab.shape[0]:stride, 0:ab.shape[1]:stride]
        q = ab[yy, xx]
        c = cert[yy, xx]
        # normalized [-1,1] over the (square) match grid -> original pixels
        k0 = np.stack([(q[..., 0] + 1.0) * 0.5 * w, (q[..., 1] + 1.0) * 0.5 * h], axis=-1)
        k1 = np.stack([(q[..., 2] + 1.0) * 0.5 * w, (q[..., 3] + 1.0) * 0.5 * h], axis=-1)
        k0 = k0.reshape(-1, 2)
        k1 = k1.reshape(-1, 2)
        conf = c.reshape(-1)
        inside = ((k0[:, 0] >= 0) & (k0[:, 0] < w) & (k0[:, 1] >= 0) & (k0[:, 1] < h)
                  & (k1[:, 0] >= 0) & (k1[:, 0] < w) & (k1[:, 1] >= 0) & (k1[:, 1] < h))
        keep = inside & (conf >= cert_thresh)
        k0, k1, conf = k0[keep], k1[keep], conf[keep]
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, conf
        return k0, k1

    return _match


# ---------------------------------------------------------------------------
# Sparse-detector control (DISK + LightGlue)
# ---------------------------------------------------------------------------

# PURPOSE: Create a DISK + LightGlue sparse keypoint matcher, used only as the
#          control measurement for how a sparse detector behaves on cloud: it
#          demonstrates whether the low-contrast cloud interior yields any
#          keypoints at all. It is not a candidate for production use here.
# INPUTS: config with optional num_features, disk_weights keys; device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1).
# KEYWORDS: disk, lightglue, sparse, control, factory
@register_backend("disk_lightglue")
def _make_disk_lightglue(config, device="cpu"):
    import cv2
    import kornia.feature as KF
    import torch

    torch_device = torch.device(device)
    extractor = KF.DISK.from_pretrained(
        config.get("disk_weights", "depth"), device=torch_device,
    ).eval()
    # DISK picks its top-n keypoints from the detection heatmap via this attribute;
    # from_pretrained does not take it as an argument.
    extractor.n = int(config.get("num_features", 4096))
    matcher = KF.LightGlueMatcher("disk").to(torch_device).eval()

    def _match(img1, img2, visualize=False, return_conf=False):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        h, w = g1.shape[:2]
        x1 = torch.from_numpy(np.ascontiguousarray(g1))[None, None].float().to(torch_device).repeat(1, 3, 1, 1) / 255.0
        x2 = torch.from_numpy(np.ascontiguousarray(g2))[None, None].float().to(torch_device).repeat(1, 3, 1, 1) / 255.0
        with torch.no_grad():
            f1 = extractor(x1)[0]
            f2 = extractor(x2)[0]
            if len(f1.keypoints) < 2 or len(f2.keypoints) < 2:
                empty = np.empty((0, 2), dtype=float)
                return (empty, empty, np.empty(0)) if return_conf else (empty, empty)
            l1 = KF.laf_from_center_scale_ori(f1.keypoints[None])
            l2 = KF.laf_from_center_scale_ori(f2.keypoints[None])
            sc, idx = matcher(f1.descriptors, f2.descriptors, l1, l2, (h, w), (h, w))
        idx = idx.detach().cpu().numpy()
        k0 = f1.keypoints.detach().cpu().numpy()[idx[:, 0]]
        k1 = f2.keypoints.detach().cpu().numpy()[idx[:, 1]]
        conf = sc.detach().cpu().numpy().reshape(-1)
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, conf
        return k0, k1

    return _match


# ---------------------------------------------------------------------------
# Depth Anything 3 (ByteDance) — depth + pose turned into a disparity field
# ---------------------------------------------------------------------------

# PURPOSE: Turn Depth Anything 3's per-view depth map and predicted relative pose
#          into a dense correspondence field, so a purely-depth model can be
#          compared on the same disparity criteria as the matchers rather than on
#          its own depth metric. Back-projection is scale-invariant: the model's
#          depth and its predicted translation share one unknown scale, which
#          cancels in the projection, so no metric assumption is imported. What
#          the network contributes is therefore a dense 2-D shift field; whether
#          that field is accurate at a 45 m baseline and 1-2 km range is the
#          measurement, not an assumption.
# INPUTS: config with optional weights (local dir), process_res, grid_stride, device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1).
# KEYWORDS: depth_anything_3, da3, monocular, multiview, pose, factory
@register_backend("da3")
def _make_da3(config, device="cpu"):
    import cv2
    import torch
    import types

    # api.py imports the Gaussian-splat export helpers at module import time; those
    # need moviepy/gsplat encoders that are irrelevant to geometry. A stub export
    # module keeps the depth/pose API importable without them.
    if "depth_anything_3.utils.export" not in sys.modules:
        stub = types.ModuleType("depth_anything_3.utils.export")
        stub.export = lambda *a, **k: None
        sys.modules["depth_anything_3.utils.export"] = stub
    from depth_anything_3.api import DepthAnything3

    weights = config.get("weights") or os.environ.get("DA3_WEIGHTS", "/home/iustimov/da3_base")
    if not os.path.isdir(weights):
        raise FileNotFoundError(
            f"DA3 weights not found at {weights}. Download depth-anything/DA3-BASE "
            "(model.safetensors + config.json) and set DA3_WEIGHTS.")
    model = DepthAnything3.from_pretrained(weights).to(device).eval()
    process_res = int(config.get("process_res", 504))
    stride = int(config.get("grid_stride", 8))
    min_conf = float(config.get("min_conf", 0.0))

    def _match(img1, img2, visualize=False, return_conf=False):
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        h, w = rgb1.shape[:2]
        with torch.no_grad():
            pred = model.inference([rgb1, rgb2], process_res=process_res)
        depth = np.asarray(pred.depth)
        conf = np.asarray(pred.conf) if pred.conf is not None else None
        E = np.asarray(pred.extrinsics, dtype=float)   # (N,3,4) world -> camera
        K = np.asarray(pred.intrinsics, dtype=float)   # (N,3,3)
        hh, ww = depth.shape[1], depth.shape[2]
        R1, t1 = E[0][:, :3], E[0][:, 3]
        R2, t2 = E[1][:, :3], E[1][:, 3]
        K1i, K2 = np.linalg.inv(K[0]), K[1]
        yy, xx = np.mgrid[0:hh:stride, 0:ww:stride]
        uv = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
        rays = (K1i @ uv.T).T
        d1 = depth[0][yy, xx].reshape(-1)
        p1 = rays * d1[:, None]
        p_world = (R1.T @ (p1 - t1).T).T
        p2 = (R2 @ p_world.T).T + t2
        z2 = p2[:, 2]
        ok = z2 > 1e-6
        uv2 = (K2 @ p2.T).T
        uv2 = uv2[:, :2] / z2[:, None]
        sx, sy = w / float(ww), h / float(hh)
        k0 = np.stack([xx.ravel() * sx, yy.ravel() * sy], axis=1)
        k1 = np.stack([uv2[:, 0] * sx, uv2[:, 1] * sy], axis=1)
        c = (conf[0][yy, xx].reshape(-1) if conf is not None
             else np.ones(len(k0)))
        keep = (ok & np.isfinite(k1).all(axis=1)
                & (k1[:, 0] >= 0) & (k1[:, 0] < w) & (k1[:, 1] >= 0) & (k1[:, 1] < h))
        if min_conf > 0:
            keep &= (c >= min_conf)
        k0, k1, c = k0[keep], k1[keep], c[keep]
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, c
        return k0, k1

    return _match


# ---------------------------------------------------------------------------
# MASt3R (naver) — dense descriptor correspondences from a pointmap model
# ---------------------------------------------------------------------------

# PURPOSE: Create a MASt3R matcher and expose its dense correspondences through the
#          project's interface. Two things are deliberately NOT used: the model's
#          own pointmaps and its (metric) scale. Only the dense per-pixel descriptor
#          field is matched, so the height still comes from this project's exact
#          triangulation and the known 45 m baseline. That keeps MASt3R comparable
#          with the other matchers and avoids importing a second, unvalidated
#          distance model into the measurement.
# INPUTS: config with optional repo_dir, weights, image_size, subsample keys; device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1) in original pixels.
#          Raises FileNotFoundError naming the missing path when the repo or the
#          weights are absent.
# KEYWORDS: mast3r, dust3r, dense, descriptors, factory
@register_backend("mast3r")
def _make_mast3r(config, device="cpu"):
    import cv2
    import torch
    from PIL import Image

    repo = config.get("repo_dir") or os.environ.get("MAST3R_DIR", "/home/iustimov/mast3r")
    weights = config.get("weights") or os.path.join(repo, "weights")
    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"MASt3R code not found at {repo}. "
            "git clone --recursive https://github.com/naver/mast3r and set MAST3R_DIR.")
    if not os.path.isdir(weights):
        raise FileNotFoundError(
            f"MASt3R weights not found at {weights}. Download "
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric (model.safetensors + config.json).")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import extract_correspondences_nonsym
    from dust3r.inference import inference
    from dust3r.utils.image import ImgNorm, _resize_pil_image

    model = AsymmetricMASt3R.from_pretrained(weights).to(device).eval()
    size = int(config.get("image_size", 512))
    subsample = int(config.get("subsample", 8))
    min_conf = float(config.get("min_conf", 0.0))

    def _view(rgb):
        # MASt3R wants a square patch grid. The project's crop is already square, so
        # with square_ok the resize is a plain scale to size x size with no crop.
        pil = Image.fromarray(rgb)
        pil = _resize_pil_image(pil, size)
        return {"img": ImgNorm(pil)[None], "true_shape": np.int32([pil.size[::-1]]),
                "idx": 0, "instance": "0"}

    def _match(img1, img2, visualize=False, return_conf=False):
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        h, w = rgb1.shape[:2]
        with torch.no_grad():
            out = inference([(_view(rgb1), _view(rgb2))], model, device,
                            batch_size=1, verbose=False)
        p1, p2 = out["pred1"], out["pred2"]
        c1 = p1["desc_conf"][0].detach().cpu().numpy()
        c2 = p2["desc_conf"][0].detach().cpu().numpy()
        xy1, xy2, conf = extract_correspondences_nonsym(
            p1["desc"][0], p2["desc"][0], c1, c2, subsample=subsample,
            device=device, ptmap_key="desc", pixel_tol=0)
        xy1 = xy1.detach().cpu().numpy() if torch.is_tensor(xy1) else np.asarray(xy1)
        xy2 = xy2.detach().cpu().numpy() if torch.is_tensor(xy2) else np.asarray(xy2)
        conf = conf.detach().cpu().numpy() if torch.is_tensor(conf) else np.asarray(conf)
        # the matcher's grid is size x size over the whole square crop
        k0 = xy1 * np.array([w / size, h / size])
        k1 = xy2 * np.array([w / size, h / size])
        keep = (conf >= min_conf) if min_conf > 0 else np.ones(len(k0), bool)
        k0, k1, conf = k0[keep], k1[keep], conf[keep]
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, conf
        return k0, k1

    return _match


# ---------------------------------------------------------------------------
# DKM (Parskatt/DKM) — dense kernelized matching
# ---------------------------------------------------------------------------

# PURPOSE: Create a DKM v3 outdoor dense matcher and expose it as correspondences.
#          Like RoMa it predicts a dense field on a regular grid with a certainty;
#          its output channels are absolute coordinates in the two images, verified
#          with the same known-shift test used for RoMa. DKM's outdoor model is
#          trained at a 4:3 resolution (864x1152), so a square crop is resized
#          anisotropically; both images are distorted identically and the mapping
#          back to the original 640x640 handles the anisotropy, but this is a
#          caveat on its numbers.
# INPUTS: config with optional grid_stride, certainty_threshold keys; device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1).
# KEYWORDS: dkm, dense, kernelized, factory
@register_backend("dkm")
def _make_dkm(config, device="cpu"):
    import cv2
    import torch
    from PIL import Image
    from dkm import DKMv3_outdoor

    model = DKMv3_outdoor(device=torch.device(device))
    stride = int(config.get("grid_stride", 8))
    cert_thresh = float(config.get("certainty_threshold", 0.5))

    def _match(img1, img2, visualize=False, return_conf=False):
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        h, w = rgb1.shape[:2]
        with torch.no_grad():
            warp, certainty = model.match(Image.fromarray(rgb1), Image.fromarray(rgb2))
        wrp = warp.detach().cpu().numpy()
        cert = certainty.detach().cpu().numpy()
        hw = wrp.shape[1] // 2
        ab = wrp[:, :hw]
        cert = cert[:, :hw]
        yy, xx = np.mgrid[0:ab.shape[0]:stride, 0:ab.shape[1]:stride]
        q = ab[yy, xx]
        c = cert[yy, xx]
        k0 = np.stack([(q[..., 0] + 1.0) * 0.5 * w, (q[..., 1] + 1.0) * 0.5 * h], axis=-1)
        k1 = np.stack([(q[..., 2] + 1.0) * 0.5 * w, (q[..., 3] + 1.0) * 0.5 * h], axis=-1)
        k0 = k0.reshape(-1, 2)
        k1 = k1.reshape(-1, 2)
        conf = c.reshape(-1)
        inside = ((k0[:, 0] >= 0) & (k0[:, 0] < w) & (k0[:, 1] >= 0) & (k0[:, 1] < h)
                  & (k1[:, 0] >= 0) & (k1[:, 0] < w) & (k1[:, 1] >= 0) & (k1[:, 1] < h))
        keep = inside & (conf >= cert_thresh)
        k0, k1, conf = k0[keep], k1[keep], conf[keep]
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, conf
        return k0, k1

    return _match


# ---------------------------------------------------------------------------
# Dense classical flow / block matching (cheap reference points)
# ---------------------------------------------------------------------------

# PURPOSE: Create a dense optical-flow matcher (OpenCV DIS) and expose it through
#          the same correspondence interface. Optical flow is the natural classical
#          fit for a smooth low-texture field because it needs no corners; unlike
#          stereo block matching it does not assume rectified epipolar rows, so it
#          can be compared fairly on this non-row-rectified pair.
# INPUTS: config with optional preset, grid_stride, min_flow_px keys; device (CPU).
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1), dense on a grid.
# KEYWORDS: optical_flow, dis, dense, classical, factory
@register_backend("cv2_dis_flow")
def _make_cv2_dis_flow(config, device="cpu"):
    import cv2

    preset = {"medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
              "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
              "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}.get(
                  str(config.get("preset", "medium")), cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    stride = int(config.get("grid_stride", 8))
    max_flow = float(config.get("max_flow_px", 64.0))

    def _match(img1, img2, visualize=False, return_conf=False):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        g1 = np.ascontiguousarray(g1)
        g2 = np.ascontiguousarray(g2)
        dis = cv2.DISOpticalFlow_create(preset)
        dis.setUseSpatialPropagation(True)
        flow = dis.calc(g1, g2, None)
        h, w = g1.shape[:2]
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        k0 = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)
        f = flow[yy, xx].reshape(-1, 2)
        k1 = k0 + f
        mag = np.hypot(f[:, 0], f[:, 1])
        keep = ((k1[:, 0] >= 0) & (k1[:, 0] < w) & (k1[:, 1] >= 0) & (k1[:, 1] < h)
                & (mag <= max_flow))
        k0, k1, mag = k0[keep], k1[keep], mag[keep]
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, mag
        return k0, k1

    return _match


# PURPOSE: Create a dense stereo block matcher (OpenCV StereoSGBM) as the reference
#          point the project owner asked for, together with the caveat its result
#          must carry: SGBM assumes the two images are row-rectified, and this pair
#          is not (an affine warp plus a measured residual vertical disparity). Its
#          matches are therefore expected to encode our geometry error, and the
#          comparison is what demonstrates that rather than assuming it.
# INPUTS: config with optional num_disparities, block_size, grid_stride keys; device.
# OUTPUTS: callable (img0, img1, visualize=False) -> (kp0, kp1); kp1 differs from
#          kp0 only along +x by the SGBM disparity, by construction of the model.
# KEYWORDS: sgbm, stereo, dense, classical, rectification, control
@register_backend("cv2_sgbm")
def _make_cv2_sgbm(config, device="cpu"):
    import cv2

    ndisp = int(config.get("num_disparities", 64))
    block = int(config.get("block_size", 5))
    stride = int(config.get("grid_stride", 8))
    bad = float(config.get("invalid_disp", -16.0))

    def _match(img1, img2, visualize=False, return_conf=False):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        g1 = np.ascontiguousarray(g1)
        g2 = np.ascontiguousarray(g2)
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=ndisp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(g1, g2).astype(np.float32) / 16.0
        h, w = g1.shape[:2]
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        k0 = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)
        d = disp[yy, xx].reshape(-1)
        valid = (d > bad) & (k0[:, 0] - d >= 0)
        k0 = k0[valid]
        k1 = k0.copy()
        k1[:, 0] = k0[:, 0] - d[valid]
        if visualize and len(k0) > 0:
            from ..viz.plots import visualize_loftr_matches
            visualize_loftr_matches(img1, img2, k0, k1)
        if return_conf:
            return k0, k1, d[valid]
        return k0, k1

    return _match


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# PURPOSE: Create and return a feature matcher function for the given backend name.
# INPUTS: backend_name (str), config (dict), device (str)
# OUTPUTS: callable with signature (img0: np.ndarray, img1: np.ndarray, visualize: bool) -> (kp0, kp1)
# KEYWORDS: factory, matcher, create, backend
# PURPOSE: Report whether a backend can match a list of pairs in one call.
# INPUTS: the callable returned by create_matcher.
# OUTPUTS: True when it exposes a match_batch attribute.
# KEYWORDS: batching, capability
def supports_batching(matcher) -> bool:
    return callable(getattr(matcher, "match_batch", None))


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
