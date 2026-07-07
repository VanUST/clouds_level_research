# PURPOSE: Run MatchAnything matchers on selected image pairs, visualize raw matches and cluster assignments,
#          and save all visualization plots (matches overlay, clustered matches, shift vectors) to an output dir.
#          Supports multiple backends via --backend argument (eloftr, roma).
# INPUTS: IMAGE_DIR (path to stereo images), image timestamps list, OUTPUT_DIR
# OUTPUTS: Saved PNG plots per image pair: raw_matches, clustered_matches, shift_clusters
# KEYWORDS: matchanything, visualization, matches, clusters, hdbscan, stereo, roma
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from collections import Counter

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

OUTPUT_DIR = "report_assets/matchanything_viz"
IMAGE_DIR = config.IMAGE_DIR
AFFINE_MATRIX = config.AFFINE_MATRIX
ORIGINAL_SIZE = config.ORIGINAL_SIZE
TARGET_SIZE = config.TARGET_SIZE
THIRD_PARTY_DIR = config.MATCHER_BACKEND_KWARGS.get("third_party_dir", "third_party")
MATCH_THRESH = config.LOFTR_MATCH_THRESHOLD

# Pick a few interesting timestamps
SELECTED_TIMESTAMPS = [
    "2025-05-23T12-00-01",
    "2025-05-23T12-05-01",
    "2025-05-23T12-30-01",
]


# ---------------------------------------------------------------------------
# MatchAnything matcher factory
# ---------------------------------------------------------------------------
def init_matcher(backend="eloftr"):
    device = "cpu"
    ma_path = os.path.join(THIRD_PARTY_DIR, "MatchAnything")
    ma_root = os.path.abspath(ma_path)
    ma_parent = os.path.abspath(os.path.join(ma_path, ".."))
    roma_path = os.path.join(ma_root, "third_party", "ROMA")
    for p in [ma_root, ma_parent, roma_path]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import torch.nn.functional as F
    from PIL import Image

    _our_src = sys.modules.pop("src", None)
    _our_src_sub = {k: v for k, v in sys.modules.items() if k.startswith("src.")}
    for k in _our_src_sub:
        del sys.modules[k]

    try:
        from MatchAnything.src.lightning.lightning_loftr import PL_LoFTR
        from MatchAnything.src.config.default import get_cfg_defaults
        base_cfg = get_cfg_defaults()

        if backend == "eloftr":
            config_path = os.path.join(ma_root, "configs", "models", "eloftr_model.py")
            model_name = "matchanything_eloftr"
        else:
            config_path = os.path.join(ma_root, "configs", "models", "roma_model.py")
            model_name = "matchanything_roma"

        base_cfg.merge_from_file(config_path)
    finally:
        if _our_src is not None:
            sys.modules["src"] = _our_src
        for k, v in _our_src_sub.items():
            sys.modules[k] = v

    base_cfg.METHOD = model_name
    base_cfg.LOFTR.MATCH_COARSE.THR = MATCH_THRESH

    if backend == "eloftr" and base_cfg.LOFTR.COARSE.ROPE:
        base_cfg.LOFTR.COARSE.NPE = [832, 832, 832, 832]

    if backend == "roma" and str(torch.device(device)) == "cpu":
        base_cfg.LOFTR.FP16 = False
        base_cfg.ROMA.MODEL.AMP = False

    model_path = os.path.join(ma_root, "weights", f"{model_name}.ckpt")
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        sys.exit(1)

    pl_model = PL_LoFTR(base_cfg, pretrained_ckpt=model_path, test_mode=True)
    net = pl_model.matcher.eval().to(device)
    print(f"Loaded {model_name} on {device}")
    return net, device, model_name


def resize_and_pad(img, df=32):
    if len(img.shape) == 2:
        h, w = img.shape
        ch = 1
    else:
        h, w, ch = img.shape
    w_new = int(w // df * df)
    h_new = int(h // df * df)
    resized = cv2.resize(img.astype(np.float32), (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)
    pad_size = max(h_new, w_new)
    if ch == 1:
        padded = np.zeros((pad_size, pad_size), dtype=np.float32)
        padded[:h_new, :w_new] = resized
        mask = np.zeros((pad_size, pad_size), dtype=bool)
        mask[:h_new, :w_new] = True
    else:
        padded = np.zeros((pad_size, pad_size, ch), dtype=np.float32)
        padded[:h_new, :w_new, :] = resized
        mask = np.zeros((pad_size, pad_size), dtype=bool)
        mask[:h_new, :w_new] = True
    h_scale = h / h_new
    w_scale = w / w_new
    return padded, mask, np.array([h_scale, w_scale])


def match_pair(net, device, img1_color, img2_color, backend="eloftr"):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np
    import cv2

    orig_size0 = np.array(img1_color.shape[:2])
    orig_size1 = np.array(img2_color.shape[:2])

    if backend == "roma":
        img1_proc = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB).astype(np.float32)
        img2_proc = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        img1_proc = np.array(Image.fromarray(img1_color).convert("L")).astype(np.float32)
        img2_proc = np.array(Image.fromarray(img2_color).convert("L")).astype(np.float32)

    (p0, mask0, hw0), (p1, mask1, hw1) = resize_and_pad(img1_proc, 32), resize_and_pad(img2_proc, 32)
    if backend == "roma":
        t0 = torch.from_numpy(p0).permute(2, 0, 1)[None] / 255.0
        t1 = torch.from_numpy(p1).permute(2, 0, 1)[None] / 255.0
    else:
        t0 = torch.from_numpy(p0)[None][None] / 255.0
        t1 = torch.from_numpy(p1)[None][None] / 255.0

    batch = {"image0": t0, "image1": t1,
             "image0_rgb_origin": t0, "image1_rgb_origin": t1,
             "origin_img_size0": torch.from_numpy(orig_size0)[None],
             "origin_img_size1": torch.from_numpy(orig_size1)[None]}

    if mask0 is not None:
        m0 = torch.from_numpy(mask0).to(device)
        m1 = torch.from_numpy(mask1).to(device)
        stacked = torch.stack([m0, m1], dim=0)[None].float()
        ts_mask = F.interpolate(stacked, scale_factor=0.125, mode="nearest", recompute_scale_factor=False)[0].bool()
        batch["mask0"] = ts_mask[0:1]
        batch["mask1"] = ts_mask[1:2]

    def to_device(d, dev):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(dev)
        return d

    batch = to_device(batch, device)
    net(batch)

    mkpts0 = batch["mkpts0_f"].cpu() * torch.tensor(hw0)[[1, 0]]
    mkpts1 = batch["mkpts1_f"].cpu() * torch.tensor(hw1)[[1, 0]]
    mconf = batch.get("mconf", None)
    return mkpts0.detach().numpy(), mkpts1.detach().numpy(), (mconf.detach().numpy() if mconf is not None else None)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_raw_matches(img1, img2, kp0, kp1, scores, save_path):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    composite = np.hstack((img1, img2))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    max_pts = min(500, len(kp0))
    idx = np.argsort(scores)[::-1][:max_pts] if scores is not None else range(max_pts)
    for i in idx:
        x1, y1 = kp0[i]
        x2, y2 = kp1[i][0] + w1, kp1[i][1]
        ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=0.3, alpha=0.4)
        ax.scatter(x1, y1, s=2, c="lime", alpha=0.6)
        ax.scatter(x2, y2, s=2, c="orange", alpha=0.6)
    ax.set_title(f"Raw Matches: {len(kp0)} total, showing top {max_pts}")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def cluster_shifts(shifts, min_cluster_size=3, min_samples=3, epsilon=0.0):
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        allow_single_cluster=True,
        store_centers="centroid",
    )
    if len(shifts) < min_samples:
        return np.full(len(shifts), -1, dtype=int), np.empty((0, 2))
    hdbscan.fit(shifts)
    labels = hdbscan.labels_
    centers = hdbscan.centroids_ if hasattr(hdbscan, "centroids_") and hdbscan.centroids_ is not None else np.empty((0, 2))
    return labels, np.array(centers) if len(centers) > 0 else np.empty((0, 2))


def save_clustered_matches(img1, img2, kp0, kp1, labels, save_path):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    composite = np.hstack((img1, img2))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = (0.5, 0.5, 0.5) if lbl == -1 else cmap(i)[:3]
        pts0 = kp0[mask]
        pts1 = kp1[mask]
        for j in range(len(pts0)):
            x1, y1 = pts0[j]
            x2, y2 = pts1[j][0] + w1, pts1[j][1]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.3, alpha=0.5)
        ax.scatter(pts0[:, 0], pts0[:, 1], s=3, c=[color], label=f"Cluster {lbl}" if lbl != -1 else "Noise")

    ax.set_title(f"Clustered Matches ({len(unique_labels)} clusters)")
    ax.legend(fontsize=8, loc="upper right", markerscale=3)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_shift_clusters(shifts, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = (0.5, 0.5, 0.5) if lbl == -1 else cmap(i)[:3]
        ax.scatter(shifts[mask, 0], shifts[mask, 1], s=5, c=[color],
                   label=f"Cluster {lbl} (n={mask.sum()})" if lbl != -1 else f"Noise (n={mask.sum()})")
        if lbl != -1:
            center = shifts[mask].mean(axis=0)
            ax.scatter(center[0], center[1], s=80, c=[color], marker="X", edgecolors="k", linewidths=1)
    ax.set_title(f"Shift Vectors by Cluster ({len(unique_labels)} clusters)")
    ax.set_xlabel("dx (px)"); ax.set_ylabel("dy (px)")
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_and_visualize(backend="eloftr"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Initializing {backend} matcher...")
    net, device, model_name = init_matcher(backend)

    for ts_str in SELECTED_TIMESTAMPS:
        img1_path = os.path.join(IMAGE_DIR, f"img-{ts_str}devID1.jpg")
        img2_path = os.path.join(IMAGE_DIR, f"img-{ts_str}devID2.jpg")
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Skipping {ts_str}: images not found")
            continue

        print(f"\nProcessing {ts_str}...")

        # Load + preprocess
        img1_orig = cv2.imread(img1_path)
        img2_orig = cv2.imread(img2_path)
        img1 = cv2.warpAffine(img1_orig, AFFINE_MATRIX, dsize=ORIGINAL_SIZE)
        img1_crop = img1[ORIGINAL_SIZE[0]//2 - TARGET_SIZE[0]//2:ORIGINAL_SIZE[0]//2 + TARGET_SIZE[0]//2,
                          ORIGINAL_SIZE[1]//2 - TARGET_SIZE[1]//2:ORIGINAL_SIZE[1]//2 + TARGET_SIZE[1]//2]
        img2_crop = img2_orig[ORIGINAL_SIZE[0]//2 - TARGET_SIZE[0]//2:ORIGINAL_SIZE[0]//2 + TARGET_SIZE[0]//2,
                              ORIGINAL_SIZE[1]//2 - TARGET_SIZE[1]//2:ORIGINAL_SIZE[1]//2 + TARGET_SIZE[1]//2]

        # Match
        kp0, kp1, scores = match_pair(net, device, img1_crop, img2_crop, backend)
        if len(kp0) == 0:
            print("  No matches found")
            continue

        shifts = kp0 - kp1
        labels, _ = cluster_shifts(shifts)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        print(f"  Matches: {len(kp0)}, Clusters: {n_clusters}")

        # Save visualizations
        base = ts_str.replace(":", "-").replace("T", "_")
        save_raw_matches(img1_crop, img2_crop, kp0, kp1, scores,
                         os.path.join(OUTPUT_DIR, f"{base}_{backend}_raw_matches.png"))
        save_clustered_matches(img1_crop, img2_crop, kp0, kp1, labels,
                               os.path.join(OUTPUT_DIR, f"{base}_{backend}_clustered_matches.png"))
        save_shift_clusters(shifts, labels,
                            os.path.join(OUTPUT_DIR, f"{base}_{backend}_shift_clusters.png"))

    print(f"\nDone. Visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    # Set to "eloftr" or "roma"
    BACKEND = "roma"
    process_and_visualize(BACKEND)
