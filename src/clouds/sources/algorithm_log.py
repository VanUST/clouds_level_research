# PURPOSE: Persist the per-pair result of the stereo analysis — a human-readable
#          summary and an archive of the raw matches — so a frame can be
#          re-inspected later without re-running the matcher.
# INPUTS: a result bundle from pipeline.analyze, the first image path, a log dir.
# OUTPUTS: a .txt summary and a compressed .npz beside it.
# KEYWORDS: logging, artefacts, npz, keypoints, reproducibility
from __future__ import annotations

import os

import numpy as np


def write_pair_logs(result: dict, img1_path: str, log_dir: str):
    """Write a .txt summary and a .npz of the raw keypoints, matches and clusters.

    Returns nothing; a pair with no clusters is skipped silently.
    """
    clusters = result.get('clusters', [])
    if not clusters:
        return

    base_filename = os.path.basename(img1_path)
    txt_name = base_filename.replace('devID1.jpg', '.txt')
    npz_name = base_filename.replace('devID1.jpg', '.npz')
    txt_path = os.path.join(log_dir, txt_name)
    npz_path = os.path.join(log_dir, npz_name)

    # --- Text log (summary, backward compatible) ---
    with open(txt_path, 'w') as f:
        f.write(f"Analysis for: {base_filename}\n")
        f.write(f"Total matches: {len(result['kp1'])}\n")
        f.write("-" * 35 + "\n")
        for i, res in enumerate(clusters):
            f.write(f"Cluster {res['cluster_id']} ({'Dominant' if i == 0 else ''}):\n")
            f.write(f"  Size: {res['size']} points\n")
            f.write(f"  Height (Distance): {res['distance']:.2f} meters\n")
            f.write(f"  Centroid UV: {res['mean_u']:.2f}, {res['mean_v']:.2f}\n")
            f.write(f"  Disparity X: {res['disp_x']:.4f}\n")
        f.write("-" * 35 + "\n")

    # --- NPZ log (raw data for reproducibility) ---
    np.savez_compressed(
        npz_path,
        kp1=result['kp1'],
        kp2=result['kp2'],
        shifts=result['shifts'],
        labels=result['labels'],
        centers=result['centers'],
    )
    print(f"  -> Logs saved: {txt_path}, {npz_path}")


# PURPOSE: Return the per-backend log directory, creating it when absent.
# INPUTS: base_dir (str) — the configured log root; backend (str) — matcher name.
# OUTPUTS: the directory path as a string.
# KEYWORDS: logging, directory, backend
def log_dir_for(base_dir: str, backend: str) -> str:
    """Return the log directory for a matcher backend, creating it if needed."""
    log_dir = os.path.join(base_dir, backend)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

