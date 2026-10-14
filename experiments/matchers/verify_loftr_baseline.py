# PURPOSE: Prove that the matching path used by this evaluation reproduces the
#          frozen W1 LoFTR correspondence cache bit-for-bit. The kornia LoFTR
#          backend and the pipeline's call path must not have drifted; a matcher
#          comparison against a moved baseline would be meaningless.
# INPUTS: experiments/w1/matches_cache_2025-05-23.npz (built by experiments/w1/epipolar_check.py
#         with configs/paper_dlcp2026.yaml, batch_size 1) and the real images.
# OUTPUTS: stdout: per-frame whether kp1 and kp2 are exactly equal; exit code 1 on
#          any mismatch.
# KEYWORDS: regression, baseline, byte_equivalence, loftr
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import harness  # noqa: E402
from clouds.matching.filters import get_corresponding_feature_points  # noqa: E402

CACHE = os.path.join(REPO, "experiments", "w1", "matches_cache_2025-05-23.npz")


def main():
    data = np.load(CACHE, allow_pickle=True)
    kp1s, kp2s, stamps, frames = data["kp1"], data["kp2"], data["stamps"], data["frames"]
    cfg = harness.load_cfg(None)
    off = (cfg.camera.original_size[0] - cfg.camera.target_size[0]) / 2.0
    bad = 0
    for k1_ref, k2_ref, st, fn in zip(kp1s, kp2s, stamps, frames):
        p1 = os.path.join(cfg.dataset.images_dir, fn)
        p2 = os.path.join(cfg.dataset.images_dir, str(fn).replace("devID1.jpg", "devID2.jpg"))
        g1, g2, _, _ = harness.prepare_crop(cfg, p1, p2)
        k1, k2 = get_corresponding_feature_points(
            g1, g2, visualize=False, match_thresh=cfg.matching.match_threshold,
            backend="kornia_loftr", filter_config=cfg.matching.filter)
        k1 = k1 + off
        k2 = k2 + off
        same = (len(k1) == len(k1_ref) and np.array_equal(k1, k1_ref)
                and np.array_equal(k2, k2_ref))
        if not same:
            bad += 1
        print(f"{st}  n={len(k1):5d}/{len(k1_ref):5d}  exact={same}")
    print(f"\n{len(kp1s) - bad}/{len(kp1s)} frames byte-identical to the frozen W1 cache")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
