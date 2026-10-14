# PURPOSE: Probe whether the installed kornia LoFTR accepts a batch of image pairs in
#          one forward pass, what shape it returns, and how much faster it is. The
#          evaluation loop currently runs one 640x640 pair per forward pass while the
#          GPU sits at ~6 % utilisation and 3 GB of 24 GB, so batching is the obvious
#          speed-up for development iteration. This probe answers the question
#          empirically rather than assuming a batch contract.
# INPUTS: six real frames from the 2025-05-23 session.
# OUTPUTS: shapes, per-item match counts, throughput, and a check that the batched
#          result for an item equals the single-item result.
# KEYWORDS: loftr, batching, throughput, api_probe, kornia
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")
import torch                                                       # noqa: E402
from clouds.matching.matchers import create_matcher, supports_batching  # noqa: E402

D = "/home/iustimov/clouds_data/images"


def main():
    from clouds.config import load_config
    cfg = load_config("configs/paper_dlcp2026.yaml")
    M = np.array(cfg.calibration_record.matrix_2x3)
    match = create_matcher("kornia_loftr", {"match_threshold": 0.6}, "cuda")
    print(f"backend supports batch matching: {supports_batching(match)}")
    names = sorted(n for n in os.listdir(D)
                   if n.endswith("devID1.jpg") and n.startswith("img-2025-05-23")
                   and "12-0" in n[15:20])[:16]
    pairs = []
    for n in names:
        b = n.replace("devID1.jpg", "")
        i1 = cv2.imread(os.path.join(D, n))
        i2 = cv2.imread(os.path.join(D, b + "devID2.jpg"))
        w = cv2.warpAffine(i1, M, dsize=(1920, 1920))
        # exactly what StereoAnalyzer does: rectify camera 1, then centre-crop both
        a = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)[640:1280, 640:1280]
        c = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)[640:1280, 640:1280]
        pairs.append((a, c))
    print(f"pairs: {len(pairs)}")

    print("\n--- one pair per forward pass (current pipeline behaviour) ---")
    t0 = time.time()
    singles = [match(a, b) for a, b in pairs]
    dt1 = time.time() - t0
    print(f"  {len(pairs)} pairs in {dt1:.2f} s  ->  {len(pairs)/dt1:.2f} pairs/s")

    for B in (4, 8, 16):
        print(f"\n--- batched, batch_size={B} ---")
        t0 = time.time()
        batch = match.match_batch(pairs, batch_size=B)
        dt = time.time() - t0
        print(f"  {len(pairs)} pairs in {dt:.2f} s  ->  {len(pairs)/dt:.2f} pairs/s  "
              f"speed-up {dt1/dt:.2f}x")
        counts_b = [len(k) for k, _ in batch]
        counts_s = [len(k) for k, _ in singles]
        print(f"  match counts batched: {counts_b}")
        print(f"  match counts single : {counts_s}")
        worst = 0.0
        for (kb, _), (ks, _) in zip(batch, singles):
            m = min(len(kb), len(ks))
            if m:
                worst = max(worst, float(np.abs(kb[:m] - ks[:m]).max()))
        print(f"  worst per-match disagreement vs single pass: {worst:.4f} px")


if __name__ == "__main__":
    main()
