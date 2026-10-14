# PURPOSE: Attribute the evaluation loop's wall time to its stages, so optimisation
#          targets the real bottleneck instead of the assumed one. The obvious
#          suspicion (the matcher) may be wrong: each frame also decodes two
#          1920x1920 JPEGs, warps one, and runs HDBSCAN on thousands of matches.
# INPUTS: real frames from 2025-05-23, the project config.
# OUTPUTS: per-stage milliseconds and a projected frames-per-second for each
#          optimisation scenario.
# KEYWORDS: profiling, throughput, batching, bottleneck
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")
from clouds.config import load_config                                       # noqa: E402
from clouds.geometry.imaging import center_crop                             # noqa: E402
from clouds.matching.matchers import create_matcher, supports_batching      # noqa: E402
from clouds.clustering.disparity import cluster_shifts                      # noqa: E402

D = "/home/iustimov/clouds_data/images"


def timed(fn, n):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) / n * 1000.0, out


def main():
    cfg = load_config("configs/paper_dlcp2026.yaml")
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    full = cfg.camera.original_size
    names = sorted(n for n in os.listdir(D)
                   if n.endswith("devID1.jpg") and n.startswith("img-2025-05-23")
                   and "12-0" in n[15:20])[:24]
    print(f"frames: {len(names)}\n")

    # 1) JPEG decode, as the pipeline does it (cv2.imread of the full frame)
    def load():
        for n in names:
            b = n.replace("devID1.jpg", "")
            cv2.imread(os.path.join(D, n))
            cv2.imread(os.path.join(D, b + "devID2.jpg"))
    ms_load, _ = timed(load, len(names) * 2)
    print(f"  JPEG decode (1920x1920)      {ms_load:7.1f} ms per image")

    imgs = []
    for n in names:
        b = n.replace("devID1.jpg", "")
        imgs.append((cv2.imread(os.path.join(D, n)), cv2.imread(os.path.join(D, b + "devID2.jpg"))))

    # 2) rectification + crop
    def prep():
        for i1, i2 in imgs:
            w = cv2.warpAffine(i1, M, dsize=full)
            center_crop(w, crop, crop)
            center_crop(i2, crop, crop)
    ms_prep, _ = timed(prep, len(imgs))
    print(f"  warp + 2 crops               {ms_prep:7.1f} ms per frame")

    gray = []
    for i1, i2 in imgs:
        w = cv2.warpAffine(i1, M, dsize=full)
        gray.append((cv2.cvtColor(center_crop(w, crop, crop), cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(center_crop(i2, crop, crop), cv2.COLOR_BGR2GRAY)))

    match = create_matcher("kornia_loftr", {"match_threshold": cfg.matching.match_threshold}, "cuda")

    # 3) matcher, one pair at a time
    ms_m1, singles = timed(lambda: [match(a, b) for a, b in gray], len(gray))
    print(f"  LoFTR, one pair per pass     {ms_m1:7.1f} ms per frame")

    # 4) matcher, batched
    ms_mb, batch = timed(lambda: match.match_batch(gray, batch_size=6), len(gray))
    print(f"  LoFTR, batch of 6            {ms_mb:7.1f} ms per frame"
          f"   ({ms_m1/ms_mb:.2f}x faster)")

    # 5) clustering
    hcfg = {"min_cluster_size": cfg.clustering.min_cluster_size,
            "min_samples": cfg.clustering.min_samples,
            "cluster_selection_epsilon": cfg.clustering.cluster_selection_epsilon,
            "allow_single_cluster": cfg.clustering.allow_single_cluster}
    shifts = [k1 - k2 for k1, k2 in singles]
    ms_cl, _ = timed(lambda: [cluster_shifts(s, hcfg, False) for s in shifts], len(shifts))
    print(f"  HDBSCAN clustering           {ms_cl:7.1f} ms per frame")

    total_now = ms_load * 2 + ms_prep + ms_m1 + ms_cl
    total_batch = ms_load * 2 + ms_prep + ms_mb + ms_cl
    print(f"\n  current pipeline             {total_now:7.1f} ms per frame  "
          f"-> {1000/total_now:.2f} fps")
    print(f"  with batched matcher         {total_batch:7.1f} ms per frame  "
          f"-> {1000/total_batch:.2f} fps")

    # 6) what if decoding were parallelised (4 threads) and the matcher batched
    def load_parallel():
        from concurrent.futures import ThreadPoolExecutor
        def one(n):
            b = n.replace("devID1.jpg", "")
            cv2.imread(os.path.join(D, n))
            cv2.imread(os.path.join(D, b + "devID2.jpg"))
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(one, names))
    ms_loadp, _ = timed(load_parallel, len(names))
    print(f"\n  (JPEG decode across 4 threads {ms_loadp*2:7.1f} ms per frame pair)")
    best = ms_loadp * 2 + ms_prep + ms_mb + ms_cl
    print(f"  with batched matcher + parallel decode   {best:7.1f} ms per frame  "
          f"-> {1000/best:.2f} fps")
    print(f"\n  stage shares of the current cost: decode "
          f"{100*ms_load*2/total_now:.0f}%, prep {100*ms_prep/total_now:.0f}%, "
          f"match {100*ms_m1/total_now:.0f}%, cluster {100*ms_cl/total_now:.0f}%")


if __name__ == "__main__":
    main()
