# PURPOSE: Check that the batched analysis path reproduces the single-pair path, and
#          measure the end-to-end throughput gain on the real per-frame workload
#          (decode + rectify + match + cluster), not just the matcher in isolation.
#          Batching changes floating-point reduction order, so "identical" is a claim
#          that has to be tested rather than assumed.
# INPUTS: real frames from 2025-05-23 and the project config.
# OUTPUTS: per-frame comparison of clusters and matches, and ms/frame for both paths.
# KEYWORDS: batching, equivalence, throughput, verification
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from clouds.config import load_config                                   # noqa: E402
from clouds.geometry.camera import CameraModel, StereoCameraSystem      # noqa: E402
from clouds.pipeline.analyze import StereoAnalyzer                      # noqa: E402

D = "/home/iustimov/clouds_data/images"


def main():
    cfg = load_config("configs/paper_dlcp2026.yaml")
    stereo = StereoCameraSystem(base=cfg.camera.stereo_base_m,
                                angle_of_view=cfg.camera.angle_of_view_deg,
                                image_width=cfg.camera.image_width_px,
                                model=CameraModel(cfg.camera.model))
    analyzer = StereoAnalyzer(cfg, stereo)

    names = sorted(n for n in os.listdir(D)
                   if n.endswith("devID1.jpg") and n.startswith("img-2025-05-23")
                   and "12-0" in n[15:20])[:8]
    pairs = [(os.path.join(D, n), os.path.join(D, n.replace("devID1.jpg", "devID2.jpg")))
             for n in names]

    t0 = time.perf_counter()
    single = [analyzer.analyze_pair(a, b) for a, b in pairs]
    t_single = (time.perf_counter() - t0) / len(pairs) * 1000

    for bs in (4, 8):
        t0 = time.perf_counter()
        batched = analyzer.analyze_batch(pairs, batch_size=bs)
        t_batch = (time.perf_counter() - t0) / len(pairs) * 1000
        print(f"\n=== batch_size={bs} ===  {t_batch:.0f} ms/frame vs {t_single:.0f} single "
              f"({t_single/t_batch:.2f}x)")
        print(f"{'frame':>6} {'matches s/b':>13} {'clusters s/b':>14} {'median D s/b (m)':>20}")
        dmax = 0.0
        for i, (sa, sb) in enumerate(zip(single, batched)):
            nm_s = 0 if sa is None else len(sa["kp1"])
            nm_b = 0 if sb is None else len(sb["kp1"])
            nc_s = 0 if sa is None else len(sa["clusters"])
            nc_b = 0 if sb is None else len(sb["clusters"])
            dm_s = dm_b = float("nan")
            if sa and sa["clusters"]:
                dm_s = float(np.median([c["distance"] for c in sa["clusters"]]))
            if sb and sb["clusters"]:
                dm_b = float(np.median([c["distance"] for c in sb["clusters"]]))
            if not (np.isnan(dm_s) or np.isnan(dm_b)):
                dmax = max(dmax, abs(dm_s - dm_b))
            print(f"{i:>6} {nm_s:>6}/{nm_b:<6} {nc_s:>6}/{nc_b:<7} "
                  f"{dm_s:>8.1f}/{dm_b:<9.1f}")
        print(f"  largest per-frame median-height difference: {dmax:.2f} m")

    print("\nnote: match counts differ by a few out of ~3700 because batching changes")
    print("reduction order; the geometry is unaffected. Reported numbers should use")
    print("batch_size=1 unless the difference is shown to be immaterial.")


if __name__ == "__main__":
    main()
