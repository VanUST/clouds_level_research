# PURPOSE: Split the post-matcher CPU cost of one real frame into its actual
#          parts — HDBSCAN clustering, cluster packaging, the tracker's spatial
#          filter, and the rest of process_frame — on cached real matches. The
#          bundled profiler stops at clustering and so never showed the spatial
#          filter, which ran a DBSCAN per HDBSCAN cluster.
# INPUTS: cached (kp1, kp2, shifts) from stage_timing.py, a config, --limit.
# OUTPUTS: per-stage ms/frame with and without the spatial filter, on stdout.
# KEYWORDS: profiling, clustering, spatial_filter, tracker, throughput
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from clouds.config import load_config                                    # noqa: E402
from clouds.geometry.camera import CameraModel, StereoCameraSystem       # noqa: E402
from clouds.pipeline.analyze import StereoAnalyzer                       # noqa: E402
from clouds.pipeline.evaluate import to_measurements                     # noqa: E402
from clouds.temporal import TemporalFilter, merge_clusters_by_height     # noqa: E402
from clouds.temporal.settings import build_temporal_settings             # noqa: E402


def timed(fn, n):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) / n * 1000.0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/_ab_batch1.yaml")
    ap.add_argument("--cache", default="experiments/w1/opt/matches.npz")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = list(np.load(args.cache, allow_pickle=True)["data"])[:args.limit]
    n = len(data)

    stereo = StereoCameraSystem(base=cfg.camera.stereo_base_m,
                                angle_of_view=cfg.camera.angle_of_view_deg,
                                image_width=cfg.camera.image_width_px,
                                model=CameraModel(cfg.camera.model))
    analyzer = StereoAnalyzer(cfg, stereo)

    ms_pkg, bundles = timed(
        lambda: [analyzer.package_matches(d[0], d[1]) for d in data], n)
    print(f"package_matches (HDBSCAN+packaging) {ms_pkg:7.1f} ms/frame")
    if any(b is None for b in bundles):
        print("WARNING: some frames returned None")
        bundles = [b for b in bundles if b is not None]

    settings, _ = build_temporal_settings(cfg, spatial_override=None)
    tracker = TemporalFilter(settings, stereo_system=stereo)
    merge_tol = cfg.temporal.cluster_merge_height_tol_m

    def track_all(tr):
        for i, b in enumerate(bundles):
            meas = to_measurements(b["clusters"])
            merged = merge_clusters_by_height(meas, merge_tol) if merge_tol > 0 else meas
            tr.process_frame(merged, float(i * 10), kp1=b["kp1"],
                             labels=b["labels"], shifts=b["shifts"])

    ms_trk, _ = timed(lambda: track_all(tracker), len(bundles))
    print(f"process_frame (spatial ON)          {ms_trk:7.1f} ms/frame")

    settings_off, _ = build_temporal_settings(cfg, spatial_override=False)
    tr_off = TemporalFilter(settings_off, stereo_system=stereo)
    ms_trk_off, _ = timed(lambda: track_all(tr_off), len(bundles))
    print(f"process_frame (spatial OFF)         {ms_trk_off:7.1f} ms/frame")
    print(f"  => spatial filter cost            {ms_trk - ms_trk_off:7.1f} ms/frame")

    # merge-only cost
    def merge_all():
        for b in bundles:
            meas = to_measurements(b["clusters"])
            if merge_tol > 0:
                merge_clusters_by_height(meas, merge_tol)
    ms_m, _ = timed(merge_all, len(bundles))
    print(f"  (of which merge_by_height)        {ms_m:7.1f} ms/frame")


if __name__ == "__main__":
    main()
