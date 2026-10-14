# PURPOSE: Attribute the real evaluation's per-frame wall time to the stages the
#          bundled profiler does not measure — above all the spatial filter, which
#          runs a DBSCAN per HDBSCAN cluster and therefore costs time proportional
#          to the number of clusters, not to the number of matches. Also settles
#          whether the bundled profiler's 190 ms "HDBSCAN" figure is the marginal
#          cost or a one-off import folded into 24 frames.
# INPUTS: real 2025-05-23 frames, a config path, a frame count.
# OUTPUTS: per-stage milliseconds/frame and structural counts (matches, clusters,
#          spatial-filter calls) on stdout.
# KEYWORDS: profiling, hdbscan, dbscan, spatial_filter, throughput, cbh
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, "src")
from clouds.config import load_config                                   # noqa: E402
from clouds.geometry.imaging import center_crop                         # noqa: E402
from clouds.clustering.disparity import cluster_shifts                  # noqa: E402
from clouds.clustering.spatial import validate_clusters_spatial         # noqa: E402
from clouds.matching.filters import get_matcher                         # noqa: E402
from clouds.sources.images import find_pairs_in_window, image_clock_window  # noqa: E402

D = "/home/iustimov/clouds_data/images"


# PURPOSE: Time a callable over n repetitions and return ms per call.
# INPUTS: fn (callable), n (int).
# OUTPUTS: (ms_per_call, last_result).
# KEYWORDS: timing, helper
def timed(fn, n):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) / n * 1000.0, out


# PURPOSE: Build the exact (kp1, kp2, shifts) the pipeline feeds to clustering for
#          the first `limit` pairs of a config, via the production prepare/matcher
#          path, so the downstream stages are timed on real distributions.
# INPUTS: cfg, limit (int).
# OUTPUTS: list of (kp1, kp2, shifts) and the config's hdbscan dict.
# KEYWORDS: fixtures, matches, prepare
def collect_matches(cfg, limit):
    win = image_clock_window(cfg)
    pairs = find_pairs_in_window(cfg.dataset.images_dir, win[0], win[1],
                                 cfg.dataset.date, cfg.dataset.pair_tolerance_s)[:limit]
    affine = np.array(cfg.calibration_record.matrix_2x3, dtype=float)
    roll = float(getattr(cfg.camera, "roll_correction_deg", 0.0) or 0.0)
    cxy = (cfg.camera.original_size[0] / 2.0, cfg.camera.original_size[1] / 2.0)
    from clouds.geometry.imaging import rotation_about
    matcher = get_matcher(cfg.matching.backend,
                          {"match_threshold": cfg.matching.match_threshold})
    out = []
    for p1, p2 in pairs:
        i1 = cv2.imread(p1)
        i2 = cv2.imread(p2)
        if roll:
            i2 = cv2.warpAffine(i2, rotation_about(cxy, roll), dsize=cfg.camera.original_size)
        w = cv2.warpAffine(i1, affine, dsize=cfg.camera.original_size)
        g1 = cv2.cvtColor(center_crop(w, 640, 640), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(center_crop(i2, 640, 640), cv2.COLOR_BGR2GRAY)
        kp1, kp2 = matcher(g1, g2)
        out.append((kp1, kp2, kp1 - kp2))
    hcfg = {"min_cluster_size": cfg.clustering.min_cluster_size,
            "min_samples": cfg.clustering.min_samples,
            "cluster_selection_epsilon": cfg.clustering.cluster_selection_epsilon,
            "allow_single_cluster": cfg.clustering.allow_single_cluster}
    return out, hcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/_ab_batch1.yaml")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cache", default="experiments/w1/opt/matches.npz")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.use_cache:
        z = np.load(args.cache, allow_pickle=True)
        data = list(z["data"])
        hcfg = {"min_cluster_size": cfg.clustering.min_cluster_size,
                "min_samples": cfg.clustering.min_samples,
                "cluster_selection_epsilon": cfg.clustering.cluster_selection_epsilon,
                "allow_single_cluster": cfg.clustering.allow_single_cluster}
    else:
        t0 = time.perf_counter()
        data, hcfg = collect_matches(cfg, args.limit)
        print(f"collected {len(data)} frames in {time.perf_counter()-t0:.1f}s")
        np.savez_compressed(args.cache, data=np.array(data, dtype=object))

    n = len(data)
    counts = [len(d[0]) for d in data]
    print(f"frames={n}  matches/frame median={np.median(counts):.0f} "
          f"min={min(counts)} max={max(counts)}")

    # Warm HDBSCAN once so any one-off import is not charged to the per-frame cost.
    cluster_shifts(data[0][2], hcfg, False)
    ms_hdb, labels_all = timed(
        lambda: [cluster_shifts(d[2], hcfg, False)[2] for d in data], n)
    print(f"cluster_shifts (warm)        {ms_hdb:7.1f} ms/frame")

    nlab = [len(set(l.tolist()) - {-1}) for l in labels_all]
    print(f"  clusters/frame median={np.median(nlab):.0f} mean={np.mean(nlab):.1f} "
          f"max={max(nlab)}")

    # Split cluster_shifts into fit vs per-cluster std loop by re-timing the fit alone.
    from sklearn.cluster import HDBSCAN
    def fits():
        for d in data:
            h = HDBSCAN(min_cluster_size=hcfg["min_cluster_size"],
                        min_samples=hcfg["min_samples"],
                        cluster_selection_epsilon=hcfg["cluster_selection_epsilon"],
                        allow_single_cluster=hcfg["allow_single_cluster"],
                        store_centers="centroid")
            h.fit(d[2])
        return None
    ms_fit, _ = timed(fits, n)
    print(f"  HDBSCAN.fit only           {ms_fit:7.1f} ms/frame")

    # Spatial filter: one DBSCAN per label, as the pipeline calls it.
    # cfg keys are spatial_filter.* which settings maps to spatial_*; replicate here.
    scfg = {"spatial_filter_enabled": cfg.spatial_filter.enabled,
            "spatial_dbscan_eps": cfg.spatial_filter.dbscan_eps_px,
            "spatial_dbscan_min_samples": cfg.spatial_filter.dbscan_min_samples,
            "spatial_density_min": cfg.spatial_filter.density_min,
            "spatial_k_neighbors": cfg.spatial_filter.k_neighbors,
            "spatial_inconsistency_max": cfg.spatial_filter.inconsistency_max,
            "spatial_disparity_std_mult": cfg.spatial_filter.disparity_std_mult}
    validate_clusters_spatial(data[0][0], labels_all[0], data[0][2], np.empty((0, 2)), scfg)
    ms_sp, _ = timed(
        lambda: [validate_clusters_spatial(d[0], l, d[2], np.empty((0, 2)), scfg)
                 for d, l in zip(data, labels_all)], n)
    print(f"validate_clusters_spatial    {ms_sp:7.1f} ms/frame  (all labels)")

    # The production call now restricts validation to the clusters that survive
    # the tracker's min-size filter; measure that restricted form too.
    min_size = cfg.temporal.min_cluster_size_for_tracking
    keep_ids = []
    for l in labels_all:
        vals, cnts = np.unique(l[l >= 0], return_counts=True)
        keep_ids.append(set(vals[cnts >= min_size].tolist()))
    print(f"  labels kept by size>={min_size}  median={np.median([len(k) for k in keep_ids]):.0f}")
    ms_sps, _ = timed(
        lambda: [validate_clusters_spatial(d[0], l, d[2], np.empty((0, 2)), scfg,
                                           cluster_ids=k)
                 for d, l, k in zip(data, labels_all, keep_ids)], n)
    print(f"validate_clusters_spatial    {ms_sps:7.1f} ms/frame  (kept labels only)")

    # JSONL event write cost, as Run.event does it (write + flush).
    import tempfile, os
    fh = open(os.path.join(tempfile.gettempdir(), "evt_probe.jsonl"), "w")
    def events():
        from datetime import datetime, timezone
        for i in range(len(data)):
            rec = {"ts_utc": datetime.now(timezone.utc).isoformat(),
                   "event": "frame", "i": i, "status": "ok"}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
    ms_ev, _ = timed(events, n)
    fh.close()
    print(f"run.event write+flush        {ms_ev:7.1f} ms/frame")

    print(f"\nHDBSCAN + all-label spatial {ms_hdb + ms_sp:7.1f} ms/frame")
    print(f"HDBSCAN + kept-label spatial {ms_hdb + ms_sps:7.1f} ms/frame")


if __name__ == "__main__":
    main()
