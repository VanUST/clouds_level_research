# PURPOSE: Run the honest cloud-base-height evaluation end to end — stereo pairs
#          through LoFTR and HDBSCAN, an explicit cluster-selection rule, the
#          optional spatial filter and the Kalman temporal filter — then score
#          the resulting series against the DOL-2 ceilometer. This is the entry
#          point that finally drives the temporal-filter modules, which shipped
#          complete but reachable from nothing.
#
#          Deliberately absent: any selection rule that consults the ceilometer.
#          The removed parse_algorithm_logs chose the cluster closest to the
#          ground-truth reading, inflating every correlation computed through it
#          (mean windowed r 0.24 -> 0.76); see docs/adr/0005 and docs/repro/dlcp2026.md.
# INPUTS: cfg (clouds.config.Config), an opened clouds.provenance.Run, and run
#         options (limit, stride, dump path, shift scan, spatial override).
# OUTPUTS: dict with the raw and filtered series, the ceilometer series, and all
#          metrics; series and metrics are also written into the run directory.
# KEYWORDS: evaluation, cloud_base_height, kalman, ceilometer, metrics, honest_path
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from ..geometry.camera import CameraModel, StereoCameraSystem
from ..metrics import alignment as align
from ..metrics import series as series_metrics
from ..sources import ceilometer as ceilo_src
from ..temporal import ClusterMeasurement, TemporalFilter, merge_clusters_by_height
from ..temporal.settings import build_temporal_settings
from ..sources.images import find_pairs_in_window, image_clock_window
from .analyze import StereoAnalyzer

# Cluster-selection rules a frame's cloud base may be chosen by. All are
# ceilometer-free by construction.
STRATEGIES = ("largest", "lowest", "lowest_reliable", "lowest_mode")


# PURPOSE: Feed the evaluation loop one frame's work at a time while a worker
#          thread runs later frames ahead of it. The un-overlapped loop is
#          strictly serial — decode, match, cluster, filter — so while the
#          consumer burns CPU in HDBSCAN and the tracker, both the JPEG decode of
#          the next pair and (optionally) its LoFTR forward pass would otherwise
#          sit idle. cv2.imread/warpAffine/cvtColor and the torch kernels release
#          the GIL, so one worker genuinely overlaps. Threading changes only when
#          a stage runs, never what it computes: the yielded arrays are identical
#          to prepare_pair()/match_prepared()'s.
# INPUTS: analyzer (StereoAnalyzer), pairs (list[(path1, path2)]),
#         prefetch (int >= 1) — pairs held in flight beyond the one being yielded,
#         include_match (bool) — also run the matcher on the worker, so the
#         consumer only clusters; when False the worker stops after preparation.
# OUTPUTS: generator, in input order, of either prepare_pair()'s tuple
#          (include_match=False) or (kp1, kp2, img1, img2) (True); None where a
#          pair is unreadable.
# KEYWORDS: prefetch, overlap, threading, decode, loftr, throughput
def _prefetch_results(analyzer, pairs, prefetch: int = 1, include_match: bool = False):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    n = len(pairs)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clouds-prep")
    pending = deque()

    def _work(i):
        prep = analyzer.prepare_pair(pairs[i][0], pairs[i][1], analyzer.affine)
        if prep is None or not include_match:
            return prep
        kp1, kp2 = analyzer.match_prepared(prep[2], prep[3])
        return (kp1, kp2, prep[0], prep[1])

    def _submit(i):
        return pool.submit(_work, i)

    try:
        nxt = 0
        while nxt < min(max(prefetch, 1), n):
            pending.append(_submit(nxt))
            nxt += 1
        for _ in range(n):
            prep = pending.popleft().result()
            if nxt < n:
                pending.append(_submit(nxt))
                nxt += 1
            yield prep
    finally:
        pool.shutdown(wait=False)

# PURPOSE: Convert the pipeline's cluster summary dicts into tracker measurements.
# INPUTS: clusters (list[dict] from StereoAnalyzer.analyze_pair).
# OUTPUTS: list[ClusterMeasurement]; empty when the frame produced no clusters.
# KEYWORDS: adapter, cluster, measurement
def to_measurements(clusters) -> list:
    out = []
    for c in clusters:
        out.append(ClusterMeasurement(
            h=float(c["distance"]),
            h_err=float(c["error"]),
            u=float(c["mean_u"]),
            v=float(c["mean_v"]),
            size=int(c["size"]),
            disp_x=float(c["disp_x"]),
            disp_std=float(c.get("disp_std", 0.0)),
            cluster_id=int(c["cluster_id"]),
        ))
    return out


# PURPOSE: Select one cluster per frame by an explicit, ground-truth-free rule.
# INPUTS: clusters (list[ClusterMeasurement]), strategy (str; one of STRATEGIES),
#         min_size (int) — clusters smaller than this are "unreliable".
# OUTPUTS: float height in metres, or None when the rule has no candidate.
#          'lowest' returns the lowest unreliable-or-not cluster, but prefers
#          clusters at or above min_size and falls back to the full set only when
#          none qualify. That fallback is what the previous implementation did,
#          and it is preserved here so the refactor does not silently change the
#          honest path's numbers.
#          'lowest_reliable' is the strict form: the lowest cluster at or above
#          min_size, or None.
#          'largest' returns the most populous cluster.
# KEYWORDS: selection, strategy, cloud_base, cbh
def select_height(clusters, strategy: str, min_size: int):
    if not clusters:
        return None
    if strategy == "largest":
        pool = [c for c in clusters if c.size >= min_size] or clusters
        return float(max(pool, key=lambda c: c.size).h)
    if strategy == "lowest_reliable":
        pool = [c for c in clusters if c.size >= min_size]
        return float(min(c.h for c in pool)) if pool else None
    if strategy == "lowest_mode":
        # The mode path emits exactly one cluster carrying the mode height, so the
        # selection is a pass-through and the estimator's own prominence rule is the
        # only thing that decides. See analyze.lowest_prominent_mode.
        pool = [c for c in clusters if c.size >= min_size] or clusters
        return float(min(c.h for c in pool))
    if strategy == "lowest":
        pool = [c for c in clusters if c.size >= min_size] or clusters
        return float(min(c.h for c in pool))
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


# PURPOSE: Run the full evaluation for one configured session.
# INPUTS: cfg (Config), run (provenance.Run), limit (int, 0 = all), stride (int),
#         dump_clusters (bool), shift_scan (bool), spatial_override (bool|None),
#         strategy (str|None — overrides the configured cbh_strategy).
# OUTPUTS: dict {times, raw, filtered, ceilo, metrics, stats}.
# KEYWORDS: evaluate, orchestration, kalman, metrics
def evaluate(cfg, run, limit: int = 0, stride: int = 1, dump_clusters: bool = False,
             shift_scan: bool = False, strategy=None) -> dict:
    settings, deviations = build_temporal_settings(cfg)
    chosen_strategy = strategy or cfg.temporal.cbh_strategy
    run.event("temporal_settings", settings=settings,
              deviations_from_library_defaults=deviations, strategy=chosen_strategy)

    stereo = StereoCameraSystem(
        base=cfg.camera.stereo_base_m,
        angle_of_view=cfg.camera.angle_of_view_deg,
        image_width=cfg.camera.image_width_px,
        model=CameraModel(cfg.camera.model),
    )
    analyzer = StereoAnalyzer(cfg, stereo)

    offset_h = float(cfg.dataset.clock_offset_h)
    win_fn = image_clock_window(cfg)
    pairs = find_pairs_in_window(cfg.dataset.images_dir, win_fn[0], win_fn[1],
                                 cfg.dataset.date, cfg.dataset.pair_tolerance_s)
    if stride > 1:
        pairs = pairs[::stride]
    if limit:
        pairs = pairs[:limit]
    run.record("counts", n_pairs=len(pairs))
    run.event("window", image_clock_window=win_fn,
              ceilometer_clock_window=(cfg.dataset.window_start, cfg.dataset.window_end),
              clock_offset_h=offset_h, n_pairs=len(pairs))

    tracker = TemporalFilter(settings, stereo_system=stereo)

    times, raw, filtered, dump = [], [], [], []
    n_no_result = 0
    t_start = time.time()
    batch_size = max(1, int(getattr(cfg.matching, "batch_size", 1) or 1))
    prefetch = int(getattr(cfg.matching, "prefetch", 1) or 0)
    overlap_match = bool(getattr(cfg.matching, "overlap_matching", False))
    run.event("matching_batch_size", batch_size=batch_size, prefetch=prefetch,
              overlap_matching=overlap_match)
    prep_iter = (_prefetch_results(analyzer, pairs, prefetch,
                                   include_match=overlap_match)
                 if (batch_size == 1 and prefetch > 0) else None)
    i = 0
    while i < len(pairs):
        chunk = pairs[i:i + batch_size]
        if batch_size > 1:
            results = analyzer.analyze_batch(chunk, batch_size=batch_size)
        elif prep_iter is None:
            results = [analyzer.analyze_pair(a, b, affine_matrix=analyzer.affine)
                       for a, b in chunk]
        elif overlap_match:
            # The worker already decoded and matched; only clustering and
            # filtering run on this thread.
            results = []
            for _ in chunk:
                matched = next(prep_iter)
                if matched is None:
                    results.append(None)
                else:
                    kp1, kp2, img1, img2 = matched
                    results.append(analyzer.package_matches(kp1, kp2, img1, img2))
        else:
            # Preparation already happened in the prefetch thread; only the
            # matcher, clustering and filtering run here.
            results = []
            for _ in chunk:
                prep = next(prep_iter)
                if prep is None:
                    results.append(None)
                else:
                    results.append(analyzer.analyze_prepared(
                        prep[2], prep[3], prep[0], prep[1]))
        for j, res in enumerate(results):
          p1, p2 = chunk[j]
          base = Path(p1).name.replace("devID1.jpg", "")
          ts_fn = dt.datetime.strptime(base.replace("img-", ""), "%Y-%m-%dT%H-%M-%S")
          ts = align.to_ceilometer_clock(ts_fn, offset_h)

          if res is None:
              n_no_result += 1
              run.event("frame", i=i + j, ts=ts.isoformat(), status="no_result")
              continue
          meas = to_measurements(res["clusters"])
          if not meas:
              run.event("frame", i=i, ts=ts.isoformat(), status="no_clusters")
              continue

          raw_h = select_height(meas, chosen_strategy,
                                cfg.temporal.min_cluster_size_for_tracking)
          merge_tol = cfg.temporal.cluster_merge_height_tol_m
          merged = (merge_clusters_by_height(meas, merge_tol) if merge_tol > 0 else meas)
          if getattr(cfg.temporal, "filter_enabled", True):
              filt_h = tracker.process_frame(merged, ts.timestamp(),
                                             kp1=res["kp1"], labels=res["labels"],
                                             shifts=res["shifts"])
          else:
              # Filter bypassed: the delivered series is the per-frame selection, so
              # the retrieval's own variability is visible with no smoother on top.
              filt_h = select_height(merged, chosen_strategy,
                                     cfg.temporal.min_cluster_size_for_tracking)
          times.append(ts)
          raw.append(np.nan if raw_h is None else float(raw_h))
          filtered.append(float(filt_h) if filt_h is not None else float("nan"))
          if dump_clusters:
              dump.append({"ts_ceilometer_clock": ts.isoformat(),
                           "clusters": [{"h": m.h, "size": m.size, "disp_x": m.disp_x,
                                         "disp_std": m.disp_std, "u": m.u, "v": m.v,
                                         "cluster_id": m.cluster_id} for m in meas]})

          # Progress every 100 frames: a long run must be observable while it is
          # running, not only when it finishes.
          done = i + j + 1
          if done % 100 == 0:
              elapsed = time.time() - t_start
              rate = done / max(elapsed, 1e-9)
              eta = (len(pairs) - done) / max(rate, 1e-9)
              run.event("progress", done=done, total=len(pairs),
                        frames_with_clusters=len(times), no_result=n_no_result,
                        elapsed_s=round(elapsed, 1), fps=round(rate, 2),
                        eta_s=round(eta, 1))
              print(f"  [{done}/{len(pairs)}] clusters={len(times)} "
                    f"no_result={n_no_result} {elapsed:.0f}s "
                    f"({rate:.2f} fps, eta {eta/60:.1f} min)", flush=True)

        i += len(chunk)

    ceilo_samples = ceilo_src.parse_ceilometer_log(cfg.dataset.ceilometer_log)
    lo = dt.datetime.strptime(cfg.dataset.window_start, "%H:%M:%S").time()
    hi = dt.datetime.strptime(cfg.dataset.window_end, "%H:%M:%S").time()
    ceilo = [(s.time, s.lowest) for s in ceilo_samples
             if s.time.date() == dt.date.fromisoformat(cfg.dataset.date)
             and lo <= s.time.time() <= hi]
    run.event("ceilometer", total_samples=len(ceilo_samples), in_window=len(ceilo))

    metrics = _score(cfg, times, raw, filtered, ceilo, shift_scan)
    stats = {
        "n_frames_processed": len(pairs),
        "n_frames_with_clusters": len(times),
        "n_no_result": n_no_result,
        "clusters_seen": tracker.total_clusters_seen,
    }
    run.record("counts", **stats)
    run.record("results", **{k: v for k, v in metrics.items() if not k.endswith("_detail")})
    return {"times": times, "raw": raw, "filtered": filtered, "ceilo": ceilo,
            "metrics": metrics, "stats": stats, "cluster_dump": dump}


# PURPOSE: Compute every comparison metric for the raw and filtered series.
# INPUTS: cfg, times, raw, filtered, ceilo (list of (datetime, height)), shift_scan.
# OUTPUTS: dict of metrics, including the per-window detail excluded from the
#          manifest by the caller.
# KEYWORDS: metrics, correlation, autocorrelation, windows
def _score(cfg, times, raw, filtered, ceilo, shift_scan: bool) -> dict:
    m = cfg.metrics
    raw_pts = [(t, h) for t, h in zip(times, raw) if np.isfinite(h)]
    filt_pts = [(t, h) for t, h in zip(times, filtered) if np.isfinite(h)]
    c_t = [t for t, _ in ceilo]
    c_h = [h for _, h in ceilo]

    out = {
        "n_raw_points": len(raw_pts),
        "n_filtered_points": len(filt_pts),
        "n_ceilo_points": len(ceilo),
        "smoothing": series_metrics.smoothing_improvement(
            [h for _, h in raw_pts], [h for _, h in filt_pts]),
    }
    for name, pts in (("raw", raw_pts), ("filtered", filt_pts)):
        wins = align.windowed_correlation([t for t, _ in pts], [h for _, h in pts],
                                          c_t, c_h, m.window_min, m.shift_search_s,
                                          m.tolerance_s, step_min=m.hop_min)
        out[f"{name}_windows"] = align.summarise_windows(wins)
        out[f"{name}_windows_detail"] = wins
        r, n = align.paired_r([t.timestamp() for t, _ in pts], [h for _, h in pts],
                              [t.timestamp() for t in c_t], c_h, 0.0, m.tolerance_s)
        out[f"{name}_global_r"] = None if np.isnan(r) else float(r)
        out[f"{name}_global_n"] = n
    if shift_scan and raw_pts:
        out["shift_scan"] = align.scan_global_shift(
            [t for t, _ in raw_pts], [h for _, h in raw_pts], c_t, c_h, m.tolerance_s)
    return out


# PURPOSE: Write the series, metrics and cluster dump next to the run manifest.
# INPUTS: run (provenance.Run), result (dict from evaluate), write_clusters (bool).
# OUTPUTS: list of written paths.
# KEYWORDS: artefacts, output, jsonl
def write_artefacts(run, result: dict, write_clusters: bool = False) -> list:
    out_dir = run.run_dir
    written = []
    series_path = out_dir / "series.json"
    series_path.write_text(json.dumps({
        "times_ceilometer_clock": [t.isoformat() for t in result["times"]],
        "raw_h": result["raw"],
        "filtered_h": result["filtered"],
        "ceilometer": [[t.isoformat(), h] for t, h in result["ceilo"]],
    }, indent=1))
    written.append(series_path)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result["metrics"], indent=1, default=str))
    written.append(metrics_path)
    if write_clusters and result.get("cluster_dump"):
        dump_path = out_dir / "clusters.jsonl"
        with dump_path.open("w") as fh:
            for rec in result["cluster_dump"]:
                fh.write(json.dumps(rec) + "\n")
        written.append(dump_path)
    return written
