# Performance — where inference time actually goes

**Date:** 2026-10-13 · Measured with `experiments/w1/profile_pipeline.py` and
`experiments/w1/batch_api_probe.py` on 2025-05-23 frames.

This note exists because the obvious optimisation was the wrong one, and the
measurement that showed that is worth keeping.

## The stage breakdown

One stereo pair, 640×640 crop, RTX 3090, measured over 24 frames:

| stage | ms / frame | share |
|---|---|---|
| JPEG decode (two 1920×1920 images) | 51 | 12 % |
| rectification warp + two crops | 3.5 | 1 % |
| **LoFTR matching** | **168** | **41 %** |
| **HDBSCAN clustering** | **185** | **45 %** |
| profiled total | 408 | → 2.45 fps |

The clustering costs **as much as the matcher**. That was not expected, and it
means any plan that optimises only the matcher is aiming at less than half the
problem.

Note also that the real evaluation runs at **~1.5–1.65 fps**, i.e. roughly
600 ms/frame, so about 200 ms/frame lies outside the profiled stages. The
suspects are the per-frame JSONL provenance event, `torch.cuda.empty_cache()`
called inside every match, and per-frame Python object churn. That gap is the
subject of the optimisation pass.

## Batching the matcher: verified, but not the answer

LoFTR accepts a batch of pairs in one forward pass (the batch output is a flat
match list plus a `batch_indexes` array — there is no padding, so no compute is
wasted on a rich frame). It is now implemented as `matcher.match_batch` and is
**numerically equivalent**: worst per-match disagreement 0.0000 px versus a single
pass, and match counts differing by a few out of ~3700 (0.08 %) because batching
changes floating-point reduction order.

Throughput, measured on 16 real frames through the full `analyze_batch` path
(decode + rectify + match + cluster):

| path | ms / frame | speed-up |
|---|---|---|
| one pair per forward pass | 562 | 1.00× |
| batch of 4 | 331 | **1.70×** |
| batch of 8 | 414 | 1.36× |
| batch of 16 | — | OOM-limited (2.6 GB allocation refused) |

So the matcher stage does speed up 1.70×. But the **end-to-end** evaluation went
from 1.65 fps to ~1.5–1.6 fps — **no gain, within noise**. The reason is
arithmetic: the matcher is 41 % of the time, so a 1.70× matcher speed-up can
improve the total by at most `1 / (0.59 + 0.41/1.7) = 1.20×`, and only if the
freed GPU time is actually used. It is not, because the CPU side (decode and
clustering) blocks between batches and nothing overlaps.

## What this implies for the optimisation order

1. **Clustering first.** 45 % of the time, and it is the stage whose output the
   plan already intends to replace (W4), because the current settings fragment
   the field into hundreds of clusters that are substantially calibration error.
   Speeding it up and replacing it are the same task.
2. **Overlap, not just batching.** With the GPU at ~3 % utilisation, the win is
   pipelining — prepare and decode batch *N+1* while the GPU runs batch *N* —
   rather than a larger batch.
3. **Decode in parallel.** `cv2.imread` releases the GIL; four threads already
   beat serial decoding.
4. **The residual ~200 ms/frame is pure overhead** and should be measurable and
   removable.

## Rules that constrain the optimisation

- `matching.batch_size` defaults to **1**, which reproduces the historical
  numbers exactly. Batching is a development convenience, and any reported figure
  must state which was used.
- An optimisation must not consult the ceilometer, must not change the fixed
  calibration, and must measure its numerical effect rather than assuming
  equivalence.
- A faster clustering must preserve the output contract: per-cluster mean
  x-disparity and its standard deviation, plus per-match labels.

---

# Optimisation pass — measured results (same date, 2026-10-13)

The hypotheses above were tested. Two of them were wrong, and the largest single
cost was one the first profiler never measured at all.

## The 200 ms/frame was the spatial filter, not provenance or `empty_cache`

`Profile`-based attribution of a real 40-frame run (`cprof_opt40`, see
`experiments/w1/opt/`) put `StereoAnalyzer.prepare_pair`, the matcher and
`cluster_shifts` on the critical path, and found a function no earlier
measurement had timed: `validate_clusters_spatial`. The tracker ran it over
**every** HDBSCAN label — 170–480 labels/frame at `min_cluster_size=3` — and
each label costs a DBSCAN fit. Measured **217.7 ms/frame** on cached real
matches (`experiments/w1/opt/stage_timing.py`). That is the missing time.

The fix is exactly result-preserving: the tracker has already dropped clusters
below `min_cluster_size_for_tracking` (and merged near-equal heights), and only
the surviving `cluster_id`s are ever looked up, so validating the discarded
labels is pure waste. `validate_clusters_spatial` gained an optional
`cluster_ids` argument and the two tracker call sites pass the survivors.
Restricted cost: **8.6 ms/frame** (spatial ON-minus-OFF in
`experiments/w1/opt/frame_stages.py`).

Null suspects, measured and rejected:

- **Per-frame JSONL provenance write.** `run.event` with a flush is
  **0.0 ms/frame** (`stage_timing.py`). Not a bottleneck.
- **`torch.cuda.empty_cache()` per match.** cProfile of the real matcher put it
  at **19 ms/frame** (`{built-in method torch._C._cuda_emptyCache}`, 0.753 s over
  40 frames). It is now removed from both matcher backends. An isolated bench
  shows it near-zero when the cache is already empty, so the cost is the real
  allocator flush after LoFTR, not the call itself.

## What was tested and rejected

- **1-D clustering on `disp_x` only.** The disparity distribution is dominated
  by one axis, but HDBSCAN on `(dx)` re-labels **95.66 %** of matches versus
  `(dx, dy)`. Not equivalent; rejected.
- **A different HDBSCAN `algorithm`.** sklearn 1.9's rewrite accepts
  `auto|kd_tree|ball_tree|brute`. `kd_tree` is what `auto` picks and gives
  byte-identical labels; `ball_tree` is 1.15× slower; `brute` is 2.2× slower and
  disagrees on 2 948/24 507 labels. No faster equivalent exists in this install.
- **Dropping `store_centers="centroid"` and computing the same weighted centroid
  inside the existing per-cluster std loop.** Bit-exact (0/10 855 centroids
  differ) but **2 ms/frame slower** (65.1 → 67.1 ms best-of-3). Reverted.
- **Vectorising per-cluster mean/std with `np.bincount`.** The two-pass form is
  the closest, but still differs from `np.std` by up to 2.2e-6 px because the
  reduction order changes. Rejected to keep `batch_size: 1` byte-reproducible.
- **`batch_size > 1` end-to-end.** With the spatial filter fixed, a batch of 4
  is only ~5 % faster (3.90 → 4.11 fps at 60 frames) and it *does* change the
  numerics (`spatial_reject` 103/256 vs 106/255), so it stays development-only.
- **Prefetch depth > 1.** Depth 2/3 gave 3.98/3.99 fps versus depth 1's 3.90 at
  60 frames — inside noise. Depth 1 is the default.

## CPU/GPU overlap

The serial loop cannot use the idle GPU. `evaluate` now runs a one-thread
prefetch pipeline (`_prefetch_results`): the worker decodes, rectifies and — with
`matching.overlap_matching: true`, the default — also runs LoFTR, while the
consumer thread clusters, filters and updates the tracker. `cv2` and the torch
kernels release the GIL, so the overlap is real. Measured 150 frames, same
output: prepare-only prefetch 3.71 fps, prepare+match overlap **4.10 fps**.

## Verified equivalence

`experiments/w1/opt/ab_evaluate.py --mode base` restores the pre-fix spatial
behaviour in-process; `--mode opt` uses the shipped code. On 150 frames the two
produce **byte-identical** `series.json`, `clusters.jsonl` and `metrics.json`
(commands in that file). The changes are exactly result-preserving.

## What is still a floor

- **LoFTR**: 110–160 ms/frame, dominated by kornia's coarse matching, whose first
  `torch.where` syncs the device. Combined with `conv2d` this is the matcher's own
  cost; GPU utilisation stays ~5 % because the stage is launch/sync-bound, not
  compute-bound. Batching it is the only lever and it is already implemented.
- **HDBSCAN fit**: 30 ms/frame at ~850 matches, 92 ms at ~2 400 (sklearn 1.9
  `_hdbscan_prims`). No faster algorithm is available in this install.

---

## Follow-up (2026-10-13): the spatial filter has been removed

The optimisation pass reduced the spatial filter's cost to 8.6 ms by validating only
surviving labels. It has since been **removed from the pipeline entirely**, on the
evidence that it rejected 37–43 % of clusters while leaving the raw height series
bit-identical and moving the filtered series by 0.3 %. See
[pipeline-steps.md](pipeline-steps.md) for the measurement, the verification, and
the other steps audited in the same pass.
