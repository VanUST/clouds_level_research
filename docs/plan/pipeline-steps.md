# Pipeline steps — what was kept, removed, and left open

**Date:** 2026-10-13. Each entry states the measurement that decided it. The
principle applied: a step stays only if it can be shown to change the product for
the better, and a step whose justification is a magic number is flagged rather than
quietly kept.

---

## Removed: the spatial-coherence filter

**What it did.** Before the tracker consumed a frame's clusters,
`validate_clusters_spatial` tested each one for spatial coherence in image space —
DBSCAN sub-clustering of its keypoints, a density threshold, and a
disparity-versus-position consistency check. The stated purpose was to reject
"false matches on sky".

**Measurement.** Two 400-frame runs on 2025-05-23, identical in every other
respect:

| | clusters seen | rejected | raw series | filtered series |
|---|---|---|---|---|
| filter ON | 1494 | **550 (37 %)** | median 1594 m | median 1864 m, jumps 6.5 m |
| filter OFF | 1494 | 0 | median 1594 m | median 1855 m, jumps 5.7 m |

Frame-aligned over the same 400 frames:

- **raw series: 0.0 m median difference — identical**;
- filtered series: **5.0 m median difference (0.3 %)**, correlation 0.895.

**Verdict: removed.** It rejects 37 % of clusters (43 % over the full 1799-frame
run, 2351 of 5510) and changes the delivered answer by 0.3 %. The raw height series
— the quantity the product is about — is bit-identical with and without it, because
the *lowest cluster* selection runs before the filter and the filter never removed
the selected cluster on this data.

It did have one real cost: before the optimisation pass it was
**217.7 ms/frame**, the largest single stage in the pipeline, because HDBSCAN emits
170–480 clusters per frame at `min_cluster_size=3` and it ran a DBSCAN fit for each
one to serve the ~5 that survive the size filter. That cost was separately reduced
to 8.6 ms by validating only surviving labels; removing the step entirely now saves
that too.

**Verification.** With the module deleted and all call sites removed, a 400-frame
run against the filter-disabled configuration gives a raw difference of **0.013 m**
and a filtered difference of **0.52 m** (correlation 0.99987) — consistent with
`spatial_pass` having been constant in that configuration anyway. Deleted:
`src/clouds/clustering/spatial.py`, its imports, the `spatial_filter` config section,
`SpatialFilterCfg`, and the `spatial_rejections` statistic.

## Removed: `cluster_selection_epsilon`

**What it did.** A clustering knob forwarded to sklearn's HDBSCAN.

**Measurement.** sklearn 1.9's HDBSCAN raises
`TypeError: only 0-dimensional arrays can be converted to Python scalars` for any
`cluster_selection_epsilon > 0` on this data. The knob could therefore never
legitimately be non-zero, and the code around it carried an exception handler whose
fallback returned *no clusters for the frame* — which is indistinguishable from a
clear sky in the output.

**Verdict: removed** from the schema, the loader and every config. The fallback now
prints the exception instead of hiding it.

## Kept, but flagged: the 300 m height merge

**What it does.** `cluster_merge_height_tol_m: 300` merges clusters whose heights lie
within 300 m before they reach the tracker.

**Measurement.** 400 frames, merge 300 vs merge 0:

| | raw median | raw jump | filtered median | filtered jump | filtered p10–p90 |
|---|---|---|---|---|---|
| merge 0 | 1594 m | 28.4 m | **1597 m** | 20.4 m | 1540–1720 m |
| merge 300 | 1594 m | 28.4 m | **1856 m** | 5.8 m | 1794–1923 m |

The raw series is identical (the merge runs after selection), but the **tracked
series moves by 249 m — 16 %** — and becomes 3.5× less jittery.

**Verdict: NOT removed, because the decision is not clear-cut.** Unlike the spatial
filter, this step has a large effect on the answer. But:

- its effect is large *and* its justification is a single magic number with no
  physical basis;
- it is doing smoothing that the Kalman filter is also supposed to do;
- the honest reading is that a 300 m tolerance is standing in for an explicit
  physical layer model, which is what plan W4/W5 is for.

**Recommendation:** replace it with the layer model rather than tuning it. Until
then it should be treated as a declared modelling choice, not a neutral step. It is
reported here rather than changed unilaterally because removing it shifts the
product by 16 % and that is a scientific decision, not a cleanup.

## Not yet assessed

- **`min_cluster_size: 3`** — the stage audit showed it produces 170–480 clusters per
  frame and that the merge, not the clustering, performs layer separation. Once W4
  replaces the clustering this becomes moot.
- **`min_cluster_size_for_tracking: 8`** — a second size threshold downstream of the
  first; its effect has not been isolated.
- **The Reliability Index / TQI** — computed per frame and stored, but nothing
  consumes them. Either they should feed the self-assessment (plan W6) or they are
  dead weight.
- **`allow_single_cluster: true`** — not isolated.

## Method note

Every decision above was made on **internal** criteria (does the delivered series
change, and by how much) with the ceilometer untouched, as guardrail G1 requires.
The correlations reported elsewhere are *measurements of the result*, never inputs
to these choices.
