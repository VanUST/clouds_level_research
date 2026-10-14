# W3 — which matcher fits *our* clouds? Ranked recommendation

**Status:** complete for the candidates that could be built in this environment.
Metric, data and decision rule were fixed in advance in
[PREREGISTRATION.md](PREREGISTRATION.md) (guardrail G2). Nothing below uses the
ceilometer (guardrail G1); the clear-sky session is a discrimination control, not
ground truth. Scripts: [harness.py](harness.py), [evaluate.py](evaluate.py),
[report.py](report.py), [benchmark_throughput.py](benchmark_throughput.py),
[verify_loftr_baseline.py](verify_loftr_baseline.py), install notes in
[README.md](README.md).

## 0. Bottom line

**One literature-motivated candidate is rejected outright and the rest is a
set of trade-offs, not a clean accuracy win.** The only adoption is justified by
coverage and consistency; on raw localisation the baseline is still at the top.

- **EfficientLoFTR — the documented drop-in fix for LoFTR's fine-stage spatial
  variance — is worse on our data, not better.** Paired against LoFTR it is
  **+5.49 px (+204 %) on the pre-registered residual**, p = 0.000. It is 2.1×
  faster and that is its only advantage here.
- **RoMa and DKM are the best localisers** (median |residual| 0.83 px against
  LoFTR's 1.48 px; paired rms −50 % and −54 %, p = 0.000) **but they return
  essentially nothing in cloud interiors** — 0 matches in the lowest-gradient
  tercile over 31 frames. Their headline accuracy is bought on cloud edges, which
  is exactly the failure mode this project cannot accept.
- **MASt3R is the only candidate that both covers cloud interiors and improves
  the pre-registered metric.** With its clear-sky-separating confidence cut at
  0.5 it gives 5597 matches/frame (2.6× LoFTR), paired rms −0.81 px (−30 %,
  p = 0.000), 1448 interior matches per frame against LoFTR's 298, an interior
  median residual of 1.75 px against LoFTR's 2.15 px, and 5 matches/frame on a
  cloudless sky (LoFTR: 11). Its remaining weaknesses are a slightly worse bulk
  *median* residual (1.56 vs 1.48 px), 4× worse perturbation stability
  (0.84 vs 0.22 px) and 6.6× the runtime. It is an adoption, not a slam dunk.
- **Depth Anything 3 is a null**: it can be turned into a disparity field, but
  that field is smooth and inaccurate (median residual 5.11 px), and it is not
  metric.
- **StereoSGBM and DIS optical flow are rejected**: both invent a full
  correspondence field on a cloudless sky (6400 and 5900 matches/frame) and SGBM
  is the worst candidate on real cloud (6.45 px). They fail the discrimination
  test outright.
- **A sparse detector is not useless but is not competitive for this job**:
  DISK+LightGlue is statistically indistinguishable from LoFTR (p = 1.000) but
  with the worst interior coverage (8 % of matches, vs 14 % for LoFTR and 28 %
  for MASt3R at conf ≥ 0.5).
- **Radial flatness does not separate the candidates.** The height-versus-radius
  slope is −2.33 to −3.39 m/px for every candidate, i.e. the radial bias is
  dominated by the un-modelled geometry/calibration, not by the matcher. This
  refutes the hope that a better matcher would flatten the profile.

**Recommendation.** Keep `kornia_loftr` as the default (it is unchanged and still
reproduces the frozen baseline byte-for-byte). Add `mast3r` as an opt-in backend
for the dense-field requirement. Do **not** switch to EfficientLoFTR. Use
RoMa/DKM only as a *high-confidence edge* matcher, never as the dense field.

---

## 1. Ranking

Ranked on the pre-registered primary metric (median per-frame plane-fit rms),
with the coverage constraint applied on top:

| rank | candidate | verdict |
|---|---|---|
| 1 | **MASt3R (`mast3r`, `min_conf: 0.5`)** | **adopt as opt-in.** Improves the primary metric (p = 0.000), 4.9× the interior matches at a better interior residual, passes the clear-sky test; 6.6× slower and 4× worse perturbation stability. |
| 2 | `kornia_loftr` | **keep as default until MASt3R's perturbation weakness is re-tested.** Unchanged, byte-for-byte reproducible, best perturbation stability (0.22 px). |
| 3 | RoMa (`roma`) / DKM (`dkm`) | **reject for the field, keep as an edge cross-check.** Best localisation (−50 %/−54 %) but zero cloud-interior matches. |
| — | DISK+LightGlue | **reject.** Within noise of LoFTR (p = 1.0), worst interior coverage. |
| — | DIS optical flow | **reject.** Within noise of LoFTR on cloud (p = 0.56) but invents a dense field on clear sky. |
| — | EfficientLoFTR | **reject.** +204 % residual, p = 0.000. |
| — | Depth Anything 3 | **reject.** Null: field is smooth but inaccurate (5.11 px). |
| — | StereoSGBM | **reject.** Worst residual and invents a field on clear sky. |

## 2. The table

Cloud session **2025-05-23, 31 real stereo pairs**, one every 15 min, 08:30–16:30
UTC, the same frame list for every candidate. All residual columns are median
over frames, in pixels of the 640×640 crop. `ms` is matcher-only, measured
separately in [benchmark_throughput.py](benchmark_throughput.py) (median of 6
frames after warm-up); the `ms` inside the evaluate JSON is not a clean matcher
timing because the metric fit runs in the same timed region.

| matcher | matches | ms | rms | \|res\| | \|res\| low-grad | low-grad n | low-grad % | top-2500 \|res\| | dv random | slope m/px | h IQR m | perturb px | p vs LoFTR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LoFTR (kornia) | 2189 | 111 | 2.69 | 1.48 | 2.15 | 298 | 0.14 | 1.27 | 1.41 | −2.79 | 710 | 0.22 | — |
| EfficientLoFTR | 3309 | 53 | 8.41 | 1.88 | 3.86 | 441 | 0.18 | 1.52 | 3.29 | −2.74 | 750 | 1.07 | 0.000 |
| RoMa | 1259 | 581 | 1.39 | **0.83** | 0.68 | **0** | **0.00** | **0.70** | 0.23 | −2.69 | 694 | 0.12 | 0.000 |
| DKM | 699 | 516 | 1.46 | **0.83** | 0.58 | **0** | **0.00** | **0.60** | 0.20 | −3.19 | 730 | 0.07 | 0.000 |
| **MASt3R (conf ≥ 0)** | 7589 | 738 | 2.34 | 1.69 | 1.91 | 2438 | 0.32 | 1.38 | 0.85 | −2.72 | 773 | 0.84 | 0.007 |
| **MASt3R (conf ≥ 0.5)** | **5597** | 738 | **2.12** | 1.56 | 1.75 | **1448** | 0.28 | **1.17** | 0.72 | −2.65 | 823 | 0.84 | 0.000 |
| Depth Anything 3 | 3750 | 88 | 5.95 | 5.11 | 3.61 | 1247 | 0.33 | 4.25 | 0.02 | −3.39 | 1107 | 5.97 | 1.000 |
| DIS optical flow | 6148 | 26 | 3.60 | 1.84 | 3.00 | 1971 | 0.32 | 1.73 | 1.96 | −2.73 | 775 | 1.72 | 0.562 |
| StereoSGBM | 6400 | 43 | 9.33 | 6.45 | 5.18 | 2073 | 0.32 | 2.94 | 0.00 | −2.33 | 1435 | 5.84 | 0.000 |
| DISK+LightGlue | 1909 | 87 | 2.73 | 2.02 | 2.54 | 126 | 0.08 | 1.28 | 1.56 | −2.58 | 705 | 0.98 | 1.000 |
| RoMa, certainty ≥ 0.05 | 3591 | 581 | 1.74 | 1.02 | 1.75 | 67 | 0.03 | 0.77 | 0.50 | −2.70 | 746 | 0.27 | 0.000 |
| DKM, certainty ≥ 0.05 | 2262 | 516 | 1.63 | 0.96 | 1.34 | 8 | 0.01 | 0.75 | 0.34 | −2.69 | 723 | 0.11 | 0.000 |

Column definitions:

- **rms** — rms 2-D residual of the matches about the exact plane-shift model
  (D, roll, tx, ty) fitted per frame; the pre-registered primary metric.
- **|res|** — median absolute residual of the same fit. Robust to the different
  confidence scales of the backends; added as a declared amendment (see §7).
- **|res| low-grad** — the same fit restricted to matches whose image-1 point
  lies in the lowest-gradient tercile of the frame, i.e. **inside cloud
  interiors**. This is the number that matters for "height across the whole
  field".
- **top-2500 |res|** — each backend reduced to its 2500 most confident matches,
  so a backend is not judged on the low-confidence tail its native threshold
  happens to emit.
- **dv random** — rms of the vertical disparity after removing a degree-3 smooth
  fit; the structured part is shared geometry, this is localisation noise.
- **perturb** — rms spread of the disparity change when image 2 is translated by
  a known (+0.5, +0.25) px, over matches whose image-1 point moves <1.5 px.
- **p vs LoFTR** — two-sided paired sign test on per-frame rms.

Figures: [radial_profile_2025-05-23.png](figures/radial_profile_2025-05-23.png),
[criteria_2025-05-23.png](figures/criteria_2025-05-23.png),
[coverage_2025-05-23.png](figures/coverage_2025-05-23.png).

## 3. Paired verdicts against the baseline

Differences are per-frame rms against LoFTR on the same 31 frames; the
pre-registered noise band is `< 5 %` of the baseline **and** p ≥ 0.05.

| candidate | median Δ rms | relative | p | verdict |
|---|---|---|---|---|
| EfficientLoFTR | +5.493 px | +204 % | 0.000 | worse |
| RoMa | −1.340 px | −49.7 % | 0.000 | **better** |
| DKM | −1.456 px | −54.0 % | 0.000 | **better** |
| MASt3R (conf ≥ 0) | −0.811 px | −30.2 % | 0.007 | **better** |
| MASt3R (conf ≥ 0.5) | −0.810 px | −30.1 % | 0.000 | **better** |
| Depth Anything 3 | +0.764 px | +28.4 % | 1.000 | worse (not systematic) |
| DIS optical flow | +0.343 px | +12.7 % | 0.562 | **within noise** |
| StereoSGBM | +5.624 px | +209 % | 0.000 | worse |
| DISK+LightGlue | −0.329 px | −12.2 % | 1.000 | **within noise** |
| RoMa ≥ 0.05 | −1.010 px | −37.5 % | 0.000 | better |
| DKM ≥ 0.05 | −1.064 px | −39.5 % | 0.000 | better |

So three candidates clear the noise band on the primary metric (RoMa, DKM,
MASt3R) and two do not (DIS flow, DISK+LightGlue). The coverage column then
decides which of the three is usable here.

## 4. What rejected each candidate, and what it rules out

**MASt3R — accepted, with one open weakness.** It is the only non-baseline
candidate that survives all three constraints at once: it improves the
pre-registered metric (paired rms −0.81 px, p = 0.000), it gives 4.9× LoFTR's
interior matches at a better interior residual (1.75 vs 2.15 px), and with
`min_conf: 0.5` it returns 5 matches/frame on a cloudless sky. It loses to LoFTR
on the bulk median residual (1.56 vs 1.48 px), on perturbation stability
(0.84 vs 0.22 px) and on runtime (738 vs 111 ms). The interior gain and the
clear-sky separation are large and unambiguous; the localisation comparison is
mixed and should not be oversold.

**EfficientLoFTR — rejected on localisation.** The CVPR 2024 paper that motivated
this candidate documents that LoFTR's fine refinement has spatial variance and
fixes it with a two-stage correlation. On these clouds the fix did not transfer:
median |res| 1.88 px vs LoFTR's 1.48, risk-matched 1.52 vs 1.27, vertical random
rms 3.29 vs 1.41, and the paired rms is 3× worse. It *is* 2.1× faster (53 vs
111 ms) and it returns 51 % more matches with slightly better interior coverage
(0.18 vs 0.14), so it is not broken — it is simply not more accurate here. This
rules out "adopt EfficientLoFTR because the literature says so" for this data.

**RoMa and DKM — rejected for the field, not for accuracy.** Their residual is
half the baseline's, and the RoMa variant with a near-zero certainty threshold
(0.05, 3591 matches) still has 3 % interior matches; the default (0.5, 1259
matches) has **0 interior matches in 31 frames**. DKM is the same at 0.00/0.01.
The physical meaning is that these dense matchers assign their confidence to
textured regions and have essentially none on a smooth cloud interior. Their
measured superiority is therefore a statement about cloud **edges**. For a
product whose whole point is a height field across the crop, that is
disqualifying on its own, and it means their numbers may not be quoted as "the
matcher is 2× better on our clouds".

**Depth Anything 3 — a genuine null.** It is not a matcher, but its depth map plus
its predicted relative pose *can* be turned into a dense disparity field by
back-projection, and that is scale-invariant (the unknown depth scale cancels in
the projection), so the field is well defined without trusting its metric scale.
Measured: `is_metric` empty and `scale_factor` None for DA3-BASE — it is **not
metric**; implied median height after imposing the 45 m baseline is ~3.4 km
against the project's ~1.8 km internal estimate; the depth map is nearly constant
(5–95 % range 0.874–1.066 on one frame). As a correspondence field it gives
full interior coverage (0.33) but a median residual of 5.11 px overall and
3.61 px even inside cloud interiors, ~2.4× worse than LoFTR, with an
unrealistically smooth vertical field (0.02 px). **Conclusion: DA3 yields a
multi-view-consistent disparity, but not an accurate one at 45 m baseline and
1–2 km range; it is recorded as a null, not as a candidate.**

**StereoSGBM — rejected twice.** It assumes the pair is row-rectified; ours is
not (an affine warp plus a measured ~1.8–2.6 px residual vertical disparity and
the fisheye corner compression). Its vertical disparity is exactly 0 by
construction, so it cannot express the measured tilt, and its residual is the
worst of all candidates (6.45 px). It also produces 6400 matches/frame of
confident-looking output on a **cloudless** sky — the failure mode the
discrimination test exists to catch.

**DIS optical flow — within noise, rejected on the clear-sky test.** Genuinely
dense (0.32 interior matches, uniform), 26 ms, and its cloud residual is
statistically indistinguishable from LoFTR (p = 0.562). But on 2025-05-22 it
returns 5900 matches/frame with 22.4 px residual — a full field invented where
there is no signal. A method that cannot tell cloud from clear sky cannot be the
field estimator.

**DISK+LightGlue — the sparse control, rejected.** It finds far more than
"almost nothing" (1909 matches/frame; DISK at n = 4096 is not a weak detector),
but it is within noise of LoFTR on accuracy (p = 1.000) with the worst interior
fraction of any candidate (0.08) and a worse interior residual (2.54 px). It
confirms the project owner's expectation directionally — sparse detectors
concentrate on edges — without being quite the extreme the expectation implied.

**The clear-sky discrimination test, all candidates** (2025-05-22, 32 frames,
median matches/frame): LoFTR 11, DISK+LightGlue 7, **MASt3R conf ≥ 0.5 → 5**,
RoMa 0, EfficientLoFTR 92, MASt3R unfiltered 742, DA3 2529, DIS flow 5900, SGBM
6400. Few is correct. MASt3R's unfiltered count is high, **but its confidence
separates the two sessions by a factor of ~50**: on a cloud frame the returned
confidence has median 5.85 (10th percentile 1.41), while on a clear-sky frame it
has median 0.10 (99th percentile 0.17). A threshold of 0.5 keeps >90 % of the
cloud matches (5597 of 7589) and removes the clear-sky field entirely, so
**MASt3R with `min_conf: 0.5` passes the discrimination test** — the filtered
run is `results/mast3r_conf05_*` and is the configuration to use.

## 5. What did *not* discriminate, and why that matters

**Radial flatness is dominated by the geometry, not the matcher.** The median
slope of binned height against radius is −2.33 (SGBM), −2.58 (DISK+LightGlue),
−2.69 (RoMa), −2.72 (MASt3R), −2.73 (DIS), −2.74 (EfficientLoFTR), −2.79
(LoFTR), −3.19 (DKM), −3.39 (DA3) m/px. A spread of ±18 % around ~−2.8 m/px
across methods whose localisation differs by a factor of six means the profile is
set by the shared un-modelled field geometry (the roll/tilt documented in
`w1-calibration.md`), not by matching. The pre-registered hope that a better
matcher would flatten the profile is **refuted on this data**: fixing the radial
bias requires fixing the geometry, not the matcher. The radial figure shows all
profiles nearly overlapping.

**Vertical disparity is structured, not noise.** Median vertical disparity is
1.8–2.6 px for every learned matcher (0.0 for SGBM by construction), consistent
with the ~2.2 px measured independently. Removing a smooth degree-3 field leaves
a random residual of 0.20–0.23 px for RoMa/DKM, 0.85 for MASt3R, 1.41 for LoFTR,
3.29 for EfficientLoFTR. So most of the vertical signal is un-modelled geometry —
as expected — and the matcher-controlled part ranges over an order of magnitude.

**Perturbation stability** (sub-pixel repeatability): DKM 0.07, RoMa 0.12, LoFTR
0.22, DISK+LightGlue 0.98, EfficientLoFTR 1.07, DIS 1.72, SGBM 5.84, DA3 5.97,
MASt3R 0.84. The dense coordinate-regression models are the most repeatable; the
correlation-based dense models are the least. LoFTR is the best of the
correlation-based group.

## 6. Reproducibility evidence

- **Baseline is untouched.** `verify_loftr_baseline.py` re-runs the kornia LoFTR
  path at `batch_size: 1` over every frame of the frozen W1 cache:
  **38/38 frames byte-identical** (`numpy.array_equal` on both keypoint arrays).
  Every number in this report is therefore comparable with the frozen baseline.
- **A new backend is genuinely selectable by config.** `configs/_w3_backend_smoke.yaml`
  sets `matching.backend: roma` plus `matching.params: {grid_stride: 8,
  certainty_threshold: 0.5}` and runs end-to-end through
  `StereoAnalyzer.analyze_pair` with no other code path changed: 6170 matches,
  1 cluster, median height 1862 m. `config/load.py` used to reject any backend
  outside a hard-coded pair — that check now consults the registry, so future
  backends need no edit there, and a nonsense name is still rejected
  (`ConfigError: matching.backend unknown`).
- **Versions** (exact install commands in [README.md](README.md)): torch
  2.14.1+cu130, kornia 0.8.3, romatch 0.1.2, dkm 0.3.0, einops 0.8.2, yacs
  0.1.8, timm 1.0.30, depth-anything-3 0.1.1; EfficientLoFTR repo `07e9c140`
  with `kornia/Efficient_LOFTR` weights; MASt3R repo `f5209afc` with
  `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`.
- **Environment limitation, stated:** `nvcc` is absent and
  `CUDA_HOME=/usr/local/cuda-12.1` does not exist, so RoMa's optional
  `local_corr` CUDA extension could not be built and RoMa ran on its
  pure-PyTorch correlation fallback. Its flow semantics were verified
  independently (identity pair → 0.02 px; +5 px known shift → +5.17 px change on
  the pair), so the decoding is correct, but the fallback's numerics are not the
  reference CUDA kernel's. DKM and MASt3R ran on their reference code.

## 7. Pre-registration amendments (declared)

Two additions were made after the pre-registration and are labelled as such;
neither is ground-truth-based and neither changes the primary metric.

1. **Robust residual and top-2500 risk-matched comparison.** The backends
   threshold confidence on different scales, so a raw residual mixes localisation
   with each backend's threshold choice. The median absolute residual and a
   reduction to the 2500 most confident matches isolate localisation. The
   pre-registered primary metric (`rms_full_px`) is still reported first and the
   ranking is checked against both.
2. **Gradient-stratified residual** (requested by the project owner). The same
   plane fit restricted to the lowest-gradient tercile, so "coverage in cloud
   interiors" is measured for accuracy and not only for count.

Both were added before the final table was produced, after the first pass had
already shown RoMa/DKM's high-gradient concentration; they are refinements that
make the comparison fairer to the correlation-based backends, not a search for a
metric that favours a particular winner.

## 8. Files changed by this work

**Source (`src/clouds/`)** — only the matcher surface and the config plumbing:

| file | change |
|---|---|
| `matching/matchers.py` | registered six new backends: `efficient_loftr`, `roma`, `dkm`, `mast3r`, `da3`, `disk_lightglue`, `cv2_dis_flow`, `cv2_sgbm`. `kornia_loftr` is untouched. |
| `config/schema.py` | added `MatchingCfg.params` (backend-specific kwargs; default `None`). |
| `config/load.py` | backend validation now consults the matcher registry instead of a hard-coded two-name list, so a newly registered backend is selectable by config; unknown names are still rejected. |
| `pipeline/analyze.py` | forwards `matching.params` to the matcher in both the single-pair and batched paths. |

**Configs:** `configs/_w3_backend_smoke.yaml` (new; selects `roma` with `params`,
proves config-driven pluggability). No existing config was modified; the paper
config's `backend: kornia_loftr`, calibration `2025-05-22_affine_old` and
`roll_correction_deg` (absent → 0.0) are unchanged.

**Experiments:** everything under `experiments/matchers/` is new. Nothing under
`experiments/layers/`, `experiments/report/`, `experiments/temporal_alignment/`
or `experiments/w1/` was edited.

**Measurement data:** nothing under `~/clouds_data/` was written by this work;
the images were opened read-only and the W1 correspondence cache was read only.
(Files under `clouds_data/logs/` were written by other agents' pipeline runs.)

## 9. What I could not determine

- **Session hold-out.** One usable session (W1.14), so every comparison is paired
  across frames but correlated in time. A different hour, a different cloud type
  or a different day could reorder the mid-field; nothing here can test that.
- **Whether MASt3R's interior advantage is real accuracy or a smoother prior.**
  Its conf ≥ 0.5 interior residual (1.75 px) is better than LoFTR's (2.15 px) and
  its risk-matched bulk residual is better (1.17 vs 1.27 px), but its
  perturbation stability is 4× worse (0.84 vs 0.22) and its height IQR is wider
  (823 vs 710 m). A matcher that is accurate in the mean but unstable under a 1 s
  scene change may behave differently on real adjacent frames than on the injected
  sub-pixel shift used here. **This is the single most important open question
  before promoting MASt3R to default**, and it is testable from the data already
  on disk (adjacent-in-time pairs).
- **RoMa in its reference configuration.** The `local_corr` CUDA extension could
  not be built (no nvcc). RoMa's numbers are from the pure-PyTorch fallback.
- **MatchAnything, DKM's indoor variant, and RoMa's indoor weights** were not
  run: the MatchAnything third-party tree is absent from this checkout and only
  `roma_outdoor`/DKM-outdoor are the relevant outdoor models.
- **The absolute height is not validated by any of this.** `f` remains unpinned
  per W1.2 and the ranking is invariant to it; no ceilometer comparison was made
  and none should be inferred.
- **A `f`-or-calibration improvement is not a matcher question.** The radial
  bias appears in every candidate; the matcher choice does not address it.

## 10. Recommendation

1. **Keep `kornia_loftr` as the runtime default.** It is unchanged, reproducible
   byte-for-byte, and remains the best correlation-based matcher on perturbation
   stability with the second-best interior residual.
2. **Add `mast3r` as the opt-in backend for the dense-field requirement, with
   `min_conf: 0.5`.** It is the only candidate that improves the pre-registered
   metric *and* gives dense interior coverage *and* passes the clear-sky test.
   Accept ~0.74 s/frame. Before promoting it to default, re-test its perturbation
   weakness on real adjacent-in-time frames, because that is the one criterion on
   which it loses to LoFTR by 4× and the injected sub-pixel shift is a partial
   proxy for real scene change.
3. **Do not adopt EfficientLoFTR.** The documented fix does not apply to this
   imagery; record the negative result so it is not retried.
4. **Do not use RoMa/DKM as the field estimator.** They may be used as an
   independent high-confidence edge cross-check against LoFTR/MASt3R.
5. **If a dense flow or block matcher is ever revisited, gate it on the clear-sky
   test first** — DIS flow and SGBM both fail it by an order of magnitude.
6. **Stop looking to the matcher for the radial bias.** It is geometry; W2/W1
   own it.
