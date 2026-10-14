# Pre-registration — W3 matcher comparison

**Written before the first measurement.** Guardrail G2 requires that the metric,
the data, the configuration and the failure condition are fixed in advance, so a
metric chosen after the result cannot select the matcher. Amended once, on
2026-10-13, to add gradient-stratified coverage after the project owner asked for
it; no candidate had been scored at that point.

Tooling: [`harness.py`](harness.py) (criteria), [`evaluate.py`](evaluate.py)
(runner), [`report.py`](report.py) (aggregation and figures).

---

## 1. Question

Which feature-matching / dense-correspondence method produces the most accurate
per-match horizontal disparity on **these** cloud images, where accuracy means
sub-pixel localisation, because the geometry is disparity-starved (18.3 px at
1500 m, so 1 px is 5.5 % of height at 1.5 km)?

## 2. Data (fixed)

- **Evaluation frames:** 2025-05-23, image-clock UTC, one stereo pair every
  **15 minutes from 08:30:00 to 16:30:00** (33 pairs), the frame nearest each grid
  time. The same list is used for every candidate, so the comparison is paired.
- **Control — clear sky:** 2025-05-22, the same 15-minute grid over
  08:30–16:30 (33 pairs). This session has no cloud signal; a matcher that
  returns many confident matches there is inventing them.
- **Calibration:** `2025-05-22_affine_old`, `camera.roll_correction_deg` as it
  ships (0.0). The roll value is **not** changed. Where the analysis needs the
  shared calibration roll removed (it inflates the residual of every candidate
  equally), it is fitted as a nuisance parameter inside the metric, not applied to
  the config.
- **f** = `camera.image_width_px / pi` = 611.15 px/rad, baseline 45 m, crop
  640×640 as the pipeline ships. `f` is a geometry constant that scales all
  heights; it cancels in every *comparison* here because all candidates use the
  same value, so the un-pinned `f` (W1.2) does not affect the ranking.

## 3. Metrics (fixed)

All are functions of `(image pair, calibration)` only. The ceilometer is not read
by any of them (guardrail G1).

| # | name | definition | better |
|---|---|---|---|
| 1 | **sub-pixel consistency** — `rms_full_px` | rms 2-D residual of the matches about the exact plane-shift model (D, roll, tx, ty) fitted per frame by `harness.plane_fit`; also reported: median ray-intersection residual (`ray_miss_median_m`) from `clouds.geometry.triangulation` | smaller |
| 2 | **epipolar discipline** — `dv_rand_rms_px` | rms of the vertical disparity after removing a degree-3 polynomial in (u, v); the structured part is reported separately | smaller random part |
| 3 | **radial flatness** — `slope_m_per_px`, `height_iqr_m` | slope of binned median height against radius, and the within-frame height IQR | smaller magnitude |
| 4 | **perturbation stability** — `perturb_ddu_rms_px` | rms spread of the change in horizontal disparity when image 2 is shifted by a known (+0.5, +0.25) px (bilinear), measured over matches whose image-1 point is within 1.5 px between the two runs | smaller |
| 5 | **coverage / throughput** | matches per frame (median), fraction of frames with ≥100 matches, ms per frame (matcher only, `torch.cuda.synchronize` around it), and the fraction of matches whose image-1 point falls in the lowest-gradient tercile of the frame | reported |
| 6 | **clear-sky behaviour** | median matches per frame on 2025-05-22 | few is correct |

**Gradient strata** are fixed as follows: Sobel magnitude of the prepared gray1
crop, smoothed with a 9×9 box; pixels below the frame's 33rd percentile are
"low-gradient", above the 67th are "high-gradient". A match is attributed to the
stratum of its image-1 point. `low_grad_frac` = low-gradient matches / all
matches; `low_grad_density` = low-gradient matches / low-gradient pixels.

## 4. Decision rule (fixed)

1. **Primary ranking is by metric 1**, median over the 33 evaluation frames.
2. A candidate is **within noise of the baseline** if the paired per-frame
   difference in metric 1 against `kornia_loftr` has a median whose magnitude is
   below 5 % of the baseline median **and** a two-sided paired sign test gives
   p ≥ 0.05. In that case the candidate does not displace the baseline: coverage,
   radial flatness and perturbation stability are then used only to describe it.
3. A candidate that is worse than the baseline on metric 1 by more than the noise
   band is rejected regardless of coverage.
4. Speed is reported and never used to rank.
5. **Failure condition stated in advance:** if no candidate improves metric 1
   beyond the noise band, the conclusion is *"the current LoFTR is within noise of
   the alternatives on this data"*, and that is reported as the result.

## 5. What would refute a favourable finding

- A candidate that wins on metric 1 only by returning very few, easy-to-match
  points: refuted by its coverage on metric 5 and by `low_grad_frac`.
- A candidate whose radial flatness is flat only because its heights are
  systematically biased or NaN-heavy: refuted by reporting `valid_fraction`.
- A candidate whose apparent stability comes from a coarse quantisation of the
  output: refuted by metric 4, which measures sub-pixel response to an injected
  sub-pixel shift.

## 6. Known limits (stated up front)

- One usable session, so there is no session-level hold-out (W1.14). Every
  comparison is paired across frames but correlated in time.
- `f` and the calibration are unpinned (W1); a matcher cannot fix that, and the
  ranking does not depend on it.
- Perturbation stability uses an injected sub-pixel translation, not real cloud
  evolution; it measures localisation repeatability, which is the part of
  "stability under the ~1 s between cameras" a matcher controls.
