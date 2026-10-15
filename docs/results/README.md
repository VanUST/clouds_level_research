# Results index

Paper-usable results, each with the artefact that produced it and the command that
reproduces it. Every number here is traceable to a script, a JSON and a figure in the
repository; nothing is a recollection.

Read this first if you are writing from this work. The **findings report**
([`../report/findings.html`](../report/findings.html)) is the narrative version;
this directory is the citable index.

## How a result is allowed to be claimed

Read [`../plan/evaluation.md`](../plan/evaluation.md) before adding one. In short: no
value produced by maximising over alternatives without a null that maximises the same
way; effective sample size reported rather than nominal; a p-value or an explicit
statement that none exists; and the *direction* of an effect reported.

One metric has already been withdrawn under this rule
([`../plan/w1-calibration.md`](../plan/w1-calibration.md) W1.15, W1.16), and one of my
own claims was retracted after a gap-safe re-analysis. That is the standard to hold to.

| # | result | artefact | reproduces with |
|---|---|---|---|
| R‑18 | **`lowest_mode`**: a prominence rule replaces the low-tail order statistic — frame-to-frame stability improves 2.4× (31.6 → 12.9 m) and the level rises 17 % (1705 → 1994 m), with 2.7 % of frames abstaining | `configs/ablations/_mode.yaml`, `analyze.lowest_prominent_mode` | `python tools/evaluate.py --config configs/ablations/_mode.yaml --limit 300` |
| R‑19 | **Field masks and the measured focal constant** raise the level 10 % and halve the sub-1000 m contamination (6.7 % → 3.3 %) | `configs/ablations/_masks_f654.yaml` | `python tools/evaluate.py --config configs/ablations/_masks_f654.yaml --limit 300` |
| R‑20 | **The static pattern is 66 % instrumental and 34 % regime-dependent** — measured by partitioning the session by the ceilometer's own regime (cross-regime r = +0.371 against a within-regime ceiling of +0.718) | `experiments/layers/w3_h3_test.py` | `python experiments/layers/w3_h3_test.py` |
| R‑21 | **Removing the instrumental component does not fix the sign — it strengthens it** (−129 m → −164 m, ρ −0.547 → −0.581, p = 5.6e−16), and the inversion survives every radial cut and the diurnal confound | `experiments/layers/w3_unmask.py` | `python experiments/layers/w3_unmask.py` |
| R‑22 | **The inversion is carried by the match count**: ρ(matches, disparity) = +0.559 and ρ(matches, reference) = +0.425; the high regime returns ~2× the matches | `experiments/w2/W3_RESULT.md` | see W3.3 |
| R‑23 | **Certainty does not explain the inversion**: the high regime has the higher disparity in ALL ten certainty deciles (+0.82 to +2.50 px), and the surplus population's median certainty is 0.782 — high, not low | `experiments/w4/w4_certainty.py` | `python experiments/w4/w4_certainty.py` |
| R‑24 | **Both regimes are equally coherent**: local residual 0.047 px (low) vs 0.041 px (high), ratio 0.865, against a +1.785 px disparity difference | `experiments/w4/w4_smoothness.py` | `python experiments/w4/w4_smoothness.py` |
| R‑25 | **The raw match rate is identical between regimes** (11,655.1 vs 11,655.8 per frame) — RoMa emits a fixed grid, so the "2× match count" of W3.3 was a clustering artefact | `experiments/w4/dump_matches.py` | `python experiments/w4/dump_matches.py` |
| R‑26 | **The regime shift is not purely along the epipolar line**: the perpendicular residual nearly doubles (0.866 → 1.885 px, KS p ≈ 0) | `experiments/w4/w4_gate3.py` | `python experiments/w4/w4_gate3.py` |
| R‑27 | **The reference predicts a −5.315 px shift; the observation is +1.798 px** — opposite sign, 15 % of the magnitude | `experiments/w4/w4_gate3.py` | see gate3_result.json |
| R‑28 | **LoFTR sees the same direction, 6.5× smaller**: +0.275 px, 95 % CI [+0.112, +0.401] — matcher-independent in sign, representation-dependent in size | `experiments/w4/w4_gate4.py` | `python experiments/w4/w4_gate4.py` |
| R‑29 | **In the low regime the camera and ceilometer agree to 111 m** (2361 vs 2250 m) — the first genuine level agreement — but the correlation within that regime is still −0.473 (p = 2.3e−13) | `experiments/w4/GATE34_RESULT.md` | see the multi-layer test |
| R‑30 | **Cloudless null PASSES**: the clear day gives 655 m against 2242 m on the cloudy day (KS p = 0.000) — the level is content-driven, not a pipeline artefact | `experiments/w4/matches_roma_20250522.npz` | `python experiments/w4/dump_matches.py 260 2 2025-05-22` |
| R‑31 | **The desync-motion hypothesis is refuted**: ρ(speed, dv) = +0.002 (p = 0.97) against ρ(regime, dv) = +0.447; the required exposure offset is 18.14 s against an archive maximum of 5 s | `experiments/w4/w4_desync.py` | `python experiments/w4/w4_desync.py` |
| R‑32 | **The inversion survives within a single layer**: low-regime ρ = −0.473 per-frame, −0.525 gridded (p = 0.030), with the estimate's spread 1225 m against the reference's 620 m | `experiments/w4/w4_lowregime.json` | see W5.3 |
| R‑33 | **The photometric (non-Lambertian) hypothesis is refuted by its own sign**: the within-frame brightness–disparity coupling is negative (−0.176) while the high regime is brighter and has a *higher* disparity; the gradient-direction signature gives ρ = −0.000 | `experiments/w6/w6_photometric.py` | `python experiments/w6/w6_photometric.py 120 3` |
| R‑34 | **~~SYNTHETIC CONTROL PASSES~~ WITHDRAWN** — the first renderer tested an identically-oriented pair that does not exist | `experiments/w7/W7_RETRACTION.md` | see the retraction |
| R‑48 | **The shape mechanism is confirmed**: the median and mode move in OPPOSITE directions (+13.0 % vs −11.8 %) while the standard deviation is unchanged (−0.5 %), the skewness collapsing 68 % — the median/mode disagreement is what a changing shape does to order statistics | `experiments/w10/w10_signal.py` | `python experiments/w10/w10_signal.py` |
| R‑49 | ~~**The dynamics boundary refined, correcting W9**~~ **WITHDRAWN** — the reported weak correctly-signed correlation (median −0.409, p = 0.038) was an artefact of binning the reference at the bin START rather than over the bin. Corrected, the median's dynamics is **+0.095 (p = 0.667)** and nothing is significant. **W9's original conclusion stands** | `experiments/w10/CODE_REVIEW.md` bug 2 | see that record |
| R‑45 | **The static field is dominated by u, not radius**: a linear u-gradient explains 66 % of its variance against radius's 25 %, and the radial dependence largely cancels in height (corr(radius,height) = −0.146) | `experiments/w9/w9_cv.py` | `python experiments/w9/w9_cv.py` |
| R‑46 | **No cross-validated correction recovers the dynamics**: 6 families × 3 estimators × 5-fold leave the dynamics ρ in [−0.166, +0.168] with no consistent sign, while the best improves the level only from +0.137 to +0.202 | `experiments/w9/W9_RESULT.md` | see that record |
| R‑47 | **The inversion is estimator-dependent**: the mode estimator is correctly signed out of sample (+0.202, blocked+embargo) while the median and p25 are inverted (−0.386, −0.30) | `experiments/w9/w9_cv.py` | see W9_RESULT.md |
| R‑44 | **Phase aliasing is refuted on real data**: no period peaks above the 0.05 autocorrelation floor in either regime, and a +1.8 px jump would need a ~1.8 px texture wavelength against 100–1000 px cloud features. The tail is nonetheless 6× heavier in the high regime | `experiments/w8/W8_PHASE_ALIASING.md` | see that record |
| R‑42 | **The 551 m W8/W7 discrepancy was a `roll_deg` default mismatch between two scripts** (0.0 vs 1.5), not a physical effect; RoMa was verified deterministic (5 calls identical), and the control now reproduces at 1984 m | `experiments/w8/W8_SUPERPOSITION_RESULT.md` | see that record |
| R‑43 | **Photometric superposition is refuted**: it widens the CORE (IQR ×9.19) where the real atmosphere widens the TAILS (IQR ×1.11), and only at 80 % clear sky against the real 2.6–3.6 % | `experiments/w8/w8_superposition.py` | `python experiments/w8/w8_superposition.py` |
| R‑40 | **The broken-field / sky-hallucination hypothesis is refuted on real data**: the sky-like population is 2.6–3.6 %, below `lowest_mode`'s 10 % prominence floor, and cannot move a median by 1.80 px | `experiments/w8/W8_INJECTION3_TEST.md` | see that record |
| R‑41 | **The high regime's disparity field is shifted AND twice as dispersed**: the 1–99 percentile span goes 37.0 → 81.7 px (×2.2) while the central mass shifts +1.5 px | `experiments/w8/W8_INJECTION3_TEST.md` | see that record |
| R‑38 | **3-D relief is cleared**: up to 20 % of cloud height (2× physical), matcher ρ = +1.000 and `lowest_mode` ρ = +1.000, errors within 75 m | `experiments/w8/w8_relief.py` | `python experiments/w8/w8_relief.py` |
| R‑39 | **Texture scale is cleared**: the renderer's texture is metric-space, giving a 4× apparent-scale range (230 → 57 px), and the matcher recovers truth at every scale | `experiments/w7/w7_renderer.py` | see W8_RESULT.md |
| R‑35 | **VALIDATED SYNTHETIC CONTROL: the matcher recovers a known height** — ρ(truth, RoMa) = +1.000, median |bias| = 0.072 px, on a renderer whose gate passes exactly | `experiments/w7/w7_control.py` | `python experiments/w7/w7_control.py` |
| R‑36 | **The shipped `lowest` selection rule errs by 325–1471 m on synthetic ground truth** (675 m for a 1000 m cloud) while `lowest_mode` errs by ≤64 m | `experiments/w7/w7_control.py` | see W7_RESULT_FINAL.md |
| R‑37 | **The renderer's rig geometry, derived from the affine**: baseline along −Y, camera 2's sensor rotated −90°, camera 2's principal point at (1014.10, 965.03) = R·c1 + t | `experiments/w7/w7_renderer.py` | `python experiments/w7/w7_renderer.py` |
| R‑1 | **Matcher selection principle** — how to choose a matcher for a metrological task | [matcher-selection.md](matcher-selection.md) | see that document |
| R‑2 | Calibration: a ~1.6° residual roll between the cameras, measured three independent ways | `experiments/w1/epipolar_check.py` | `python experiments/w1/roll_by_session.py` |
| R‑3 | Lens distortion: cubic coefficient `a = −0.30` in `θ = (r/f)(1 + a(r/f)²)`, flattening the radial height trend from 36.7 % to 3.2 % | `experiments/layers/distortion_fit.py` | `python experiments/layers/distortion_fit.py` |
| R‑4 | Focal constant `f ≈ 654 ± 11 px/rad` against the assumed `ξ₀/π = 611` | `experiments/w1/f_identifiability.py` | `python experiments/w1/f_identifiability.py` |
| R‑5 | The ceilometer's lowest layer is bi-modal (40 % at 2250–2750 m, 53 % at 3750–4250 m) | `experiments/report/ceilometer_layers.py` | `python experiments/report/ceilometer_layers.py` |
| R‑6 | Sub-pixel matcher accuracy and differential sensitivity | `experiments/matchers/subpixel_accuracy.py`, `experiments/matchers/differential_sensitivity.py` | see [matcher-selection.md](matcher-selection.md) |
| R‑7 | RoMa's low-certainty interior predictions are accurate (0.68 px vs a 7.02 px null) | `experiments/matchers/roma_interior_check.py` | `python experiments/matchers/roma_interior_check.py` |
| R‑8 | Inference throughput: 2.03× with byte-identical outputs; the bottleneck was the spatial filter (217.7 ms/frame) | [`../plan/performance.md`](../plan/performance.md) | `experiments/w1/opt/ab_evaluate.py` |
| R‑9 | Pipeline steps removed on measured grounds (spatial filter, `cluster_selection_epsilon`) | [`../plan/pipeline-steps.md`](../plan/pipeline-steps.md) | see that document |
| R‑10 | Negative results, all of them | [negative-results.md](negative-results.md) | see that document |
| R‑11 | **Lens distortion applied end-to-end**: exact triangulation with `a = −0.30` flattens the pipeline's own radial height profile from 19.1 % (exact, ideal lens) / 4.1 % (paraxial) to **2.4 %**, with the median moving only −2.0 % | `experiments/layers/r2_radial_profile.py` | `python experiments/layers/r2_radial_profile.py <dump> <dump> <dump>` |
| R‑12 | **Clear-day Sun reference**: 3.5× tighter azimuth (p90 1.34° vs 4.69°) but no better radially; independently gives `f` = 663–683 px/rad | `experiments/w1/r9_clear_day.py` | `python experiments/w1/r9_clear_day.py` |
| R‑13 | **Variance decomposition**: the delivered estimate varies 12 % as much as the reference, from two causes — the raw retrieval under-responds 4× (24 %) and the Kalman filter halves it again (48 %) | `experiments/layers/r10_variance.py` | `python experiments/layers/r10_variance.py` |
| R‑14 | **The two geometry corrections together reduce the within-frame height spread 2.9×** (689 → 235 m) and raise the level 31 % (1441 → 1886 m), halving the gap to the reference | `experiments/w1/matches_cache_2025-05-23.npz` | isolated on identical matches, `docs/plan/research-workplan.md` RECONCILIATION |
| R‑15 | **The estimated height is dominated by a STATIC image-position pattern**: the height map correlates r = 0.98 (roll only) / 0.91 (roll + distortion) between the session's two halves, against a shuffled null of −0.002 | `experiments/layers/r11_identity.py` | `python experiments/layers/r11_identity.py` |
| R‑16 | **The static calibration error is dominated by a ~16 % horizontal SCALE, not by radial distortion**: linear-in-u explains 41.9 % of the static disparity field, radial terms only 13.5 % | `experiments/layers/r8_static_field.py` | `python experiments/layers/r8_static_field.py` |
| R‑17 | **The temporal filter is unwired and separable**: delivered std 91 m filtered vs 182 m unfiltered against a raw 176 m | `configs/ablations/_nofilter.yaml` | `temporal.filter_enabled` |

## Data facts that belong in any write-up

- **One usable session.** 2025-05-23, 08:00–17:00 UTC, 10 s cadence, two 1920×1920
  fisheye cameras 45 m apart. 2025-05-22 is cloud-free and yields single-digit
  correspondences per hour, so there is no independent session and no session-level
  hold-out.
- **Filenames are UTC; the ceilometer log is MSK; the offset is exactly +3 h**,
  confirmed independently by the diurnal brightness envelope and by a solar ephemeris
  matching the Sun's image azimuth to better than 0.5° across 14 hours.
- **No meteorology and no site coordinates exist for the session.** Auxiliary
  instruments cover 2023–2024.
- **The geometry is disparity-starved**: 18.3 px at 1.5 km, so 1 px is 5.5 % of height
  at 1.5 km and 10.9 % at 3 km. This single number sets the accuracy budget.

## Methodological notes worth citing

1. **Maximising over a search is not a measurement.** The project's own
   windowed-correlation metric searched a time shift per window and reported the
   maximum; a circular-roll null that performed the same search produced *higher*
   values than the observation. See [`../plan/temporal-alignment.md`](../plan/temporal-alignment.md).
2. **Effective sample size, not nominal.** A Kalman-filtered series had ρ = 0.979 over
   five hours — 18.8 independent samples from 1,784 — which caps the smallest
   detectable correlation at ~0.49. Smoothing destroys the variability validation
   depends on.
3. **A proxy metric can reward the wrong thing.** A plane-fit residual ranks a
   consistently quantised field as highly as an accurate one; MASt3R ranked first on
   it and has zero differential response.
4. **Measure the property the product consumes.** See
   [matcher-selection.md](matcher-selection.md).
