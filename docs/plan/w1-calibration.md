# W1 — calibration verification and the workable margin

**Status:** in progress. Findings below are falsifiable and each states the
evidence that would refute it. `f` is **not yet pinned** to the precision the plan
requires; the reason is recorded.

Tooling: [`experiments/w1/sun_reference.py`](../../experiments/w1/sun_reference.py)
(solar ephemeris, Sun detection, infinity test). Raw residuals:
`experiments/w1/infinity_test.json`.

---

## W1.1 — The epoch and the camera orientation are now measured, not assumed

The Sun's image azimuth is predicted by a solar ephemeris plus **one constant**
orientation offset, and the prediction matches the detected Sun to **better than
0.5° across 14 hours** (04:00–17:00 UTC on 2025-05-23, 14 sampled frames).

| frame (UTC) | sun zenith | ephemeris azimuth | predicted image azimuth | detected |
|---|---|---|---|---|
| 04:00 | 68.7° | 95.8° | 141.5° | 141.5° |
| 06:00 | 52.2° | 69.7° | 115.4° | 115.3° |
| 08:00 | 38.8° | 34.5° | 80.2° | 80.4° |
| 12:00 | 44.9° | 305.7° | 351.4° | 355.3° |
| 16:00 | 77.2° | 251.8° | 297.5° | 297.5° |
| 17:00 | 85.0° | 239.9° | 285.6° | 285.6° |

The offset is **A₀ = −45.7°** (image azimuth = sun azimuth − A₀).

Three consequences, each independently useful:

1. **The filename clock is UTC.** An ephemeris cannot match a clock that is three
   hours wrong; this is an independent confirmation of the conclusion reached in
   the audit from the diurnal brightness envelope, by a completely different
   route.
2. **The rig's azimuth orientation is known** to ~0.5°, which the calibration
   matrix does not by itself record.
3. It provides the **absolute angular reference** the project previously lacked.

**It would be refuted by:** the same test on the 2025-05-22 session failing to
reproduce the offset, or a residual that grows systematically across the day
(which would indicate a wrong site, wrong timebase, or a non-equidistant
projection).

## W1.2 — The projection constant `f` is bounded but NOT yet pinned

`f` is the one geometry constant that scales every height (`D = f·B/Δξ`). Evidence
gathered, all from 2025-05-23:

| method | result | ratio to ξ₀/π = 611.2 |
|---|---|---|
| Sun radius vs zenith, fitting `r = f·θ + r₀` (12 sun-centred blobs) | **f = 597.3 ± 2.7**, r₀ = 67.5 px, rms 8.3 px | 0.977 |
| same data, forcing r₀ = 0 | **f = 664.0** | 1.087 |
| image-circle radius 1078–1156 px, *if* the lens spans exactly 180° | **f = 686–736** | 1.12–1.20 |

**The first two rows are the same data and they disagree**, because `f` and a
radial bias of the blob centroid are degenerate over this zenith range
(θ = 0.62–1.48 rad). The measurement cannot currently distinguish
"*f ≈ 600 with the blob biased 68 px outward*" from "*f ≈ 664 with no bias*".

So the honest statement is:

> **f lies between roughly 600 and 690 px/rad.** The assumed value 611 sits at the
> low end. If the truth is 664–690, every published height is **8–11 % too low**
> (a 1500 m cloud reported as 1330–1356 m).

**It would be refuted by:** a bias-free Sun-disc position, or an independent
angular reference. Both are the next task (§W1.4).

## W1.3 — The infinity test runs, but its estimator is not yet trustworthy

The test is implemented and produces residuals: the Sun's position in devID1 is
predicted in devID2 through the fixed affine and compared with the Sun found
there. Over 27 frames the residual ranged **4.3 – 175 px, median 24.7 px**
(1 px = 5.5 % height error at 1500 m).

**These numbers must not be read as calibration error.** The detected point is the
centroid of the whole saturated region, which is much larger than the solar disc
and changes shape between the two views. The residual therefore mixes calibration
error with blob-shape difference.

Two things it does establish:

- The residual is **small and stable** (4–11 px) on the frames with the most
  compact blob, and **blows up** near the horizon (75–175 px at zenith 74–85°)
  where the blob is clipped by the image circle and by trees. That pattern is
  consistent with a shape artefact rather than with a calibration error, which
  would not depend on zenith angle in this way.
- A grossly wrong calibration is **excluded**: it would produce a large residual
  at every zenith angle, not a small one in the middle of the day.

## W1.4 — Negative results (recorded so they are not repeated)

1. **"Brightest saturated blob" is not the Sun.** Thresholding at 250–252 finds
   saturated *cloud*, with areas 10⁴–4×10⁴ px against ≈26 px for the solar disc
   (0.53° at f ≈ 611). An early attempt to use it as the Sun measured cloud
   disparity and was discarded.
2. **Gradient-convergence glow centring fails on this imagery.** The idea — the
   intensity gradient points at the Sun, so the gradient lines intersect at its
   centre — gives r/θ scattered from 492 to 945 px/rad and azimuths wrong by up
   to 15°. The solar glow is not radially symmetric (clouds, trees, the image
   circle edge dominate it), so the method has no consistent centre to find.
3. **The 300 m-baseline literature is not directly applicable to `f`.** Those
   rigs measure `f` from star fields; this project has no usable star data (night
   frames are too dark to contain stars).

## W1.5 — Tooling defect found and fixed (affects any future ephemeris work)

The azimuth branch of the solar position requires the **signed** hour angle. My
first implementation folded the hour angle into [0, 360°), which makes the
`if ha > 0` test always true and mirrors every morning measurement by 180°. The
bug is subtle because it produces a plausible-looking table. Fixed in
`sun_reference.py`, with a comment stating why the sign must be preserved.

A second defect was mine, not the code's: retyping the calibration matrix by hand
dropped leading zeros (`1.42` for `0.00142`), which silently invalidated a run of
residuals. `sun_reference.py` now loads the matrix from the registry and never
retypes it.

## W1.6 — Next tasks, in order

1. **Build a real Sun-disc detector.** The disc is a ~6 px disc inside a much
   larger saturated region, with a sharp rim. Candidate approaches: fit the rim
   (the gradient magnitude peaks on the disc boundary); use the disc's known
   angular size; or use the temporal consistency of the disc centre, which must
   follow the ephemeris smoothly while cloud blobs do not.
2. **Re-run the infinity test with the disc centre.** Then the residual is a
   calibration measurement, and the per-frame scatter is its uncertainty.
3. **Break the `f` degeneracy** by combining the disc radius (bias-free) with the
   image circle, and by testing the three candidate projection laws
   (Kannala–Brandt equidistant, equisolid, stereographic) against the Sun's
   *trajectory* rather than single frames.
4. **Repeat on 2025-05-22** to measure session drift.
5. **Produce the margin statement:** for a stated height range, what calibration
   error is tolerable, what the fixed calibration delivers, and the resulting
   height uncertainty — the M1 go/no-go.

## W1.7 — Where this leaves the deliverable

Not yet settled, but the shape of the answer is visible. The geometry is
disparity-starved (18.3 px at 1.5 km) and the residual calibration uncertainty is
currently of the order of a few pixels, i.e. **tens of percent in height at
1.5 km**. Until the disc detector lands, the honest position is:

- **dynamics** — supportable, and the necessary condition;
- **absolute height** — carry a multiplicative uncertainty dominated by the
  un-pinned `f` (±15 % from W1.2 alone) and by the calibration residual.

That is a statement about the measurement, not about the method, and it is what
M1 exists to decide.

---

## W1.8 — Update: `f` estimated independently in four streams

`experiments/w1/f_identifiability.py` estimates the orientation offset `A0` and `f`
separately for each camera and each session, so the spread is a direct measure of
how well `f` is identified.

| stream | n | A0 (deg) | f (free intercept) | r0 (px) | rms (px) | f (r0 forced to 0) |
|---|---|---|---|---|---|---|
| 2025-05-22 dev1 | 35 | 313.99 | 605.4 | 57.1 | 8.3 | **665.3** |
| 2025-05-22 dev2 | 40 | 227.26 | 585.8 | 56.2 | 21.3 | **641.8** |
| 2025-05-23 dev1 | 28 | 312.60 | 633.4 | 27.9 | 48.2 | **663.1** |
| 2025-05-23 dev2 | 33 | 227.48 | 638.0 | 7.7 | 28.0 | **646.2** |

**Three results.**

1. **The rig's orientation is stable and the timebase is confirmed for both days.**
   `A0` reproduces to **1.4°** for dev1 and **0.2°** for dev2 across the two
   sessions, with mean azimuth residuals below 2°. A wrong clock, a wrong site or
   a moved camera cannot produce that. (The two cameras differ by ~86.6°, which is
   the ~90° relative mounting rotation the affine encodes.)

2. **The through-origin estimator is far more consistent across streams than the
   free-intercept one** — spread 3.5 % against 9 %. If the blob centroid carried a
   large constant radial bias, `r0` would be reproducible; instead it ranges from
   8 to 57 px, i.e. it is absorbing per-stream conditions rather than measuring a
   fixed instrument bias. The more consistent estimator is therefore the more
   credible one:

   > **f ≈ 642–665 px/rad, mean 654 ± 11 (1σ), i.e. a ratio of 1.070 ± 0.018 to
   > the assumed ξ₀/π = 611.2.**

3. **Consequence for height.** Since `D = f·B/Δξ`, assuming too small an `f`
   makes every height too low by `f_assumed/f_true`:

   | f true | heights biased | a 1500 m cloud reports as |
   |---|---|---|
   | 642 | 4.8 % low | 1428 m |
   | 654 | 6.6 % low | 1401 m |
   | 665 | 8.1 % low | 1379 m |

   This is a **purely multiplicative, cloud-height-independent** bias — it does not
   distort dynamics, which is consistent with the observation that the estimated
   series tracks the ceilometer's motion while sitting systematically below it.

**Open item raised by this run.** The camera-to-camera roll implied by the two
`A0` values is 86.6°, while the calibration's rotation block encodes 89.9° — a
**~3.3° discrepancy**, which at the crop corner (r = 452 px) would be a ~26 px
disparity error. That is far larger than the 4–11 px infinity residual measured on
the best frames, so at least one of the two interpretations has a sign or
handedness convention wrong. It must be resolved before either number is used:
it is the difference between "calibration is good to ~1 px" and "calibration is
~26 px wrong".

## W1.9 — Margin statement (draft, pending the open item above)

At 1500 m the disparity is ≈18 px (f = 611) or ≈19.6 px (f = 654). Height error
from a disparity error of r px is `r/Δξ · D`:

| disparity error | height error at 1500 m | at 3000 m |
|---|---|---|
| 0.5 px | 2.7 % (41 m) | 5.5 % (163 m) |
| 1 px | 5.5 % (82 m) | 10.9 % (327 m) |
| 2 px | 10.9 % (164 m) | 21.8 % (653 m) |
| 5 px | 27 % (410 m) | 54 % (1630 m) |

So a **10 % absolute-height target at 1.5–3 km requires the calibration to be good
to ≈1–2 px**, and the `f` uncertainty alone (±1.8 %, 1σ) already contributes
≈2 % — acceptable. The binding term is therefore the calibration residual, which
is currently bounded only as "≤ ~5 px on the best frames", not measured.

**M1 status: not yet decided.** Dynamically the estimator is supportable; the
absolute scale carries a known ~7 % multiplicative bias from `f` plus an
unmeasured calibration residual whose upper bound is of the same order as the
10 % target. The disc detector (W1.6 task 1) and the ~3.3° open item are what
stand between this and a verdict.

---

## W1.10 — The open item is not resolvable with the Sun-blob reference

The ~3.3° gap between the camera-to-camera roll implied by the two `A0` estimates
and the roll encoded in the calibration matrix was checked:

| quantity | value |
|---|---|
| affine 2×2 rotation | **89.92°** |
| `A0(dev1) − A0(dev2)`, two-day means | **85.93°** |
| gap | **3.99°** |
| displacement this implies at r = 200 / 320 / 452 px | 13.9 / 22.3 / **31.5 px** |
| infinity residual actually measured on the best frames | **4–11 px** |

The two measurements contradict each other by a factor of about three. Each `A0`
comes from matching the *saturated blob centroid* to the ephemeris azimuth, and the
blob is not the Sun; the mean azimuth residuals were 0.62° (dev1) and 1.95° (dev2),
so each `A0` carries a blob-shape bias of order a degree or two — and the two
cameras view the same clouds from different geometry with image frames rotated
~90° apart, so their biases need not cancel. The gap is therefore larger than the
estimator's noise floor but of the same order as its **systematic** bias.

**Conclusion: the Sun-blob centroid cannot resolve calibration errors below roughly
4°, i.e. ~30 px at the crop corner.** W1's target is ~1 px. The reference is two
orders of magnitude too coarse, and no amount of cleverness with the *same*
estimator will fix that — the saturated region is not the disc.

This makes the Sun-disc detector a **hard requirement**, not a refinement. Two ways
forward, both falsifiable:

**(a) Recover the disc inside the saturated region.** The disc is ~6 px across
(0.53°). Approaches: fit the rim of the saturated area and use the known angular
size; use the radial profile's inflection; or exploit that the disc centre follows
the ephemeris *smoothly and predictably* while cloud blobs do not — a temporal
consistency filter across a day's frames, which the `A0` stability result (0.2–1.4°
across sessions) shows should work.

**(b) Abandon the infinity reference and estimate the relative pose from the
images themselves.** This is what the comparable published work does (Crispel &
Roberts 2018): take correspondences, estimate the essential matrix, decompose it,
and reject solutions inconsistent with the known rig geometry (`R ≈ I`,
`t ≈ (b,0,0)` — the project's baseline is 45 m, known). For a *single stereo pair*
the scene is instantaneously rigid, so the essential matrix is well defined even
though clouds move over time. Two independent correspondence sets are available:
the cloud field inside the crop, and the **static ground and buildings** at the
periphery, which are rigid by construction and therefore the more trustworthy of
the two.

Route (b) has a large advantage: it does not depend on any object being at
infinity, it uses the data the runtime pipeline already has, and it yields the
relative *rotation* directly, which is the quantity that limits the crop corners.
It is the recommended next step, with (a) as a parallel attempt since a working
disc detector also gives an absolute orientation reference and a clock check.

## W1.11 — Status after this pass

Established:

- the timebase is UTC, verified by two independent methods;
- the rig's orientation is stable to 0.2–1.4° across sessions;
- **f ≈ 654 ± 11 px/rad (ratio 1.070 ± 0.018 to the assumed ξ₀/π)**, which implies
  all heights are currently **~6–7 % too low**, multiplicatively and independently
  of height — so dynamics are unaffected and absolute level is shifted;
- the Sun-blob reference is too coarse for the calibration question by ~30×.

Not established:

- the calibration residual (the quantity W1 must produce), because no reference
  yet has the accuracy to measure it;
- therefore the M1 go/no-go, though the margin arithmetic is written down
  (§W1.9): 10 % absolute height at 1.5–3 km needs calibration good to ~1–2 px.

---

## W1.12 — The calibration residual is measured: a ~1.6° roll

`experiments/w1/epipolar_check.py` fits the disparity field of every frame with the
exact geometry plus the calibration terms, using **no Sun and no ground truth** —
route (b) of §W1.10. Model: a plane at height D, an additional roll of camera 2
about its optical axis, and a translation. For any (D, roll) the optimal
translation is the median residual, so only two parameters are searched.

Over 38 frames of 2025-05-23 (09:01–10:19 UTC):

| model | median rms residual |
|---|---|
| D only | 13.18 px |
| D + translation | 8.46 px |
| **D + translation + roll** | **2.64 px** |

and the fitted roll is **−1.57° ± 0.10°** — stable to a tenth of a degree over an
hour. Fitting without translation freedom gives **−1.45°**, so the roll is not
absorbing a translation. This is an order of magnitude tighter than the Sun-blob
estimate and it **resolves the W1.10 open item**: the true roll is ~1.6°, not the
~4° the blob azimuths suggested. The blob-shape bias was indeed the culprit, as
suspected there.

### Why this matters more than a scale error

A roll about the frame centre displaces points **tangentially**, so its horizontal
component is `δ·Δv` and it **changes sign across the centre**:

| radius | displacement |
|---|---|
| 160 px | 4.4 px |
| 320 px | 8.8 px |
| 452 px (crop corner) | 12.4 px |

That is a swing of **±8.8 px in the horizontal disparity across the 640 crop**,
against a total disparity of 18.3 px for a 1500 m cloud:

| cloud height | true disparity | disparity range across the crop | implied heights |
|---|---|---|---|
| 1000 m | 27.5 px | 18.7 – 36.3 px | 758 – 1470 m |
| 1500 m | 18.3 px | 9.5 – 27.1 px | 1014 – 2883 m |
| 3000 m | 9.2 px | 0.4 – 18.0 px | 1531 – very large |

**This is the root cause of the fragmentation documented in the stage audit (§4).**
The audit found `min_cluster_size=3` producing 478 clusters on one frame and a
median cluster size of 5 across the run, and concluded that the 300 m height-merge
was doing the real layer separation. The reason is now visible: a large part of the
within-frame disparity **spread** is the calibration roll, not cloud structure. The
clustering stage is attempting to segment an artefact.

Two consequences that change the plan:

1. **W4 cannot succeed before W1.** Clustering the raw disparity field is clustering
   calibration error. The roll term must be removed (or carried as a known nuisance
   parameter) before any layer extraction is meaningful.
2. **The median is largely protected.** Because the roll's contribution changes sign
   across the centre, the median disparity is roughly unbiased — which is consistent
   with the pipeline's heights being plausible in level while the spread is far
   wider than the cloud justifies. It also means the roll biases a *non-symmetric*
   cloud (one occupying one side of the field) and does not bias a centred one.

### A degeneracy to respect

When translation is free, the fitted D runs to the **search boundary** (6250 m) for
most frames: over a 640 px field, a plane at large distance and a translation are
nearly the same displacement, so D is not separable from the translation. The roll
is stable across both fits, so the **roll is measurable while D is not** by this
method. The scene-based fit must therefore not be used to refine heights; it is a
calibration instrument only.

### Status of the finding

Measured twice by independent methods and agreeing in magnitude: the scene-based
fit gives 1.45–1.57°, and the infinity test's 4–11 px residuals at r ≈ 400–476 px
correspond to 0.5–1.6°. **Falsifiable prediction:** if the roll is real, applying
−1.57° about the principal point to camera 2 before computing disparities should
collapse the within-frame disparity spread and remove the need for the 300 m merge.
If the spread does not collapse, the roll is absorbing something else.

---

## W1.13 — The prediction was tested and confirmed

§W1.12 predicted that if the roll is real, correcting it should collapse the
within-frame disparity spread. Tested on the cached correspondences of 38 frames
by rotating camera 2 about the principal point and re-measuring the horizontal
disparity, ground truth untouched:

| applied rotation | median du | **IQR of du** | std of du | implied height range for a 1500 m cloud |
|---|---|---|---|---|
| 0.0° (as shipped) | 14.94 px | **8.14 px** | 5.82 | 1446 – 2530 m |
| −1.5° (wrong sign) | 15.75 px | 15.64 px | 9.94 | 1167 – 3467 m |
| +1.0° | 14.35 px | 3.56 px | 3.57 | 1705 – 2188 m |
| **+1.5°** | 14.22 px | **2.11 px** | 2.98 | **1800 – 2088 m** |
| +2.0° | 13.83 px | 2.43 px | 3.08 | 1828 – 2179 m |
| +3.0° | 13.15 px | 6.70 px | 4.91 | 1667 – 2807 m |

The wrong sign **doubles** the spread; the right sign reduces it **3.9×**; the
optimum is sharp and sits at **+1.5°**, matching the independently fitted 1.57° in
magnitude (the sign convention differs because the fit applied the rotation to the
predicted position rather than to the observation).

Restricted to the inner core (r < 200 px) the same result holds: IQR 4.15 px → 1.50 px.

**The median barely moves** (14.94 → 14.22 px, 4.8 %), exactly as predicted, because
the roll's contribution changes sign across the centre. So:

- **level** — affected only mildly, which is why the pipeline's heights looked
  plausible;
- **spread** — inflated ~4×, which is why the clustering stage fragments.

### The roll is now corroborated three independent ways

1. scene-based field fit: **1.57° ± 0.10°** across 38 frames;
2. spread-collapse optimum: **+1.5°**, sharp;
3. infinity test residuals of 4–11 px at r ≈ 400–476 px ⇒ 0.5–1.6°.

Three methods, one answer. The correction is derived from an **internal** criterion
(minimising disparity spread, on the physical expectation that a cloud layer has
small intrinsic spread) and never from the ceilometer, so guardrail G1 holds. It
must nevertheless be declared as a *measured correction applied on top of the fixed
calibration*, not as a new calibration — the user's constraint is that the
calibration cannot be changed, and this does not change it, it compensates a
measured residual.

### What this changes downstream

- **W4 is unblocked but must consume the corrected disparity.** After correction the
  residual IQR is 2.11 px, so the remaining spread is candidate real structure plus
  match noise rather than calibration error. The 300 m merge should be re-examined
  against the corrected field: if it was compensating the roll, its justification is
  gone.
- **W3's uncertainty budget changes.** The dominant within-frame error was
  calibration, not the matcher. Matcher localisation must now be characterised
  against a ~2 px floor, not an 8 px one.
- **The height range for a representative cloud tightens from ±27 % to ±7 %**, which
  is a genuine improvement in the product and not a metric-fitting exercise.

---

## W1.14 — The second session cannot answer the drift question at all

Attempting to repeat the roll measurement on 2025-05-22 produced **zero usable
frames**. The cause is not a bug: **that session's sky is essentially cloud-free**,
so LoFTR has no texture to match. The direct measure — actual correspondences,
which require cloud structure — by hour:

| hour UTC | 2025-05-22 | 2025-05-23 |
|---|---|---|
| 08:00 | 12 | **1104** |
| 09:00 | 13 | **1867** |
| 10:00 | 2 | **3002** |
| 11:00 | 2 | 183 (marginal) |
| 12:00 | 4 | **3719** |
| 13:00 | 1 | **1088** |
| 14:00 | 14 | **1067** |
| 15:00 | 12 | **2991** |
| 16:00 | 0 | **1442** |
| 17:00 | **1919** | **1553** |

Visual confirmation: the 2025-05-22 frames in the analysis window show a smooth
cloudless dome, the 2025-05-23 frames show cumulus fields
(`experiments/audit/session_compare.png`). This is corroborated independently by
the ceilometer: 250522 has 610 valid readings for the day against 2341 for 250523,
which is what a clear day looks like.

**Consequences, all of which change the plan:**

1. **Session drift is not measurable.** W1 asked for it across 2025-05-22/23. There
   is one usable session, so the answer is "not measurable from this data", with
   the reason demonstrated rather than assumed. The roll of −1.57° is therefore
   measured on 2025-05-23 only; nothing here shows it is or is not stable across
   days.
2. **The project has one session, not two.** 2025-05-23 is usable from 08:00 to
   17:00 UTC (11:00–20:00 MSK); the paper's window (09:00–14:00 UTC) sits inside it.
   Every result in this project comes from that window.
3. **Dataset splits must be temporal, within 2025-05-23.** The plan's W0 assumed
   splitting by session. That is impossible. Tuning and validation must be separated
   in time inside one day, which weakens the independence of any held-out validation
   and must be stated as a limitation rather than glossed (guardrails G2, G3).
4. **Runtime validation on an unseen session is not possible with this data.** The
   plan's W6 falsification test ("run on a session it has never seen") cannot be
   satisfied here. It has to be replaced by a held-out time block plus the internal
   consistency checks, and the limitation recorded.
5. Any historical figure attributed to 2025-05-22 stereo should be treated as
   suspect until re-derived.

### The 2025-05-23 roll, restated cleanly

| estimator | value |
|---|---|
| model fit, per-frame median (n = 38) | **−1.57° ± 0.10°** |
| spread optimum | **+1.69°** (opposite sign by model convention, same magnitude) |
| disparity IQR, as shipped → corrected | **8.14 px → 1.92 px (4.2× tighter)** |
| residual rms, full model | 2.64 px |

The sign convention of the two estimators differs because the model applies the
rotation to the predicted position while the spread test applies it to the
observation; the magnitudes agree to 0.12°, which is the meaningful comparison.

---

## W1.15 — Correction: earlier correlation figures in this document are withdrawn

Findings reported here and in [../report/findings.html](../report/findings.html) from
the project's `windowed_correlation` metric are **withdrawn**, following a null test
documented in [temporal-alignment.md](temporal-alignment.md).

Rolling the algorithm series circularly against a fixed ceilometer — preserving both
autocorrelations and destroying only their relative timing — and running the *same*
±300 s scan gives a **null** mean scan-max r of **0.346** raw / **0.421** filtered,
against observed values of 0.189 and 0.209. The observation is below what random
re-alignment produces. A 30-minute window holds ~152 paired samples but the
ceilometer's lag-1 autocorrelation is 0.906, leaving **~3.4 effective samples**;
maximising over 121 shifts of three points is selection.

Specifically withdrawn:

- any statement that the corrected pipeline agrees or disagrees with the ceilometer
  at a stated r;
- the claim that separating the ceilometer's layers triples the correlation
  (§11 of the report). Repeating the null *within each regime* gives p = 0.992 (all),
  **0.758 (lower layer)**, 0.067 (upper layer) — the improvement is not demonstrated;
- the "±300 s boundary in ~45 % of windows" figure, which is 15–25 % (max 30 %
  across stored runs).

What survives is unaffected, because none of it used this metric: the measured focal
constant, the measured roll, the lens distortion, the ceilometer's bi-modal layer
structure, and the reliability conclusions about the estimator's internal geometry.

**Consequence for the plan:** the project currently has **no demonstrated external
agreement** with the ceilometer. Every W-package validation step must be rebuilt on
a criterion with real statistical power — paired samples that are not 90 %
autocorrelated, a spatial rather than point comparison, or an internal criterion.
Quoting a windowed-r range as evidence is now prohibited by
[guardrails.md](guardrails.md) G11 and this document.

---

## W1.16 — Second correction: the dynamics were not anti-correlated

W1.15 withdrew the windowed correlations. The properly-powered replacement (§3 of
[evaluation.md](evaluation.md)) initially reported that the estimate's frame-to-frame
changes were **anti-correlated** with the ceilometer at r = −0.248 (n = 469). **That
was a protocol artefact** and is withdrawn too.

The differences had been taken on a *masked* 30 s grid, and 30 of the 469 spanned a
missing epoch, silently pairing samples minutes apart. On strictly adjacent pairs:

| comparison | n | r | 95 % CI |
|---|---|---|---|
| level, whole session | 470 | **−0.376** | [−0.653, −0.262] |
| differences, gap-crossing (flawed) | 469 | −0.248 | — |
| **differences, strictly adjacent** | 439 | **−0.045** | [−0.100, −0.007] |

**The failure is in LEVEL, not in dynamics.** The estimate is offset and inverted
against the reference; it is not moving in opposition to it. The corrected picture,
and the measurements that reframe the problem as one of *target identity*
(the estimate is flat while the reference swings; where it responds the sign is
inverted; the flatness is in the height field, not the selection rule; the dominant
error is a 1000–1400 m within-frame scatter) is in
[evaluation.md](evaluation.md) §8 and [research-workplan.md](research-workplan.md) R1.
