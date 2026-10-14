# Briefing for supervisor consultation

**Project:** passive stereo cloud-base-height (CBH) retrieval from a pair of ground-based
fisheye cameras
**Status:** instrument characterised; retrieval **not** demonstrated, and the reason is
now measured
**Prepared:** 2026-10-14

This document is self-contained. It states the task, the data, the instruments, the
algorithm, the calibration, the processing, the validation method, everything that is
established, everything that is not, and the specific questions we need help with. Every
number is traceable to a script and an artefact in the repository; nothing is recalled.

---

## 1. The task

### 1.1 What we are trying to produce

A cloud-base-height estimate from **two photographs taken simultaneously by two fisheye
cameras 45 m apart**, with no active instrument.

### 1.2 The runtime contract

At inference time only the following exist:

| available | not available |
|---|---|
| two synchronised fisheye photographs | ceilometer |
| the camera calibration | meteorology |
| a timestamp per photograph | any historical statistic |
| camera location and pose | any parameter fitted against ground truth |

Everything in the delivered algorithm must therefore be a function of
`(image pair, calibration, time, location)` alone. Anything requiring the ceilometer is a
research instrument for validation only, and is labelled as such throughout.

### 1.3 The physical quantity and its known ambiguity

The stereo measures the **geometric** lower boundary of the cloud: where the two lines of
sight to the same cloud feature intersect.

The reference instrument — a DOL-2 laser ceilometer — measures an **optical** quantity:
the range at which backscattered light crosses a detection threshold. That threshold is
reached *inside* the cloud, not at its lower boundary, so the ceilometer is expected to
report a **higher** value than the geometric base. The published literature attributes
residual camera-versus-ceilometer offsets to unmodelled three-dimensional cloud-side
geometry for the same reason.

**Consequence for the whole project:** we do not expect the two instruments to agree, and
we must not tune the algorithm to minimise the difference. The goal is (a) that the
estimate moves when the cloud moves, and (b) that any persistent offset is *explained*
by a physical mechanism rather than absorbed into a correction.

---

## 2. Site and instruments

### 2.1 Cameras

| property | value | how known |
|---|---|---|
| projection | fisheye, nominal 180° field | config; **not verified against a datasheet — the lens model is unknown** |
| frame size | 1920 × 1920 | measured from the files |
| cadence | median 10 s, min 1 s, max 16 s | measured |
| baseline | 45 m | configured; stated in the written work |
| mounting | two units, image frames rotated ~90° relative to each other | measured from the calibration |
| EXIF | **none** — no timestamp, no make/model | measured |

The absence of EXIF matters: filename and file mtime are the only time evidence, and they
disagree by a constant 3 hours.

### 2.2 Reference ceilometer

| property | value | how known |
|---|---|---|
| type | DOL-2 laser ceilometer | written work |
| slot cadence | 15 s (240 lines/hour) | measured |
| reported fields | `H1`, `H2`, `L`, `Q` | log format |
| layers | up to two (`H1`, `H2`); 9999 = no detection | log format |
| stamp clock | **MSK**, labelled on every line | log format |
| valid rate | 3 to 240 detections per hour | measured |
| stated range in the work | 10–3000 m | written work |
| **measured range** | `H1` reaches **7010 m** | measured — **the stated range is contradicted by the data, unresolved** |

The field `L` (backscatter width), which would characterise layer depth and could support
a better reference, is **populated in 2 of 5759 slots** and never in the validation window.

### 2.3 Auxiliary instruments — present but not usable

`meteo1/`, `meteo2/` (surface meteorology), `mtp/` (microwave temperature profiler),
`rvp/` (water-vapour radiometer) are all in the data root. **They cover 2023–2024 only.**
There is no meteorology, and in particular **no wind measurement**, for either image
session. No site coordinates are recorded anywhere in the project.

---

## 3. The data

### 3.1 Coverage

| date | frames | unique timestamps | first | last |
|---|---|---|---|---|
| 2025-05-22 | 17,264 | 9,091 | 00:00:03 | 23:59:54 |
| 2025-05-23 | 17,265 | 9,041 | 00:00:04 | 23:59:56 |

Total 34,529 JPEGs, 9.6 GB. 18,132 unique timestamps; 16,397 pair by exact filename;
pairing by nearest timestamp within 2 s recovers the rest. Median cadence 10 s, **no gap
above 20 s on either day**. One frame is corrupt JPEG around 2025-05-23T12:50.

### 3.2 The clock offset — established twice, independently

Image filenames are **UTC**; the ceilometer log is **MSK**; the offset is exactly **+3 h**.
Two independent confirmations:

1. **Diurnal brightness envelope.** The 2025-05-23 image sequence brightens from 00:45 to
   18:15 in filename time, which corresponds to sunrise/sunset of ~04:00/21:00 MSK.
2. **Solar ephemeris.** A solar position algorithm matched against the detected Sun's
   image azimuth reproduces the observed position to better than **0.5° across 14 hours**
   with a single constant orientation offset. An ephemeris cannot match a clock that is
   three hours wrong.

The filename/mtime disagreement is exactly 3 h in every sampled frame, so `mtime` is MSK.

### 3.3 Usability — only ONE session has cloud

This is the single most limiting fact in the project.

| hour UTC | 2025-05-22 matches/frame | 2025-05-23 matches/frame |
|---|---|---|
| 09:00 | 13 | 1867 |
| 12:00 | 4 | 3719 |
| 15:00 | 12 | 2991 |
| 17:00 | 1919 | 1553 |

**2025-05-22 is cloudless** and yields single-digit correspondences at every hour except
17:00. Visually confirmed (a smooth cloudless dome), and corroborated independently by the
ceilometer: 610 valid readings that day against 2,341 on 2025-05-23. Stereo needs texture;
a clear sky has none.

Consequences, which constrain every validation design:

- **Session-level hold-out is impossible.** There is no second session.
- **Any "independent period" is a time block of the same day**, sharing the same
  calibration, the same instrument state, and the same weather regime.
- **Validation on an unseen session cannot be performed with this data.**
- 2025-05-22 is nevertheless valuable for *calibration* (see §6.4) and as a **clear-sky
  control** for matcher behaviour.

### 3.4 The validation window

The written work's window is 12:00–17:00 **MSK** = 09:00–14:00 UTC, giving **1,799 stereo
pairs**, five hours at 10 s. That window sits inside 2025-05-23's usable period
(08:00–17:00 UTC).

---

## 4. The measurement principle

### 4.1 Geometry

Two cameras with a horizontal baseline `B = 45 m`. A cloud feature at height `D` projects
to image positions `p₁` and `p₂` in the two rectified frames. Its **disparity** is the
difference of those positions. For a zenith-pointing fisheye with an **equidistant**
projection, image radius is proportional to the angle from the optical axis:

```
r = f · θ ,     f = ξ₀ / π = 1920 / π = 611.15 px/rad = 10.67 px/deg
```

The pipeline's height relation is the paraxial (small-angle) form:

```
D = f · B / |Δξ|
```

The exact relation triangulates the two viewing rays and is what we now use (§7.3).

### 4.2 The accuracy budget — this number governs everything

| true height | disparity | m per pixel | **% of height per pixel** |
|---|---|---|---|
| 500 m | 55.0 px | 9.1 | 1.8 % |
| 1000 m | 27.5 px | 36.4 | 3.6 % |
| **1500 m** | **18.3 px** | **81.8** | **5.5 %** |
| 3000 m | 9.2 px | 327.2 | 10.9 % |
| 5000 m | 5.5 px | 909.0 | 18.2 % |

Height error is **relative**: `σ_D/D = σ_Δξ/Δξ`. So:

- **1 px of disparity error is 5.5 % of the height at 1.5 km and 10.9 % at 3 km.**
- Reaching 10 % accuracy at 1.5–3 km needs the disparity good to **~1 px**.
- The written work's "≈30 m per pixel" is true only near 550 m; at 1.5 km it is 81.8 m
  and at 3 km, 327 m — **the stated sensitivity understates the problem ~3×** at typical
  heights.

### 4.3 The rig is disparity-starved by construction

At the same cloud height, comparable published rigs produce far more disparity:

| rig | baseline | disparity at 1500 m |
|---|---|---|
| **this project** | **45 m** | **18.3 px** |
| Crispel & Roberts 2018 (AMT) | 150 m | ≈94 px |
| Beekmans et al. 2016 (ACP) | 300 m | ≈187 px |

With 5–10× less disparity, the same angular error costs 5–10× more height error. This is a
property of the hardware and bounds what any algorithm can deliver.

---

## 5. Statistical power — a constraint discovered the hard way

Two effective-sample-size facts must be stated before any comparison:

| series | nominal n | lag-1 ρ | **effective n** | decorrelation time |
|---|---|---|---|---|
| estimate, raw per-frame | 1784 | 0.34 | **881** | ~20 s |
| estimate, Kalman-filtered | 1784 | **0.98** | **18.8** | ~949 s |
| ceilometer `H1` | 905 | 0.89 | **51** | ~267 s |

**Any comparison is capped by the slower series.** The filtered output has ~19 independent
samples over five hours, so the smallest detectable correlation against it is ~0.49. The
Kalman filter destroys the very variability a validation depends on.

Consequence: **validation must target the per-frame series or the layers, not the smoothed
output.**

---

## 6. Calibration

### 6.1 What it is

A single **2 × 3 affine matrix** mapping camera 1's image coordinates into camera 2's
frame. It is fixed, is applied to both sessions, and may not be re-derived from
calibration imagery (none exists).

```
[[  0.0014207881638163794,  -0.9999989906799874,  1972.73673196935 ]
 [  0.9999989906799874,      0.0014207881638163794,     3.6708577729732 ]]

rotation 89.919°, translation (1972.74,  3.67) px, scale 1.000000
```

Its provenance is **inferred, not verified**: the registry records
`calibrated_on_verified: false`, method "sun position; `cv2.estimateAffine2D`, RANSAC
initialisation", the date inferred from the filename. There is **no calibration for
2025-05-23**; the 2025-05-22 matrix is applied to it.

### 6.2 What a 2-D affine cannot represent — a structural limitation

An affine warp can express **rotation about the optical axis** and **translation**. It
**cannot** express a relative **tilt** between the two cameras (pitch or yaw), nor a
difference in focal length, nor lens distortion.

After such a warp the pair is therefore **not genuinely rectified** away from the
principal point: the epipolar lines are not horizontal, and the disparity acquires a
vertical component the geometry does not predict. Measured vertical disparity is a
near-constant **~2.2 px** across the field — the signature of a small relative tilt, and
not expressible by the matrix above.

### 6.3 What has been measured about it

**a. A residual roll of ≈ −1.6°.** The disparity field of 38 frames fitted with the exact
geometry plus a roll and a translation gives **−1.57° ± 0.10°**; the roll that minimises
the disparity spread gives **+1.69°** (opposite sign by model convention, same
magnitude); and the infinity-test residuals are consistent. Removing it collapses the
within-frame disparity IQR from **8.14 px to 1.92 px** (4.2×) and, through the pipeline,
the per-frame height spread from **689 m to 341 m**.

**b. The focal constant is ~654–686 px/rad, not the assumed 611.** Five independent
estimates:

| method | f (px/rad) |
|---|---|
| Sun radius vs zenith, four streams (through-origin) | 642, 646, 663, 665 |
| Sun metric on the clear day (R9) | 683 |
| Sun metric on the cloudy day (R9) | 663 |
| image-circle radius, *if* the lens spans 180° | 686–736 |

The assumed `ξ₀/π = 611` presumes the 180° field maps onto the circle inscribed in the
frame. **The measured circle radius is 1078–1156 px, not 960.** The edge is soft, so this
is a bound rather than a precise number, but every method agrees that the assumed value is
too small.

**Consequence:** heights are currently **~7–10 % low**, purely multiplicatively and
independently of height. This does not distort dynamics; it shifts the level.

**c. A cubic lens distortion, a = −0.30.** With `θ = (r/f)(1 + a(r/f)²)`, fitting `a` to
the requirement that a flat cloud layer gives a height independent of image radius gives a
clean minimum at **a = −0.30**, taking the radial spread from **36.7 % to 3.2 %**; stable
against `f` (−0.30 at both 611 and 654). At the crop corner this is ~16 % angular
compression. Confirmed end-to-end through the pipeline: radial profile IQR **19.1 % →
2.4 %**, moving the median only −2.0 %.

### 6.4 A missed calibration opportunity, and what came of it

The Sun is the only object at infinity in these images, so its two images must coincide
after rectification. The first attempt on 2025-05-23 failed: thresholding the brightest
pixels finds the whole saturated region, whose centroid is dragged by bright cloud beside
it, leaving the method at ~4° resolution against a 1 px target.

**2025-05-22 is cloudless**, so the same test was run there:

| | clear (05-22) | cloudy (05-23) |
|---|---|---|
| median saturated blob | **14,970 px** | 6,266 px |
| azimuth residual p90 | **1.34°** | 4.69° |
| radial fit rms | 74.7 px | 67.5 px |

The **orientation** measurement is 3.5× tighter on the clear day — the hypothesis was
right about the mechanism. But the blob is **2.4× larger**, not smaller: with no clouds to
occlude the sky, the glow saturates a *wider* area, so the **radial** estimate is no
better. The two coordinates have different quality, which the first attempt treated as one
number.

### 6.5 A fit that looked brilliant and was wrong — recorded as a warning

Before finding the distortion, a **relative camera rotation** was fitted to the same
flatness criterion and appeared to succeed spectacularly: pitch +0.99°, yaw +0.82°,
spread **36.7 % → 2.5 %**. It is **wrong**. That rotation predicts **13.7 px** of vertical
disparity from the tilt alone, while the measured vertical disparity is **2.24 px**. It
was exploiting the degeneracy between rotation and height scale. Bounded to what the
vertical disparity allows, the trend is unchanged at 33.8–39.8 %.

**Rule derived:** a fit whose parameters are not independently constrained by a second
observable is not a measurement.

---

## 7. The algorithm, stage by stage

Entry point `tools/evaluate.py --config <cfg>` → `clouds.pipeline.evaluate`.

### 7.1 Pre-processing

1. **Load** two JPEGs at 1920 × 1920 (~25 ms each).
2. **Rectify camera 1** by the affine matrix (`cv2.warpAffine`).
3. **Apply the measured roll** to camera 2 about the principal point (configurable;
   1.5° in the working configuration).
4. **Centre-crop both to 640 × 640** (`target_size`).
5. **Convert to grayscale.**

**Known defect at this stage:** the crop is a **square**, whose corners reach 452 px from
its centre, into an annulus that contains near-field trees, a building and a mast. The
natural field of a circular fisheye image is a circle; a square crop includes objects that
cannot be cloud. No occluder mask is applied.

### 7.2 Matching

Current default **RoMa** (`romatch`), dense, with `grid_stride: 8` and
`certainty_threshold: 0.05`, producing ~1,200–5,600 correspondences per frame.

Ten backends are registered behind one factory contract and selectable by config
(`tools/matchers.py list` / `check`). The selection was made on a pre-registered,
ground-truth-free protocol (§9.4).

### 7.3 Height estimation

Two paths exist:

- **Paraxial** (shipped): `D = f·B/Δξ` applied to a cluster's mean horizontal disparity.
  Biased off-axis — up to **+11.7 %** at the crop corner for a 1 km cloud — and it cannot
  express lens distortion at all.
- **Exact** (now default): per-match unit-sphere triangulation,
  `clouds.geometry.triangulation.heights_from_matches`, with the distortion coefficient,
  returning per-match heights and the ray-intersection residual as a free quality signal.

### 7.4 Clustering

`sklearn.cluster.HDBSCAN` on the 2-D disparity vectors with `min_cluster_size: 3`.

**Known defects:**

- At `min_cluster_size = 3`, one frame yields **478 clusters**; at 5, ten; at 50, one.
  Median cluster size across the run is **5**. The stage shreds the match set.
- `cluster_selection_epsilon > 0` **crashes** sklearn 1.9 HDBSCAN on this data
  (`TypeError: only 0-dimensional arrays...`); the knob has been removed and the silent
  all-noise fallback deleted.
- A **300 m height merge** then collapses ~232 fragments per frame into ~7 groups. It is
  that merge, an undocumented workaround, that performs the actual layer separation.

### 7.5 Temporal processing

A 1-D innovation-adaptive **Kalman filter** on the selected cluster's height, with an RTS
smoother, a Reliability Index and a TQI. `mode: simple` selects one cluster per frame by
the `lowest` rule.

**The filter is currently unwired** (`temporal.filter_enabled: false`) because it removes
**48 % of the raw variation** of a signal already ~4× less responsive than the reference,
and confounding the two makes both unreadable. It is a switch, not a deletion.

### 7.6 Selection

`cbh_strategy: lowest` — among clusters with size ≥ 8, the lowest height wins. Three
strategies exist (`largest`, `lowest`, `lowest_reliable`); **all three are flat over five
hours**, which is itself diagnostic (§8.3).

### 7.7 Post-processing and output

Per-frame series written to `series.json`; per-cluster dump optionally to
`clusters.jsonl`; a run manifest with hashes of config, calibration entry point and data,
plus a JSONL event log, written by `clouds.provenance`.

**Steps removed on measured grounds:** the spatial-coherence filter (rejected 37–43 % of
clusters while leaving the raw series **bit-identical** and moving the filtered series
0.3 %); `cluster_selection_epsilon` (crashes). Both documented in `docs/plan/pipeline-steps.md`.

**Steps kept but flagged:** the 300 m merge changes the tracked output by **249 m (16 %)**
on a single magic constant, and is doing smoothing the Kalman filter is also supposed to
do. It should be replaced by an explicit layer model, not tuned.

---

## 8. What has been established, and how

### 8.1 Claims that hold

| claim | how verified |
|---|---|
| Filenames are UTC, ceilometer MSK, +3 h | two independent methods (brightness envelope; solar ephemeris to <0.5° over 14 h) |
| Focal constant ~654–686 px/rad, not 611 | five independent estimates agreeing within a few percent |
| Residual roll ≈ −1.6° | three independent methods agreeing in magnitude |
| Cubic lens distortion a = −0.30 | fitted from the flatness requirement; stable against `f`; confirmed end-to-end through the pipeline |
| Ceilometer's lowest layer is **bi-modal** | 40 % of samples 2250–2750 m, 53 % 3750–4250 m, 3.8 % of steps >500 m |
| Correlations are cloud-driven | the clear day gives single-digit matches/hour against 1,100–3,700 |
| Matcher default is RoMa on principle | pre-registered, ground-truth-free protocol (§9.4) |
| Inference is 2.03× faster, byte-identical | 500-frame A/B; outputs `cmp`-identical |
| The reference matcher path reproduces exactly | 38/38 frames byte-identical, re-verified after every edit |

### 8.2 Claims that do NOT hold

- **Any windowed correlation figure.** Withdrawn — see §9.1.
- **The estimate tracks the reference in level.** Level correlation is negative and
  significant (r = −0.376, 95 % CI [−0.653, −0.262]).
- **The estimate tracks the reference in dynamics.** Gap-safe first-difference
  correlation is **nil** (−0.045, CI [−0.100, −0.007]).
- **The response has the correct sign.** Stratified by the reference's own regime, the
  estimate sits at **1944 m when the cloud is low and 1839 m when it is high** — a shift of
  **−105 m** (KS p = 3×10⁻²⁴), strengthening to **−154 m** in the inner field, with an
  identical match-radius distribution across regimes.
- **The estimate follows the cloud rather than the pixel.** See §8.3.
- **Advection explains the observed time shifts.** A fixed-baseline advection fit gives
  R² = 0.10 (p = 0.18).

### 8.3 The critical finding — the estimator is dominated by a STATIC spatial pattern

This is the most important measurement in the project.

**The test.** If the retrieval assigns a height to *cloud*, the height map must move as the
cloud moves. If it assigns a height to a *place in the image*, the map is identical all
session. The measured cloud motion (median 3.27 m/s at the estimated height, image azimuth
213° ± 9°) moves air through the field many times over in five hours, so a static map
cannot mean the cloud stood still.

**The result.** Correlation of the height map between the session's **first half** and its
**second half** — disjoint frames, 2.5 hours apart:

| configuration | r between halves | shuffled null | spatial spread | per-frame spread |
|---|---|---|---|---|
| roll correction only | **+0.981** | −0.002 | 173 m | 215 m |
| **roll + distortion** | **+0.912** | −0.002 | 92 m | 231 m |

**A large part of the estimated height is determined by WHERE in the image a match lies,
not by WHEN it was measured.**

**One mechanism explains every failure above.** A static map has a constant average, so
the delivered series is nearly flat (130 m against the reference's 1450 m). Every selection
rule samples the same static map, which is why all three are flat. The residual cloud
signal rides on a pattern that does not move, so its sign is meaningless. Adding matches
widens the pattern rather than adding signal.

### 8.4 The static field characterised — it is a scale error, not the distortion

The static part of the disparity field **is** the error field, because the cloud moves and
the error does not. Fitting it (167 frames, 406,289 matches) to candidate forms, as a
fraction of its own 0.84 px standard deviation:

| model | rms residual | explains |
|---|---|---|
| constant | 0.837 px | 0 % |
| **linear in u** | **0.486 px** | **41.9 %** |
| linear in v | 0.813 px | 2.9 % |
| radial (r) | 0.726 px | 13.3 % |
| radial (r², r³ combined) | 0.725 px | 13.4 % |
| roll | 0.817 px | 2.4 % |
| **radial + linear + roll** | **0.168 px** | **79.9 %** |

**The dominant error is a horizontal SCALE, not a radial distortion.** The linear-in-u
coefficient is **+0.00394 px per px** — a **+16.6 % change in disparity across the 640 px
crop**. At 1.5 km, where the entire disparity is 18.3 px, that moves the height by more
than 15 % from one side of the image to the other.

The distortion is real and **separate** — it is radial, and applying it already removed
about half the static pattern's amplitude.

### 8.5 An error that was compensating for another

Applying exact triangulation **without** the distortion made the radial trend **worse**,
not better: profile IQR went from 4.1 % (paraxial) to **19.1 %** (exact, ideal lens),
against 2.4 % with the distortion applied. The paraxial formula's off-axis bias and the
lens distortion act in opposite directions and partially cancel. **Two errors that cancel
are still two errors, and fixing either in isolation is a regression.**

### 8.6 The variance decomposition

| quantity | value |
|---|---|
| reported height (per-frame selection) | std **176 m**, p5–p95 574 m |
| reference `H1` over five hours | std **743 m**, p5–p95 1720 m |
| Kalman-filtered estimate | std **91 m** |

Two multiplicative causes: the **raw retrieval under-responds ~4×** (24 % of the
reference's variation), and the **filter halves what is left** (48 % reduction), leaving
**12 %**. Neither is visible from the delivered series alone.

With both geometry corrections applied, the within-frame spread falls **2.9×** (689 → 235 m)
and the level rises **31 %** (1441 → 1886 m), halving the gap to the lower reference layer.

---

## 9. Validation methodology

### 9.1 The original metric was measuring noise — and has been withdrawn

The project's windowed-correlation metric searched a time shift (±300 s) inside each
30-minute window and reported the **largest** correlation found. A null test — rolling the
estimate against a fixed reference and performing the **same** search — gives a **higher**
mean scan-max r (**0.346** raw, 0.421 filtered) than the observation (**0.189**, 0.209).
The observation is below chance. Cause: a 30-minute window holds ~152 paired samples, but
the reference's autocorrelation leaves **~3.4 independent ones**, and maximising over 121
shifts of three points is selection, not measurement.

**Every windowed correlation figure in the project is withdrawn**, as is the
"±300 s boundary in ~45 % of windows" figure (measured 15–25 %).

### 9.2 Two of my own claims were retracted

- **"The estimate's dynamics are anti-correlated" (r = −0.248).** A protocol artefact: 30
  of 469 differences spanned a missing epoch in the resampling grid. Gap-safe:
  **r = −0.045, CI [−0.100, −0.007]**. The failure is in **level**, not dynamics.
- **"Separating the reference's layers triples the correlation."** Fails the null within
  each regime (p = 0.758 for the lower layer). The layer *structure* stands; the
  *improvement* does not.

### 9.3 The evaluation design now in force

- **Effective sample size reported, never nominal.** The grid on which two series are
  compared must be spaced at the **larger** decorrelation time.
- **Every claim against a null** that destroys the claimed structure and preserves
  everything else — circular roll for timing, spatial shuffle for agreement.
- **No value produced by maximising over alternatives** without a null that maximises
  identically.
- **Synthetic ground truth validated by an independent exact estimator** before any method
  is judged by it (phase correlation confirmed the matcher test construction at ratio
  0.88–1.36).
- Pre-registration before the first number; a p-value, or an explicit statement that none
  exists; and the *direction* of any effect reported.

### 9.4 The matcher selection principle — a reusable methodological result

The first matcher evaluation ranked candidates by a **plane-fit residual** and put MASt3R
first. A direct test of **differential sensitivity at the working disparity** — can the
matcher resolve a 0.5 px change on a 20 px disparity — gives MASt3R a slope of
**0.000**: no response at all. **A consistently quantised disparity field satisfies a
plane fit as well as an accurate one.**

| matcher | differential slope (1.00 exact) | scatter | interior coverage | clear sky |
|---|---|---|---|---|
| **RoMa** | **1.008** | **0.016 px** | 0 % at cert 0.5, **dense grid; low-certainty interiors measured accurate to 0.68 px against a 7.02 px null** | **0 matches** |
| DKM | 0.994 | 0.014 px | 0 % at cert 0.5 | — |
| kornia LoFTR | 1.043 | 0.110 px | 13.6 % | 11 matches |
| EfficientLoFTR | 1.299 | 0.214 px | 13.3 % | 92 matches |
| **MASt3R** | **0.000** | 0.000 | 25.9 % | 5 matches |

The generalisable rules: **rank on the property the product consumes, measured directly;
stratify by the region that matters; validate synthetic ground truth independently;
distinguish a structural limit from a threshold setting; and use a null that can falsify.**

---

## 10. What remains unknown, and the specific questions

### 10.1 The gating unknown

Subtracting the exact geometry of a horizontal plane from the static field leaves a
residual **larger** than the field it was subtracted from (mean +1.88 px, std 1.57 px,
~11 % of the disparity), so the disagreement is **systematic**, not marginal. Three
explanations remain live and are **distinguishable**:

1. **A wrong principal point.** The assumed centre is the image centre; a small offset
   would produce a position-dependent disparity error of this order.
2. **Differing focal lengths between the two cameras.** The affine has scale exactly 1.0;
   two nominally identical lenses need not be. A relative scale difference is very close
   to the observed linear-in-u signature.
3. **A confound in which clouds occupy which image regions.** If low cloud
   preferentially occupies the image centre and high cloud the periphery, the per-cell
   session median would encode cloud climatology rather than instrument error.

**We have deliberately not fitted a correction**, because fitting one to a confounded field
would encode the confound.

### 10.2 Questions for the supervisor

1. **Is the principal-point / relative-focal-length separation worth pursuing first, or is
   there a standard target-free calibration we should be using instead?** The rig has no
   calibration targets, no site coordinates, and the only object at infinity is the Sun —
   whose saturated region is too broad radially (though its azimuth is good to 1.3° on the
   clear day). Star-field calibration is closed: the night frames are pure noise.

2. **How would you separate an instrument error field from a cloud-climatology confound
   using one session?** This is the crux. Our candidate is to exploit the two cameras'
   differential behaviour (a focal-length difference affects both images coherently and
   symmetrically; cloud climatology does not).

3. **Is the per-cloud route the right next step?** The owner proposes segmenting individual
   clouds and computing a height per cloud — either by post-hoc clustering of disparities,
   or by finding visual boundaries where a cloud has a distinct edge. The attraction is
   that the *distribution* of heights within one cloud becomes a test: a real cumulus base
   is not a plane, so its spatial spread should be physically plausible rather than
   near-zero. Our concern is that a diffuse cloud may not have a findable boundary, and
   that the static field would bias every cloud in the same direction.

4. **Is the ceilometer a valid reference for this comparison at all?** Its footprint is a
   ~10 m column against the camera's **1956 m** footprint at 1837 m (a ratio of ~200:1),
   and its `H1` switches between two layers in long stretches. It also reports up to
   7010 m against a stated 10–3000 m range. If the two instruments are measuring
   physically different objects, no correlation is the right expectation and the entire
   validation framing needs to change.

5. **Would a learned recalibration be defensible here?** One option raised is a few-shot
   post-processing network calibrated on one half of the day and validated on the other.
   Our concern is that with one session, a spatial field of unknown origin, and no
   independent period, a network would fit the confound. We would want your view on
   whether there is a validation design that makes this honest.

6. **What would you want checked first?** Our ordered candidates are: (a) the
   principal-point / focal-length separation; (b) a per-cloud height distribution analysis;
   (c) an occluder mask and a circular field mask; (d) a spatial-sampling model to decide
   whether a level comparison is defensible at all.

### 10.3 What the data cannot support — stated so it is not attempted

An independent session; session-level hold-out; unseen-session runtime validation;
meteorological or wind context for the session; site coordinates; the instrument
separation; a survey-grade reference; the ceilometer's backscatter width `L`; and any
absolute-height claim better than the focal constant's ~2 % (which is ~7 % in level).

---

## 11. Repository map for a reader

| path | contents |
|---|---|
| `docs/report/state.html` | one-page current state, readable in a browser |
| `docs/results/` | citable index of 17 results, the matcher-selection principle, 30+ negative results |
| `docs/plan/research-workplan.md` | R1–R12: hypotheses, discriminators, results where run |
| `docs/plan/evaluation.md` | how a result is allowed to be claimed |
| `docs/plan/w1-calibration.md` | the calibration measurements in full, including withdrawals |
| `docs/plan/temporal-alignment.md` | why the correlation metric was withdrawn |
| `docs/plan/pipeline-steps.md` | what was removed from the chain, and the measurement that removed it |
| `src/clouds/` | the pipeline: `geometry/`, `matching/`, `clustering/`, `temporal/`, `pipeline/` |
| `experiments/` | every measurement script and its JSON, grouped by question |
| `tools/matchers.py` | list and verify the ten registered matcher backends |

---

## 12. Summary in one paragraph

The camera system is now characterised to a few percent — a measured focal constant
~7–10 % above the assumed value, a measured residual roll of −1.6°, and a measured cubic
lens distortion that the pipeline had been ignoring — and applying the last two together
reduces the within-frame height spread 2.9× and raises the level by 31 %. The retrieval,
however, is **not** demonstrated, and the reason is no longer a mystery: **the estimated
height is dominated by a static, image-position-dependent pattern that correlates r = 0.98
between the two halves of the session while the cloud moves**, which is a calibration
signature rather than cloud. The dominant component of that pattern is a horizontal scale
error of about 16 % across the frame, distinct from the radial distortion. Until the
origin of that field is separated from a possible cloud-climatology confound, no
comparison against the ceilometer can be interpreted, and the honest position is that we
can say precisely what the instrument does and cannot yet say that the height it reports
is a cloud base.
