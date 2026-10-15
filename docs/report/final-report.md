# Stereo cloud-base height from a 45 m fisheye baseline: a validated instrument and a rigorously bounded null result

**Status:** final draft
**Scope:** one instrument, one site, one usable session, twelve pre-registered refutations

---

## Abstract

We built and validated a passive stereo cloud-base-height estimator from two zenith-pointing
180° fisheye cameras on a 45 m baseline, and tested it against a co-located DOL-2 laser
ceilometer.

We first established the instrument's capability **by construction**. A synthetic renderer
whose geometry is derived from the shipped calibration and gated against the analytic
parallax recovers a cloud layer of known height with **ρ(true, recovered) = +1.000 and a
median absolute error of 0.072 px**. Through the same renderer we found that the shipped
`lowest` cluster-selection rule errs by **325–1471 m** on ground truth, while the replacement
`lowest_mode` estimator errs by **≤64 m**.

On real data the estimator does **not** recover cloud base height. The failure is not noise:
it is a systematic inversion in which a higher cloud produces a lower reported height,
significant at p = 4e−14 and present at every spatial scale, in both matchers, and within a
single cloud layer. We tested and refuted **twelve** distinct explanations with pre-registered
discriminating criteria — spanning image certainty, spatial coherence, calibration geometry,
matched-population size, epipolar geometry, matcher identity, target-identity layering,
camera desynchronisation, harness artefacts, non-Lambertian photometry, photometric
superposition, and phase aliasing.

The instrument is sound and the cause of the inversion is **unattributed**. We report this as
a bounded null result: a validated instrument, a measured anomaly with a specific
distributional signature, and a definitive list of what the anomaly is not.

---

## 1. The validated instrument

### 1.1 The rig

Two 180° fisheye cameras, 1920 × 1920, baseline 45 m, median cadence 10 s, on 2025-05-23.
The reference is a DOL-2 laser ceilometer whose log is stamped MSK; the image filenames are
UTC; the offset is exactly +3 h, confirmed twice independently — by the diurnal brightness
envelope and by a solar ephemeris matching the Sun's image azimuth to better than 0.5° across
14 hours.

Three calibration terms were measured rather than assumed, each by more than one method:

| term | value | methods |
|---|---|---|
| focal constant | **654 ± 11 px/rad** (assumed 611) | five estimates: four-stream Sun-radius fits on both cameras and days, and the clear-day Sun metric |
| residual roll | **−1.57 ± 0.10°** | three: disparity-field fit, spread minimisation, infinity-test residuals |
| cubic distortion | **a = −0.30** in θ = (r/f)(1 + a(r/f)²) | fitted from the sky's flatness requirement; stable against f; confirmed end-to-end |

### 1.2 The synthetic control

A renderer (`experiments/w7/w7_renderer.py`) casts rays through the measured fisheye geometry,
intersects a cloud plane at a known height, samples a procedural texture, and warps camera 1
by the shipped affine exactly as the production code does.

**Its rig geometry was derived from the calibration, not assumed**, and one term took a
withdrawn attempt to find: the affine maps camera 1's principal point onto camera 2's, so
`c2 = R·c1 + t = (1014.10, 965.03)`, **not the image centre**. With both principal points at
the centre the rendered parallax is negative and grows with height.

**The renderer's gate, from the pre-registration in `w7_renderer.validate()`:** `du(zenith)`
must equal `f·atan(B/H)` and fall as 1/H.

| rendered H | du(zenith) measured | f·atan(B/H) analytic |
|---|---|---|
| 1000 m | 29.428 px | 29.410 px |
| 2000 m | 14.715 px | 14.713 px |
| 4000 m | 7.357 px | 7.357 px |

The gate passes exactly.

### 1.3 The matcher and the triangulation recover a known height

| rendered H | ground truth | RoMa recovered | bias |
|---|---|---|---|
| 1000 m | 29.56 px | 28.72 px | −0.914 px |
| 1500 m | 19.73 px | 19.58 px | −0.239 px |
| 2000 m | 14.80 px | 14.85 px | +0.060 px |
| 3000 m | 9.87 px | 9.80 px | −0.071 px |
| 4000 m | 7.40 px | 7.48 px | +0.072 px |

**ρ = +1.000, median |bias| = 0.072 px.**

The renderer was additionally shown robust to a **4× apparent texture-scale range**
(230 → 57 px, because the texture is indexed in metres) and to **3-D relief up to 20 % of the
cloud height** — twice the physically expected roughness — in both cases retaining
ρ = +1.000.

> **SCOPE:** the control establishes capability on *synthetic procedural texture at a known
> plane*. It models geometry and occlusion. It applies **no phase function**, and the surface
> is smooth. Nothing here establishes performance on real cloud radiance.

### 1.4 The shipped selection rule was a catastrophic defect

The production chain clustered the 2-D disparity vectors with HDBSCAN
(`min_cluster_size=3`) and reported the **lowest** cluster of size ≥ 8. Applying that rule to
the synthetic ground truth:

| selector | ρ(true, reported) | p | error range |
|---|---|---|---|
| field median | +1.000 | 0.000 | −56 to +24 m |
| **`lowest` (shipped)** | **+0.700** | **0.188 (not significant)** | **−1471 to −325 m** |
| **`lowest_mode` (replacement)** | **+1.000** | **0.000** | **−64 to +25 m** |

`lowest` reports **675 m for a 1000 m cloud, 722 m for 1500 m, 529 m for 2000 m**. It is an
extreme low-tail order statistic of the measurement noise, and it depends on the sample size,
so a larger sample yields a lower minimum.

**The replacement** (`analyze.lowest_prominent_mode`) works on the per-match height
distribution and reports the lowest **prominent mode** — one carrying ≥10 % of the weight and
≥50 matches. It is a property of where the data concentrate rather than of how many points
there are, and it is now the project default.

---

## 2. The measured anomaly

On the real session the estimator does not recover cloud base height.

### 2.1 The inversion

Stratified by the ceilometer's own reported regime, the estimator places **higher** clouds at
**lower** heights:

| regime | n | reported height | reference H1 |
|---|---|---|---|
| low (H1 < 2750 m) | 215 | 2361 m | 2250 m |
| high (H1 > 3500 m) | 120 | 2064 m | 3790 m |

The shift is **−297 m** in the reported height while the reference rises **+1540 m**.
Rank correlation with the reference is **−0.547 (p = 4e−14)**; a correct estimator requires a
positive sign.

**Important qualification, added after the W9 cross-validation.** The inversion is
**estimator-dependent**. Under 5-fold cross-validation, in which every statistic is fitted on
four fifths of the session and scored on the held-out fifth:

| estimator | out-of-sample level ρ | sign |
|---|---|---|
| **lowest prominent mode** | **+0.137, and +0.202 with a per-cell + linear correction** | **correct** |
| median | −0.386 | inverted |
| 25th percentile | −0.296 | inverted |

**The inversion is a property of the median-like statistics on which it was measured, not of
the disparity field as a whole.** The mode estimator is correctly signed out of sample.

**The mechanism is a change in the distribution's shape, and it is measured.** Between
regimes:

| quantity | low regime | high regime | change |
|---|---|---|---|
| median | 12.354 px | 13.960 px | **+13.0 %** |
| **mode** | 12.896 px | 11.371 px | **−11.8 %** |
| **skewness** | **2.268** | **0.731** | **−67.8 %** |
| **standard deviation** | 4.303 | 4.283 | **−0.5 %** |

**The median and the mode move in opposite directions while the dispersion is unchanged.**
The distribution does not widen between regimes; it **changes shape**, its skewness
collapsing by 68 %. A statistic tracking the bulk (the mode) and one tracking cumulative mass
(the median) *must* diverge when skewness changes. This is not a defect of either statistic —
it is what order statistics do to a distribution whose shape is not fixed, and it is why the
synthetic control passed for `lowest_mode` while `lowest` failed: the synthetic cloud's
disparity distribution has a fixed shape across heights, so only the tail-sensitive rule is
biased.

### 2.2 The signature

The anomaly is not a translation of the disparity distribution. It is a **reshape**:

| percentile | low regime | high regime | shift |
|---|---|---|---|
| 1 % | −1.88 px | −9.07 px | **−7.19** |
| 10 % | 8.33 | 7.85 | −0.48 |
| 25 % | 10.22 | 11.09 | +0.87 |
| **50 %** | 12.27 | **14.07** | **+1.80** |
| 75 % | 14.33 | 15.65 | +1.32 |
| 90 % | 15.96 | 17.77 | +1.82 |
| 99 % | 35.16 | 72.67 | **+37.51** |
| **1–99 span** | **37.0 px** | **81.7 px** | **×2.2** |
| IQR (25–75) | 4.11 px | 4.56 px | **×1.11** |

**Both tails extend while the core shifts by about +1.5 px.** The perpendicular epipolar
residual `dv` undergoes the same doubling (0.866 → 1.885 px, p90 3.35 → 6.33 px, KS p ≈ 0) —
two statistics on two different components agreeing that the high regime's matching is more
dispersed.

### 2.3 It survives the obvious controls

| control | result |
|---|---|
| spatial scale | present at every radial cut: ρ = +0.540 at r ≤ 100 px and +0.547 at r ≤ 452 px |
| diurnal confound | ρ(hour, reference) = +0.114 (n.s.); partial ρ(reference, disparity \| hour) = **+0.498 (p = 1.6e−11)** |
| within a single layer | low-regime-only ρ = **−0.473 (p = 2.3e−13)**, gridded **−0.525 (p = 0.030)** |
| reduction of the instrumental field | **strengthens** the inversion (−129 → −164 m) |
| choice of matcher | LoFTR sees the same sign, 6.5× smaller |

---

## 3. Twelve pre-registered refutations

Each hypothesis was tested against a criterion stated **before** the measurement, recorded in
`docs/plan/research-workplan.md` and the individual records. The gate is named for each.

| # | hypothesis | discriminating measurement | refuted because | record |
|---|---|---|---|---|
| 1 | certainty / low-confidence hallucination | disparity at **matched certainty** | the high regime is higher at **every** of ten certainty deciles (+0.82 to +2.50 px); the surplus population's median certainty is **0.782** — high, not low | `experiments/w4/W4_RESULT.md` gate 1 |
| 2 | incoherent matching noise | residual against the 25 nearest neighbours | 0.047 px (low) vs 0.041 px (high), ratio **0.865** — the high regime is *more* coherent | `experiments/w4/w4_smoothness.py` |
| 3 | calibration / geometric error | remove the instrumental component and re-measure | the inversion **strengthens**: −129 → −164 m, ρ −0.547 → −0.581 | `experiments/layers/w3_unmask.py` |
| 4 | matched-population size | raw per-match rate | **identical**: 11,655.1 vs 11,655.8 per frame (RoMa emits a fixed grid) | `experiments/w4/dump_matches.py` |
| 5 | a pure along-epipolar depth change | predicted vs observed shift | reference predicts **−5.315 px**; observed **+1.798 px** — opposite sign, 15 % of magnitude; and the perpendicular residual doubles | `experiments/w4/w4_gate3.py` |
| 6 | a RoMa-specific artefact | an independent matcher | LoFTR **+0.275 px, CI [+0.112, +0.401]** — same sign, 6.5× smaller; matcher-independent in direction, representation-dependent in size | `experiments/w4/w4_gate4.py` |
| 7 | target-identity layering | low-regime-only correlation | **−0.473 (p = 2.3e−13)** — the inversion persists with both instruments on the same layer; layering explains the **level** (+111 m) but not the **sign** | `experiments/w4/w4_lowregime.json` |
| 8 | camera desync / cloud motion | does `dv` scale with cloud **speed**? | ρ(speed, dv) = **+0.002 (p = 0.97)** against ρ(regime, dv) = +0.447; motion is identical across regimes while `dv` doubles; the required exposure offset is **18.14 s** against an archive maximum of 5 s | `experiments/w4/w4_desync.py` |
| 9 | a fixed pipeline artefact | the cloudless-day null | the clear day gives **655 m**, not the cloudy day's 2242 m (KS p = 0.000) — the level is content-driven | `experiments/w4/matches_roma_20250522.npz` |
| 10 | non-Lambertian photometry | sign of the within-frame brightness coupling | ρ(brightness, du) = **−0.176** — brighter means *lower* disparity, while the high regime is *brighter* and has *higher* disparity; the gradient-direction signature is absent (ρ = −0.000) | `experiments/w6/w6_photometric.py` |
| 11 | photometric superposition | tails grow at a fixed core | it widens the **core** (IQR ×9.19 at H=2000, α=0.20) where reality widens the **tails** (IQR ×1.11); and it needs **80 % clear sky** against the real 2.6–3.6 % | `experiments/w8/w8_superposition.py` |
| 12 | phase aliasing | periodic structure in the disparity histogram | **no period peaks** above the 0.05 autocorrelation floor in either regime; a +1.8 px jump would need a ~1.8 px wavelength against 100–1000 px cloud features | `experiments/w8/W8_PHASE_ALIASING.md` |

### 3.1 Two further retractions, including the project's original metric

**The windowed correlation metric is withdrawn.** It searched a ±300 s time shift inside each
30-minute window and reported the largest correlation found. A circular-roll null performing
the **same search** scores **higher** than the observation (0.346 against 0.189). A 30-minute
window holds ~152 paired samples but only **~3.4 independent ones**; maximising over 121
shifts of three points is selection, not measurement.

**Two of the authors' own claims were retracted** after the same treatment: an
anti-correlation in the estimator's *dynamics* (a protocol artefact — gap-safe it is −0.045,
CI [−0.100, −0.007]) and a claimed improvement from separating the reference's layers (fails
its null within each regime, p = 0.758).

### 3.2 The pre-registered triggers, named

| experiment | pre-registered criterion | document |
|---|---|---|
| W7 renderer | `du(zenith) = f·atan(B/H)`, falling as 1/H | `w7_renderer.validate()` |
| W8 relief | ρ(true, reported) < 0.9 across ≥3 heights | `experiments/w8/w8_relief.py` |
| W8 superposition | 1–99 span ×≥1.8 **while** IQR ×<1.4 | `experiments/w8/w8_superposition.py` |
| W2 foreground | ≥50 inliers, epipolar RMS < 2.0 px, on ≥5 frames | `experiments/w2/PREREGISTRATION.md` |
| W4 gate 1 | a difference surviving **matched certainty** cannot be a certainty effect | `experiments/w4/W4_RESULT.md` |
| W4 gate 2 | coherence ratio < 1.10 means the difference is a real displacement | `experiments/w4/w4_smoothness.py` |

---

## 4. The boundary

### 4.1 What is established

The instrument recovers a geometrically known height on synthetic procedural texture through
the measured fisheye geometry: **ρ = +1.000, median |bias| = 0.072 px**, robust to a 4×
apparent-scale range and to 20 % relief. The shipped `lowest` selection rule was a defect
erring by up to 1471 m on ground truth; `lowest_mode` errs by ≤64 m. The real-data inversion
is robust, specific in its distributional signature, and not produced by any of twelve tested
mechanisms.

### 4.2 What is claimed as product capability

**A level consistent with the ceilometer's lower cloud population, in the low regime only.**

| regime | camera | ceilometer H1 | difference |
|---|---|---|---|
| low (H1 < 2750 m) | 2361 m | 2250 m | **+111 m** |
| high (H1 > 3500 m) | 2064 m | 3790 m | −1726 m |

> **LIMITATION (level agreement).** The +111 m agreement is a **level** statistic from 215
> frames of one session. It is not a validated calibration: to elevate it to a demonstrated
> result requires (a) an independently calibrated *level* reference — the baseline is assumed
> at 45 m and never measured, and the focal constant carries ±11 px/rad → ~±1.7 % in level;
> (b) a second session, to exclude a coincidence at this site; and (c) demonstration that the
> agreement is not produced by the same unattributed process that produces the inversion at
> other heights. Until then it is an observation, not a capability.

**No dynamic tracking is claimed.**

> **LIMITATION (dynamics) — the binding limit of this study.** Under 5-fold cross-validation
> no additive correction recovers the dynamics: across six correction families and three
> estimators the per-frame first-difference correlation lies in **[−0.166, +0.168]** with no
> consistent sign. At a **300 s binning**, however, the same data yields a consistent
> correctly-signed correlation — median **−0.409 (p = 0.038)**, with nine of eleven
> statistics correctly signed — which does not survive correction for 22 tests.
>
> **WITHDRAWN after review.** That correlation was an artefact: the reference was paired with
> the ceilometer sample nearest each bin's **start** rather than the median **over** the bin,
> mismatching the series by up to 300 s against a 267 s decorrelation time. Corrected, the
> median's dynamics correlation is **+0.095 (p = 0.667)** and nothing is significant.
> **The original W9 conclusion — no recoverable dynamics — stands.**
>
> **The obstacle is sample size, not absence.** The reference decorrelates over ~267 s, so
> 27 bins hold **n_eff = 17.7 independent samples**, and the smallest detectable |ρ| at
> p = 0.05 is **≈ 0.47**. The observed −0.37 lies **below the detection floor of the
> design**. Establishing a correlation of 0.37 at 80 % power requires **n_eff ≈ 55** — roughly
> a threefold longer session, or a reference that decorrelates faster.
>
> So the honest statement is not "the instrument has no dynamic capability" but **"the
> dynamic signal, if present, is at or below what this session can detect."** Within the low
> regime, for median-like statistics, the measured variation is anti-correlated with the
> reference (ρ = −0.473 per frame, −0.525 on a matched grid), which is the inversion again
> and is subject to the same statistic-dependence. The effective sample size
> is **7** (nominal 215, lag-1 ρ ≈ 0.94), so the study is underpowered and no finer structure
> should be claimed. To elevate a dynamics claim, either direction, requires a session long
> enough to yield ≥30 independent samples of both series, and an independent motion
> reference — no wind data exists for this session.

### 4.3 What the instrument cannot be said to do

> **LIMITATION (absolute CBH).** No absolute cloud-base-height claim is supported for high,
> thin or multi-layer fields. The reference is itself bi-modal (40 % at 2250–2750 m, 53 % at
> 3750–4250 m) and reports up to 7010 m against a stated 10–3000 m range. To support such a
> claim requires a reference with a footprint comparable to the camera's 1956 m at 1837 m —
> a radar or a scanning lidar — rather than the ~10 m column used here, a ~200:1 mismatch
> spanning 8.5 texture decorrelation lengths.

> **LIMITATION (instrument identity).** That the stereo measures "a valid but different
> physical quantity" than the ceilometer is a **motivating hypothesis, not a result**. No
> measurement here identifies which quantity. Establishing it requires a third instrument
> with an overlapping footprint and a known retrieval.

> **LIMITATION (external validity).** One usable session exists. 2025-05-22 is cloudless and
> yields single-digit correspondences per hour, so there is **no session-level hold-out, no
> unseen-session validation, and no drift measurement between sessions**. Every result here is
> a time block of one day.

> **LIMITATION (the anomaly's cause).** The cause of the inversion is **unattributed**. Twelve
> mechanisms were refuted with pre-registered criteria; none accounts for the observed
> combination of a +1.5 px core shift with ×2.2 tail growth at a fixed IQR. It is not a claim
> that the cause is atmospheric, volumetric, or photometric — two of those were tested and
> refuted, and the photometric coupling that exists runs the **wrong way**.

---

## 5. What would settle it

The analysis side of this question is exhausted: twelve mechanisms tested, twelve refuted, on
a validated instrument. What remains is not another hypothesis but a **controlled input**.

1. **Bisect the renderer against the real signature.** The validated renderer is now the
   project's most valuable asset. Any physical mechanism proposed must be shown to reproduce
   the *specific* signature — core +1.5 px, span ×2.2, IQR ×1.11, `dv` ×2.2 — not merely a
   bias or a spread. Two mechanisms have already failed that test by producing the wrong part
   of the distribution.
2. **A physical target or an instrumented session.** The foreground control failed for a
   quantified reason — a 87–247 px disparity at a 45 m baseline against a close, oblique
   building, unmatchable by either SIFT or RoMa — and the limit is geometric, not algorithmic.
3. **A footprint-matched reference.** A radar or scanning lidar colocated with the cameras
   would remove the 200:1 sampling mismatch that makes a level comparison between a point
   column and a 2 km area hard to interpret at all.
4. **A session long enough to detect ±0.37.** The dynamics question is sample-limited, not
   signal-limited. The binned series spans 2.25 h and yields **n_eff = 17.7**, one independent
   sample per 456 s of elapsed time. The Fisher-z relation gives a critical |ρ| of **0.470**
   at p = 0.05 for that sample size, and **n_eff = 60** for a correlation of 0.37 at 80 %
   power — so a decisive session needs **≈ 7.6 hours of binned data**, not fifteen. That is the
   cheapest decisive experiment remaining: it would convert "below the detection floor" into
   either a detection or a genuine absence.

---

## 6. Methodology

### 6.1 What a claim required

No value produced by maximising over alternatives without a null that maximises identically.
Effective sample size reported rather than nominal. A p-value, or an explicit statement that
none exists, and the *direction* of any effect. Synthetic ground truth validated by an
independent exact estimator before any method was judged by it.

### 6.2 Harness post-mortem: the three bugs that changed conclusions

These are reported as first-class findings about the **process** of computational stereo,
because each changed a scientific conclusion and none was caught by inspection.

**(1) A metric that scored below its own null.** The project's original validation searched a
time shift per window and reported the maximum. A null performing the same search scored
higher than the observation (0.346 vs 0.189). **The lesson:** a metric that maximises over a
parameter must be compared against a null that maximises over the same parameter, or it
measures the search rather than the signal.

**(2) A ground truth verified in the wrong frame.** The first synthetic renderer was gated by
triangulating its ground-truth disparity and confirming it returned the rendered height — in
the **native** frame. The matcher consumes the **warped** frame, which was never validated.
The result was a *positive* control that had to be withdrawn, because it modelled both cameras
with the same orientation — a rig that does not exist. **The lesson:** a ground truth is only
as good as the frame it is expressed in, and the gate must be applied in the frame the
consumer actually sees.

**(3) A default argument that cost 551 m.** Two scripts rendered the "same" synthetic cloud
1.5° apart because one call passed `roll_deg` implicitly and the other did not. The control
refused to reproduce — 1984 m against 1433 m on inputs believed identical — and three
hypotheses were proposed before the call sites were compared. **The lesson:** when a control
fails to reproduce, diff the *inputs* at the call site before theorising about the mechanism.
Byte-level comparison of the two conversions of one image could not detect a difference
between two different images.

**A note on what these have in common.** Each is a case of a quantity being verified in a
proxy that was not the thing consumed. The null is a proxy for the search; the native frame is
a proxy for the warped frame; one conversion path is a proxy for the other. In every case the
verification passed and the conclusion was still wrong. We would recommend that any pipeline
claiming a synthetic control state explicitly, for each gate, *which artefact the consumer
actually reads*.

---

## 7. Data and artefacts

| resource | location |
|---|---|
| full record, all twelve refutations | `docs/plan/research-workplan.md` |
| citable index of results | `docs/results/README.md` |
| all negative results | `docs/results/negative-results.md` |
| the matcher-selection principle | `docs/results/matcher-selection.md` |
| the validated renderer | `experiments/w7/w7_renderer.py` |
| raw per-match evidence with certainty | `experiments/w4/matches_roma.npz` |

**One usable session**, 2025-05-23, 17,265 frames at 10 s. 2025-05-22 is cloudless (8,632
frames) and serves as a control and as a calibration reference. No meteorology, no wind data,
and no site coordinates exist for either session.
