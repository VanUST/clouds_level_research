# Stage audit — reassessment from step one

**Date:** 2026-10-13 · **Status:** working document, findings are falsifiable claims

Every number below was produced from the historical data in `~/clouds_data/` and
can be re-derived. Measurements are stated with the method that produced them.
Nothing here is tuned to any metric.

The purpose is to list what is **wrong or unverified** in the current approach
before rebuilding it, not to defend it.

---

## 1. Calibration

### 1.1 The geometry is disparity-starved, so calibration precision is the binding constraint

With the configured parameters (frame width ξ₀ = 1920 px, baseline B = 45 m,
equidistant fisheye so f = ξ₀/π = 611.15 px/rad) the height relation
D = f·B/Δξ gives:

| D true | disparity | m per px | % per px | px needed for 10 % | px needed for 1 % |
|---|---|---|---|---|---|
| 500 m | 55.0 px | 9.1 | 1.8 % | 5.50 | 0.55 |
| 1000 m | 27.5 px | 36.4 | 3.6 % | 2.75 | 0.28 |
| 1500 m | 18.3 px | 81.8 | 5.5 % | 1.83 | 0.18 |
| 3000 m | 9.2 px | 327.2 | 10.9 % | 0.92 | 0.09 |
| 5000 m | 5.5 px | 909.0 | 18.2 % | 0.55 | 0.06 |

Height error is **relative**: σ_D/D = σ_Δξ/Δξ. Consequences:

- The written work's "≈30 m per pixel" is true only near 550 m. At 1.5 km it is
  82 m/px and at 3 km, 327 m/px. The sensitivity claim in the text understates
  the problem by ~3× at typical heights.
- Reaching **10 %** height accuracy at 1.5–3 km requires the disparity to be
  accurate to **≈1 px**. A 1 px calibration error produces 5.5 % error at 1.5 km
  and 10.9 % at 3 km.
- A calibration **translation** error of t px biases every match by t px, i.e.
  directly scales all heights by (1 + t/Δξ). At t = 1 px that is +5.5 % at
  1.5 km; at the drift the written work reports (up to 30 px between sessions)
  it is a factor of ~2.6.

### 1.2 A rotation error is not a constant bias

A rotation error δθ displaces a match by ≈δθ·r, so it varies across the field:

| δθ | r = 160 px | r = 320 px | r = 452 px (crop corner) |
|---|---|---|---|
| 0.05° | 0.14 px (0.8 %) | 0.28 px (1.5 %) | 0.39 px (2.2 %) |
| 0.10° | 0.28 px (1.5 %) | 0.56 px (3.0 %) | 0.79 px (4.3 %) |
| 0.50° | 1.40 px (7.6 %) | 2.79 px (15.2 %) | 3.94 px (21.5 %) |

(Percentages at D = 1500 m.) A single global affine can absorb a rotation error
exactly only if it is the *same* rotation for all frames; any drift is
height- and position-dependent and therefore not removable by a constant shift.

**Unverified:** the current calibration is a single matrix for two sessions, dated
by filename only (`calibrated_on_verified: false`), with no residuals, no source
images and no per-session check. Whether it is inside the ±0.2 px needed for
~1 % accuracy **is unknown**.

### 1.3 The focal constant f is assumed, not measured — and the assumption looks wrong

The pipeline computes f = ξ₀/π, i.e. it *assumes* the 180° field maps exactly
onto the image circle inscribed in the 1920 px frame. Nothing measures it.

Measuring the radius at which the frame goes black and stays black, per azimuth,
over four daytime frames:

| frame | median edge radius | implied f = r/(π/2) | ratio to ξ₀/π |
|---|---|---|---|
| 09:00 | 1078 px | 686 | 1.12 |
| 12:00 | 1156 px | 736 | 1.20 |
| 15:00 | 1078 px | 686 | 1.12 |
| 17:00 | 1084 px | 690 | 1.13 |

The edge is soft (the frame fades to black between r ≈ 1000 and r ≈ 1150), so the
radius is bounded rather than precise — but **every measurement is well outside
the r = 960 px that f = ξ₀/π implies**. If the lens really spans 180° across its
circle, then f is 10–20 % larger than assumed, and since D ∝ f, **all heights
would be biased low by roughly that factor**.

This is a first-order systematic error on absolute height, it is independent of
matching quality, and it is exactly the kind of thing that would produce a
persistent offset against the ceilometer. It must be settled by an independent
reference, not by assumption.

The camera's true field of view, the exact image-circle radius, and the 
manufacturer's projection model are **not recorded anywhere in the project**.

### 1.4 There is no absolute calibration reference in the loop

A two-camera rig with a horizontal baseline cannot separate "calibration" from
"height": any disparity can be explained by some height. The project's own
calibration used the Sun (an object at infinity, whose images must coincide after
rectification), but that check is not part of the current pipeline and no
residual was preserved.

**A first attempt at the Sun test failed and is reported as a negative result:**
thresholding the brightest pixels finds saturated *cloud*, not the solar disc
(detected areas 10⁴–4×10⁴ px against a disc of ≈26 px), so the resulting
"sun shift" was really cloud disparity. A usable test needs a proper disc
detector.

### 1.5 The 45 m baseline is the structural limit

Disparity scales as `f·B/D`, so a short baseline costs height accuracy directly.
Comparable published ground-based fisheye stereo, evaluated at the same cloud
height:

| rig | baseline | f | disparity at 1500 m |
|---|---|---|---|
| **this project** | **45 m** | 611 px/rad (assumed) | **18.3 px** |
| Crispel & Roberts 2018 (AMT) | 150 m | ≈937 px/rad (2944 px, 180°) | ≈93.7 px |
| Beekmans et al. 2016 (ACP) | 300 m | — | ≈5–10× this rig |

With 5–10× less disparity, the same angular error costs 5–10× more height error.
No improvement to matching, geometry or tracking removes this: it bounds what the
rig can deliver and must be stated rather than worked around. It also explains
why the calibration margin question is acute here in a way it is not in the
published work — those rigs can tolerate calibration error this one cannot.

### 1.6 Calibration error budget is not stated anywhere

The project has no statement of the calibration accuracy it needs, no measurement
of the accuracy it has, and no runtime check. The margin question — how wrong can
calibration be before the product is useless — is the central question of this
work and is currently unanswered.

---

## 2. Height estimation geometry

### 2.1 The formula is paraxial but the crop is not

The pipeline uses D = ξ₀·B/(π·|Δξ|) on the x-component of the mean disparity.
That relation is the small-angle limit. Evaluating the exact equidistant
projection against it over the 640 px crop:

| D true | position | θ | exact Δu | paraxial D | bias |
|---|---|---|---|---|---|
| 1000 m | centre | 0° | 27.48 | 1000.7 | +0.1 % |
| 1000 m | x = 150 | 8.5° | 27.06 | 1016.4 | +1.6 % |
| 1000 m | corner (320,320) | 24.3° | 24.61 | 1117.4 | **+11.7 %** |
| 1500 m | corner | 16.8° | 17.40 | 1580.2 | +5.4 % |
| 3000 m | corner | 8.6° | 9.05 | 3040.7 | +1.4 % |

The bias is **position-dependent**, so it both offsets and broadens any
distribution of heights, and it is worse for low clouds. It is removable exactly
by triangulating properly instead of using the paraxial formula.

### 2.2 The vertical disparity is discarded

Two matched points rectified from a horizontal baseline have a **non-zero**
vertical disparity away from the optical axis, and it grows with off-axis
distance. The pipeline ignores it entirely. Measured on one frame
(2025-05-23T12:00:01, 3719 matches): `disp_y` median **+1.78 px**, 66 % of matches
with |disp_y| > 2 px, 42 % with |disp_y| > 5 px.

That is not necessarily a calibration fault — the exact model predicts a vertical
component off-axis. But it means the pipeline is throwing away information and
has no consistency check. Modelling the full 2-D shift lets the vertical
component be used, and its *residual* against the model is a direct calibration
diagnostic.

### 2.3 A single horizontal plane does not fit the field

Fitting the exact forward model of a horizontal plane at height D to one frame's
full 2-D shift field leaves a median residual of 6.4 px (frame 12:00:01) and
5.7 px (14:00:37) against shifts of 16–22 px — i.e. 26–40 % of the signal.

This is **expected** if the cloud base has real vertical relief, and it is not by
itself evidence of a fault. It does mean a plane fit is the wrong estimator, and
that per-match heights (with their spread) are the right output rather than one
plane height per frame.

---

## 3. Matching

- LoFTR yields **3719 matches** on a 640 px crop of 2025-05-23T12:00:01 — the
  density is not a limitation. A second frame gave 735, so count varies a lot.
- No left-right / cycle consistency check is enabled by default.
- No per-match uncertainty is produced, although the whole method's error budget
  is dominated by disparity error (§1.1).
- Rectification residual is unmeasured (§2.2).

**Not yet tested:** sub-pixel localisation accuracy of LoFTR on this imagery;
behaviour in textureless clear sky; whether matches concentrate on cloud edges
rather than cloud interiors; sensitivity to the matcher threshold; comparison
against any alternative matcher.

---

## 4. Clustering and layer extraction

The HDBSCAN stage is **not producing physical layers**, and the parameter that
looks like a detail decides everything. One frame (3719 matches):

| min_cluster_size | min_samples | clusters found | largest | noise |
|---|---|---|---|---|
| 3 | 3 | **478** | 130 | 24.0 % |
| 5 | 3 | **10** | 901 | 0.8 % |
| 10 | 5 | 9 | 901 | 1.6 % |
| 20 | 10 | 8 | 895 | 2.3 % |
| 50 | 20 | **1** | 3684 | 0.9 % |

Across the full validation run the dump holds **232 clusters per frame on
average, median cluster size 5**. The configured `min_cluster_size: 3` shreds the
match set; the pipeline's 300 m height-merge then collapses ~232 fragments into
~7 groups, and it is that merge — an undocumented workaround — that actually
performs layer separation. The reported "spatial rejection rate" (15.4 %) and the
cluster counts are therefore artefacts of two interacting parameters, not
measurements of cloud structure.

Additional defects found:

- **`cluster_selection_epsilon > 0` crashes.** `sklearn.cluster.HDBSCAN` raises
  `TypeError: only 0-dimensional arrays can be converted to Python scalars` at
  ε = 0.2 and 0.5 (verified). The config exposes this knob and it does not work.
- Within-cluster disparity spread is *tiny* (median σ_disp = 0.082 px, i.e.
  0.6 % of the disparity). This is not measurement uncertainty — it is the spread
  of a handful of correlated matches, and using it as an error bar would
  understate the true uncertainty by a large factor.

---

## 5. Acquisition geometry and the scene

Viewing a full frame with a radius overlay shows two things the pipeline ignores:

- The fisheye circle is inscribed and the **frame corners are outside the image
  circle** (black).
- There are **near-field occluders**: trees intruding to r ≈ 400–500 px on one
  side, a building and a mast at the bottom, and a wire crossing the upper field.

The pipeline's crop is a **square** 640×640, whose corner radius is 452 px, so
its corners reach into that foliage. A square crop is also not the natural field
of view of a circular image. A circular, radius-limited mask would exclude
near-field objects that cannot be clouds.

---

## 6. Temporal processing

- `mode: simple` selects the **lowest** cluster each frame with no physical model
  of a cloud layer; the Kalman filter then smooths whatever that rule picked.
  Selection errors are smoothed, not corrected.
- The Reliability Index is computed but **not used** for selection.
- The 300 m merge tolerance (§4) is a tuning constant standing in for layer
  identity.
- Wind advection between the two instruments is real and is why the comparison
  needs a per-window shift; the pipeline has no advection model.

---

## 7. What is not wrong

Stated so the rebuild does not discard working parts:

- The height↔disparity relation, in its exact form, is correct for an equidistant
  fisheye; only its paraxial approximation is being used (§2.1).
- LoFTR produces dense correspondences on this imagery; density is not the
  bottleneck (§3).
- The temporal filter demonstrably smooths: autocorrelation 0.069 → 0.978,
  mean frame-to-frame jump 338 m → 18 m (crop 320), 78.5 → 28.5 m (crop 640).
  Whether it smooths the *right* signal is a separate question.
- Rectification is not grossly broken: the warped frame and the reference frame
  overlay visually and the centre crop is fully filled.

---

## 8. Ranked candidate causes of the absolute disagreement with the ceilometer

Ordered by expected size, to be tested in this order (each is falsifiable):

1. **f assumed rather than measured** (§1.3) — 10–20 % scale, one-directional.
2. **Calibration translation/rotation error** (§1.1–1.2) — unbounded; drift is
   documented at up to 30 px between sessions.
3. **Paraxial formula used off-axis** (§2.1) — up to ~12 % at the crop corner.
4. **Layer selection** (§4, §6) — picks a fragment, not a layer; sign varies.
5. **Genuine optical-vs-geometric difference** between the instruments — the
   physical offset the written work describes, which must be *characterised*,
   not assumed.
