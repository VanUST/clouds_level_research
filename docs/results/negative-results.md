# Negative results

Recorded because a negative result is a result, and because each of these cost real
time and would otherwise be repeated. Every entry names the measurement that
established it.

## Calibration and geometry

| attempt | outcome |
|---|---|
| Brightest saturated blob as the Sun, to give an infinity reference | finds saturated **cloud** (10⁴–4×10⁴ px against ~26 px for the disc) |
| Gradient-convergence glow centring for the Sun's position | fails — r/θ scattered 492–945 px/rad; the glow has no consistent centre |
| Star-field calibration at night | impossible — the frames are pure noise, the exposure is far too short |
| The Moon as an angular reference | no Moon in the night frames |
| **A relative camera rotation as the cause of the radial trend** | the fit gave 37 % → 2.5 % but predicted **13.7 px** of vertical disparity against **2.24 px** measured — a height/rotation degeneracy, not a measurement |
| Four ideal lens laws (equisolid, stereographic, orthographic) as the radial cause | each leaves 20–26 % of the trend; only a fitted cubic distortion removes it |
| A radial scale error as the cause | removing it changes the trend from 7.2 % to 7.3 % — nothing |
| A second session as an independent period | 2025-05-22 is cloud-free and yields no stereo signal |

## Matching

| attempt | outcome |
|---|---|
| **EfficientLoFTR**, despite the literature promising a fix for LoFTR's fine-stage variance | **+204 %** on the pre-registered residual; a **30 %** gain error differentially. Faster, not better |
| **MASt3R** as the primary matcher | dense and internally consistent, and **zero differential response** — a 1.5 px change in a 20 px disparity produces no change at all |
| **RoMa and DKM for coverage** | the most accurate on the matches they return and **0 % interior coverage** at the default certainty — though for RoMa this proved a threshold, not a limit (see [matcher-selection.md](matcher-selection.md)) |
| **Depth Anything 3** | can be turned into a disparity via depth + pose, but is non-metric and inaccurate (5.11 px) |
| **StereoSGBM and DIS optical flow** | both invent a full field on a cloudless sky (6400 and 5900 per frame) |
| **DISK + LightGlue** (sparse control) | 6.6 % interior — sparse detectors concentrate on cloud edges |
| **MatchAnything / ELoFTR** | not run; its third-party tree is absent from this checkout |
| Radial flatness as a matcher-selection criterion | does **not** discriminate — every candidate gives −2.3 to −3.4 m/px, so the radial bias is geometry, not matching |
| **One-dimensional clustering** on the horizontal disparity alone | 95.66 % of match labels change versus two dimensions; not equivalent |
| HDBSCAN `algorithm` variants | `auto` == `kd_tree`; `ball_tree` 1.15× slower; `brute` 2.2× slower and 2,948 of 24,507 labels differ |
| Dropping `store_centers='centroid'` | bit-exact but 2 ms/frame **slower** |
| `np.bincount` vectorisation of the cluster statistics | differs from `np.std` by up to 2.2 × 10⁻⁶ px — rejected to keep `batch_size: 1` bit-reproducible |
| Batching the matcher end-to-end | 1.70× on the matcher stage, **~1.0× overall** — the matcher is 41 % of the time and nothing overlapped |
| Prefetch depth beyond 1 | within noise (3.98 vs 3.90 fps) |
| Per-frame JSONL logging as a bottleneck | 0.0 ms/frame — the suspicion was wrong |
| A second meteorology or wind source for the session | none exists for 2025 |
| The ceilometer's backscatter width `L` as a better reference | populated in **2 of 5,759** slots, 0 in the validation window |
| `cluster_selection_epsilon > 0` | crashes sklearn 1.9 HDBSCAN on this data, and its fallback silently returned *no clusters for the frame* — indistinguishable from clear sky |

## Geometry: an error that was compensating for another

Applying **exact triangulation without the lens distortion made the radial trend
worse, not better** — the profile's interquartile range went from 4.1 % (paraxial,
ideal lens) to 19.1 % (exact, ideal lens), against 2.4 % once the measured distortion
is applied. The paraxial formula's off-axis bias and the lens distortion's radial
effect act in opposite directions and partially cancel, so removing one alone
destroys the compensation. Two errors that cancel are still two errors, and fixing
either in isolation is a regression.

## The foreground control (W2) — the one available geometric control does not work

| attempt | outcome |
|---|---|
| SIFT on the rigid foreground (mast, building, roof panels) with the mask applied | 11 matches at ratio 0.75, 109 at 0.85 — ambiguous at any defensible threshold |
| RoMa on the foreground band | **zero** matches at certainty 0.9, 0.7 and 0.5; 35 at 0.2 |
| RoMa on the building region alone | **zero** at 0.5 and 0.3; 246 at 0.15 with a disparity IQR of **91 px**, i.e. incoherent |
| **cause** | the foreground disparity is **87–247 px**; at that displacement the cameras view the building ~20–40° apart, so local appearance changes too much to match |

Consequence: the H1/H2/H3 attribution of the static pattern could not be performed, and
the pre-registered protocol halted at its first gate. The recommendation is not to retry
with another matcher — the limit is the 45 m baseline against a close, oblique foreground.

## Candidate diagnoses that were tested and rejected

| hypothesis | outcome |
|---|---|
| **The static field is caused by the pipeline triangulating affine-warped camera-1 coordinates about the crop centre** (a "coordinate-frame bug") | **rejected on measurement.** The code fact is confirmed and the affine does carry a 54.34 px residual, but applying the proposed correction (rays about the true principal point) makes the static field **worse**, not better: r between session halves 0.912 → 0.939, spatial spread 92 → 115 m |
| A relative camera rotation explains the radial trend | the fit gave 37 % → 2.5 % but predicted 13.7 px of vertical disparity against 2.24 px measured — a height/rotation degeneracy |

## A headline that was true of the wrong configuration

The "1000–1400 m within-frame scatter" was measured on the **uncorrected** instrument
(LoFTR, no roll correction). On identical matches the corrected configuration gives
**235 m**. Both numbers are correct; quoting the first as a property of the method was
not. Any statement about field width must name the configuration it was measured on.

## R10 mechanisms

| hypothesis | outcome |
|---|---|
| The reported height is an order statistic set by the field's spread and the match count | **refuted** — R&sup2; = 0.001; both coefficients are indistinguishable from zero |
| A persistent artefact dominates the low tail the selection rule picks | partly supported — the selection is 5.5&times; more concentrated than the field, at a median radius of 350 px, but a genuinely low cloud in one part of the sky would look the same |

## Evaluation

| attempt | outcome |
|---|---|
| The windowed best-shift correlation as the validation metric | a circular-roll null that performs the same shift search scores **higher** than the observation (0.346 vs 0.189 raw) — withdrawn |
| The "±300 s boundary in ~45 % of windows" figure | wrong; measured 15–25 %, max 30 % across stored runs |
| Claiming the estimate's **dynamics** are anti-correlated | a protocol artefact — 30 of 469 differences spanned a missing epoch; gap-safe the value is −0.045, CI [−0.100, −0.007] |
| Claiming that separating the ceilometer's layers **improves** the correlation | fails the null within each regime (p = 0.758 for the lower layer); the layer *structure* stands, the improvement does not |
