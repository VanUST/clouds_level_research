# Research workplan — what could actually improve the result with the data we have

**Status:** proposed. This is a *survey and a plan*: it does not implement the
research, it ranks what should be attempted and states what the data cannot
support.

**Companion documents:** [README.md](README.md) (W0–W6 and the objectives),
[evaluation.md](evaluation.md) (the metric hierarchy this plan adopts),
[w1-calibration.md](w1-calibration.md) (measured `f`, roll, distortion),
[temporal-alignment.md](temporal-alignment.md) (why the old metric was withdrawn),
[stage-audit.md](stage-audit.md) (the original audit),
[pipeline-steps.md](pipeline-steps.md) (what was removed),
[guardrails.md](guardrails.md) (G1–G13, binding).

**Measurements made for this workplan:**
[`experiments/planning/anti_correlation_probe.py`](../../experiments/planning/anti_correlation_probe.py)
→ `anti_correlation_probe.json`. Every number introduced in §1 below is in that
file and reproducible with:

```bash
source /home/iustimov/miniconda3/etc/profile.d/conda.sh && conda activate clouds
cd /home/iustimov/clouds && python experiments/planning/anti_correlation_probe.py
```

---

## 0. How to read this plan

Two different things are needed and they must not be confused:

- an **estimate** must be produced without the ceilometer (guardrail G1, G13);
- a **comparison** may use the ceilometer, after the estimate exists.

Nothing here selects a matcher, a cluster rule, a mask or a geometry model by its
agreement with the ceilometer. Every selection is made on an internal criterion
(guardrail G1, G4). The ceilometer appears only as a *comparison* and as a
*stratification variable for comparisons that are already defined*, never as an
input to an estimate.

The plan deliberately prefers few, well-argued directions over a long list. Eight
ranked directions (§3; R3 is the enabling evaluation item rather than a research
direction on its own), a can't-support list (§4), and an ordered sequence (§5).

---

## 1. What the estimator is actually doing — three measurements that set the agenda

All three are new measurements on stored artifacts, made for this plan. They change
which question is central.

### 1.1 The "frame-to-frame anti-correlation" does not survive a gap-safe protocol

The motivating result (`evaluation.md` §3) is r = −0.248 on 469 "near-independent
differences". Reproducing it exactly reproduces the number — and shows where it
comes from:

| series | n | r | 95 % block-bootstrap CI |
|---|---|---|---|
| levels, 30 s grid | 470 | **−0.376** | [−0.653, −0.262] (1800 s blocks) |
| first differences, `np.diff` over the masked array (the reported protocol) | 469 | **−0.248** | — |
| first differences, only genuinely 30 s-adjacent pairs | **439** | **−0.045** | [−0.100, −0.007] (600 s blocks) |

The reported difference series contains **30 pseudo-differences that span missing
epochs** (the estimator has no result in 23 frames and the reference in others; on
the 30 s grid, 470 valid cells form 31 separate blocks). `np.diff` over the masked
array silently differences across every gap. Removing the 30 cross-gap pairs
collapses r from −0.248 to **−0.045**, with a CI that just excludes zero.

**Consequence.** The *level* anti-correlation is real and robust (r = −0.376,
bootstrap p < 0.001). The *dynamics* anti-correlation is an order of magnitude
smaller and protocol-dependent. The project's problem is therefore **not** that the
estimator tracks the cloud with the wrong sign frame-to-frame; it is that the
estimator's **level** is unrelated to — and weakly inverted against — the
reference's level. `evaluation.md` states the effect should be confirmed on an
independent period; there is no independent period (§4), so the correct move is to
make the protocol gap-safe and pre-register it, not to quote either number as final.

### 1.2 The estimator is nearly flat while the reference moves, and its response is inverted

Roll-corrected run, per-frame raw series, hourly medians (MSK):

| hour MSK | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|
| estimator, raw per-frame median (m) | 1590 | 1704 | 1573 | 1571 | 1570 |
| ceilometer H1 median (m) | 3150 | 2520 | 3920 | 3930 | 2480 |

The estimator varies by **±4 %** over five hours; the reference varies by **±30 %**.
Stratifying every dumped frame of the baseline run (1776 frames; 77 806 size-≥8
cluster records across the two regimes) by the reference's own regime gives the
direction:

| field | lower regime (`H1 < 2750 m`) | upper regime (`H1 > 3500 m`) | shift |
|---|---|---|---|
| frames | 551 | 738 | — |
| per-frame median of candidate cluster heights (m) | 1944 | 1839 | **−105** |
| KS statistic / p | | | 0.292 / 3.0 × 10⁻²⁴ |
| median within-frame height IQR (m) | 1434 | 1082 | — |
| median match count per frame | 716 | 1154 | — |

and restricted to the core `r ≤ 250 px`, so a residual radial geometry term cannot
produce it:

| field | lower | upper | shift |
|---|---|---|---|
| per-frame median height (m) | 1993 | 1839 | **−154** |
| KS statistic / p | | | 0.406 / 9.8 × 10⁻⁴⁷ |
| median within-frame height IQR (m) | 1097 | 578 | — |
| median match radius (px) | 227.9 | 227.7 | 0 |

Three things follow. (i) The estimator **does** respond to the reference's layer —
significantly, in the core, and the response survives radius restriction. (ii) The
response has the **wrong sign**: when the reference's lowest layer is ~1.5 km
higher, the estimator reports ~150 m *lower*. (iii) The between-regime shift
(−105 to −154 m) is **≈7–14× smaller than the within-frame spread** (1082–1434 m
full field). The height field is dominated by scatter that is not the cloud layer.

The match-radius distribution is identical in the two regimes, so this is not the
cloud field moving to a worse part of the lens.

### 1.3 The reported height is an order statistic of a heavy-tailed field

The shipped rule is `cbh_strategy: lowest`, with `min_cluster_size_for_tracking: 8`.
On the baseline dump (1776 frames):

| rule | median height (m) | hourly medians, MSK 12→16 (m) |
|---|---|---|
| `lowest` (shipped) | 1156 | 1162, 1193, 1164, 1124, 1142 |
| `largest` | 1744 | 1760, 1749, 1761, 1703, 1744 |
| `median of all candidates` | 1878 | 1834, 2004, 1889, 1873, 1847 |
| ceilometer H1 | | 3150, 2520, 3920, 3930, 2480 |

**Every internal rule is flat over five hours.** No selection rule fixes the level,
because the flatness is in the height field, not in the rule.

The selected `lowest` cluster has median height 1156 m (IQR 88 m), median radius
334 px, `u` median 508, `v` median 46, and **22.6 % of frames put it in one
60 × 60 px position bin** — a persistent feature at the top edge of the crop. But it
is not one removable occluder: matches with `v < 64` have median height 1181 m
against 1911 m elsewhere, yet excluding the top 128 px lifts the selected median
only to 1463 m, and a symmetric 128 px border exclusion to 1485 m. The low
population is a **broad tail**, and a `lowest` rule over a heavy-tailed field
reports the *extremum* of the noise, not the layer. The reported minimum, drawn
from a median of 716–1154 matches per frame, moves −105 m when the cloud moves
+1500 m: that is what an order statistic does.

**This is the single most important statement in the plan: the estimator is
reporting a low-tail order statistic of a noisy height field, whose level is set by
the field's spread and the match count, not by the cloud.**

### 1.4 The reference's secondary fields

`Zve_Sci_250523.txt` (DOL-2, `H1`, `H2`, `L`, `Q`):

| field | validation window (12:00–17:00 MSK, 906 slots) | whole file |
|---|---|---|
| `H1` valid | 906 | 2341 of 5759 slots |
| `H2` valid | 267 (29 % of `H1`) | 556 |
| `L` valid | **0** | **2 of 5759 slots** |
| `Q` | 906; 13 codes seen (7, 15, … 99) | 14 codes in the raw file (adds `Q = 0`) |

`Q = 0` occurs exactly when no layer is detected, so `Q` is not an independent
quality gate; it is a coarse discrete index (a cloud-amount-like code) and can be a
*stratification* variable only. **`L` is unavailable**, which makes the
literature-suggested test of the optical-vs-geometric offset (offset vs backscatter
width) impossible on this session (§4).

---

## 2. The central scientific question, and how it can be answered here

**Question.** With one usable session, one camera footprint, and a reference that is
spatially mismatched and structurally bimodal, can a physically-valid stereo
cloud-base estimator be shown to measure the same thing as the reference — and if
not, can the divergence be attributed to a named mechanism rather than absorbed?

**How it can be answered.** Split the claim, as `README.md` §3 and
`guardrails.md` G8 already require, and use the metric hierarchy of
`evaluation.md` §4, which is ordered by statistical power:

- **Tier 1 — internal, ground-truth-free.** Ray-intersection residual, radial
  flatness, cross-region agreement, temporal coherence, layer stability. Thousands
  of independent samples, available for every frame, and the tier that actually
  produced the project's real findings (focal constant, roll, distortion). **This is
  the primary target.**
- **Tier 2 — external, distributional.** Compare the *distribution* of per-match
  heights with the *distribution* of the reference over the session. No time
  alignment, so autocorrelation does not bite; the sample size is the number of
  measurements, not the number of independent epochs.
- **Tier 3 — external, temporal.** Only after Tiers 1 and 2 pass; grid at the
  *larger* decorrelation time; a null that searches identically; one comparison over
  the session; pre-registered direction and threshold. It has ~19 independent
  degrees of freedom on the filtered series and ~51 on the reference's own
  decorrelation time, so it cannot carry the weight the project placed on it.

The central question is answered by driving Tier 1 improvements until either the
Tier-1 criteria are at their physical floor *and* the Tier-2 distributions overlap,
or the residual divergence is traced to a named mechanism with a measurement. A
result that improves Tier 3 while leaving Tier 1 unchanged is not evidence
(G8, G12).

The two candidate answers this plan is built to distinguish:
**(A)** the estimator is retrieving the wrong *target*; **(B)** the estimator's
*geometry* maps a real cloud to a flat, inverted height. §1.2–1.3 show the field is
flat and heavy-tailed; §3 R1–R2 are the measurements that separate A and B.

---

## 3. Ranked directions

Ranking rule: **expected reduction of the dominant unexplained error per unit
effort**, where the dominant unexplained error is the ~1000–1400 m within-frame
scatter plus the ~600 m level gap, not the correlation coefficient. Effort is
ranked on the project's own artifacts: `L` = hours on cached data, `M` = a bounded
re-run (the full run is 1776 frames; the cluster dump exists), `H` = new geometry or
a new detector.

| rank | direction | why here | effort |
|---|---|---|---|
| R1 | target identity and the wrong-sign mechanism | decides whether the estimator is measuring cloud at all; highest information | L–M |
| R2 | apply the measured lens distortion in a joint geometry refit | measured, unapplied, and the largest known Tier-1 defect | M |
| R3 | the evaluation currency: Tier-1 composite + a spatial-sampling model | every later choice needs a currency; the point-vs-footprint comparison must be bounded | L |
| R4 | the level offset, decomposed against corrected geometry | synthesis; only meaningful after R1–R3 | M |
| R5 | multi-layer retrieval re-test under corrected geometry | the existing single-mode result must be re-confirmed, not assumed | M |
| R6 | reference refinement with `H2` and `Q` | real but low-power; `L` is dead | L |
| R7 | motion-based internal consistency (sector shear) | the only independent physical signal at 10 s cadence | M |
| R8 | calibration margin: `f`, tilt, translation as declared nuisance terms | bounds the deliverable's absolute scale; largely infrastructure | M–H |

### R1 — Target identity and the wrong-sign mechanism *(rank 1)*

**Hypothesis.** The estimator is not reporting the reference's lowest layer. The
`lowest` rule reports a low-tail order statistic of a heavy-tailed height field
(§1.3), and the field's small regime-to-regime shift (−105 to −154 m, inverted) is
a residual of the matched population, not of the cloud's base.

**Candidate mechanisms, with the measurement that separates them:**

| # | mechanism | prediction if true | measurement that separates it |
|---|---|---|---|
| M1 | **Order-statistic selection** — the field has a low tail and `lowest` takes it | replacing `lowest` with a mode/quantile estimator removes most of the level *and* the regime inversion, with no change to the matcher | compare `lowest` / `largest` / weighted median / mode on **Tier-1 criteria only**; report which is stable under its own parameters |
| M2 | **3-D cloud-side geometry** (the literature's candidate: unmodelled cloud sides, vertical parallax) | the low tail appears where the cloud has vertical extent; the shift should scale with the within-frame spread and vanish for thin cloud | per-match height vs local gradient and vs the vertical disparity component; cloud-thickness proxy from the match-count/IQR |
| M3 | **Texture-selectivity** — more matches in the upper regime (1154 vs 716), the extra matches bias low | the regime shift should vanish or reverse when the added matches are removed (equalise the match count across regimes) | match-count-matched subsampling within each regime; bootstrap over frames |
| M4 | **Radial geometry residual** (distortion/tilt) coupling match radius to height | the shift should vanish or reverse when restricted to the core, or when the radius distributions are matched | already partly excluded (§1.2: the shift *strengthens* to −154 m at `r ≤ 250` and the radius distribution is identical); re-test after R2 |
| M5 | **Regime mixture + low statistical power** (the reference switches between two clouds) | the inversion is entirely a between-regime effect and there is no within-regime tracking | within-regime level/difference correlation with a block bootstrap; already measured: `evaluation.md` §3 gives the lower-layer-only fixed-shift r = −0.069, upper-only +0.146, neither above the circular-roll null |

**Falsifiable prediction.** If M1 dominates, then a fixed-fraction quantile
estimator (e.g. the 25th percentile of per-match heights, or the dominant mode of
the height histogram) evaluated on the **same matches** will (a) reduce the
regime inversion from −105 m to within the block-bootstrap CI of zero, and (b)
reduce the within-frame height IQR's contribution to the reported level. If instead
the inversion survives a mode estimator, M1 is dead and M2/M3 are the survivors.

**Data and method.** `experiments/w1/matches_cache_2025-05-23.npz` (38 frames),
`experiments/layers/heights.npz` (167 frames, exact-triangulation per-match heights,
roll-corrected), and a new `--dump-clusters` run on the **roll-corrected** config
(the existing dump is the shipped roll = 0 config; that is itself a limitation to
state). Metrics: regime shift of the per-frame estimator, its block-bootstrap CI,
the within-frame height IQR, the match-radius distribution, and the Tier-1
criteria — never the ceilometer.

**Effort.** L on existing dumps; M for the roll-corrected re-dump (~1776 frames,
bounded by the existing evaluation run time).

**Risk.** The `lowest` rule may already be the only thing giving a low-variance
series; replacing it with a mode may trade level accuracy for jitter. Report both,
do not hide the trade-off.

**What kills it.** If no internal estimator (min, quantile, mode, weighted median)
both removes the inversion and improves Tier 1, then the low tail is in the
*matching*, not in the *selection*, and effort must move to R2 and the matcher.

### R2 — Apply the measured lens distortion in a joint geometry refit *(rank 2)*

**Hypothesis.** The radial trend in apparent height is lens distortion,
`θ = (r/f)(1 + a (r/f)²)` with `a = −0.30`, measured but not applied anywhere. It
is stable across `f = 611` and `f = 654` and flattens the profile from 36.7 % to
3.2 %. Applying it changes each match's ray direction by ~16 % at the crop corner.

**The competing explanation must be named.** `experiments/layers/paraxial_vs_exact.py`
raises it explicitly: the fixed calibration is a 2-D affine, which **cannot** express
a relative tilt; exact triangulation then shows a falling height-vs-radius profile
that the paraxial formula does not. A tilt constrained by the measured vertical
disparity (`|pitch| ≤ 0.3°`) leaves 33.8–39.8 % of the trend, which is what forced
the search to the lens model. Distortion and tilt are **degenerate on radial
flatness** and only the tilt bound separates them.

**Falsifiable prediction.** Applying `a = −0.30` jointly with a refitted roll:
1. the within-frame height IQR falls by >20 %;
2. the **frame median moves by <3 %** — distortion is a radial term whose
   contribution changes sign across the field, so it cannot by itself explain the
   ~600 m level gap;
3. the level anti-correlation changes by less than the block-bootstrap CI width
   (|Δr| < 0.15 at n = 470);
4. the optimal roll moves by <0.1°, because a rotation about the principal point
   commutes with any purely radial map.

If instead the median moves >10 %, the radial trend was aliasing into the level
through the match-radius distribution, and distortion is a **level** term — which
would be a major reclassification.

**Data and method.** The per-match rays in `experiments/layers/*`; the geometry
model in `src/clouds/geometry/` (owned by W2 — this plan specifies the change, it
does not make it). Refit `(a, roll)` jointly on the flatness criterion, then re-run:
radial flatness, ray-intersection residual, within-frame IQR, frame median, mode
count, and the regime shift of §1.2. Ground truth untouched.

**Effort.** M.

**Risk.** Fitting `a` and the roll together on the same flatness criterion that
motivated them risks overfitting the field. Mitigate with a temporal hold-out
(tune on 12:00–14:00 MSK, report on 15:00–17:00 MSK, within the one session), and
require `a` to be stable across frames (the roll was stable to 0.10°).

**What kills it.** If no single `a` flattens the profile under **both** exact
triangulation and the roll correction — i.e. the flattening depends on which radius
range is used, or `a` is unstable across the session — then "distortion" is a
misfit and the radial trend must be attributed to the affine's inability to express
tilt, with the deliverable reporting an uncalibrated radial term.

### R3 — The evaluation currency: Tier-1 composite plus a spatial-sampling model *(rank 3)*

This is not a research direction so much as the measuring stick every other
direction needs, and it is ranked third because without it R1, R2 and R4 have no
acceptable acceptance criterion.

**Part A — the Tier-1 composite.** Pre-register, before use: radial flatness
(slope of binned median height vs radius), cross-region agreement (disjoint sectors,
uncertainty-aware), ray-intersection residual, temporal coherence (a stated motion
bound), and layer stability — each with a rejection count and a coverage figure so
a metric cannot be improved by rejecting frames (G5, G6). Selection uses Tier 1
only (G1). A change that improves Tier 3 but not Tier 1 is rejected (G8).

**Falsifiable prediction.** Distortion + mask should improve radial flatness and
leave cross-region agreement not worse; a rule change that improves the level by
increasing rejections is visible as reduced coverage and is rejected.

**Risk.** Tier-1 metrics can be gamed by over-smoothing or by selection. Mitigate by
reporting the pair (metric, coverage) and by a Pareto rule rather than a scalar
score if two metrics trade off.

**Part B — is any point-vs-footprint comparison defensible?** The documented
numbers: at 1837 m the footprint is 1956 m across against the ~10 m ceilometer
column (≈200:1), spanning 8.5 texture decorrelation lengths of 229 m, advancing
9–45 of them per 30-min window; within-frame relief is ~12 % of the height.

**Hypothesis.** There is no defensible *point-vs-footprint level* comparison;
the only defensible external comparisons are (i) distributional (Tier 2) and (ii)
an **envelope** test: does the reference's point series lie inside the band of
footprint-mean values the cloud field could plausibly produce?

**The measurement that decides it.** Compute, per frame, the *spatial* standard
deviation of corrected per-match heights (the footprint's internal variability) and
compare with the *temporal* standard deviation of the footprint median over the
session. If spatial ≥ temporal, a point reference cannot track the footprint, and
the envelope test is the only fair external statement. If spatial << temporal, the
footprint is homogeneous and a level comparison becomes defensible — which would
change the plan.

**Data and method.** `experiments/layers/heights.npz` (167 frames, per-match heights
and coordinates), plus the divergence term already measured in
`temporal-alignment.md` §5. Effort L.

**Falsifiable prediction.** The spatial s.d. exceeds the temporal s.d. of the
footprint median by a factor >2, so no single-number level comparison should be
quoted as validation.

**What kills it.** If the spatial s.d. is small compared with the temporal s.d., the
spatial-mismatch excuse disappears and the level disagreement becomes a hard defect
that R1/R2 must fix.

### R4 — The level offset, decomposed against corrected geometry *(rank 4)*

**Known arithmetic** (all traceable): `f = 654 ± 11` implies heights are ~6–7 % low
(×1.070, multiplicative, height-independent); the roll leaves the median almost
untouched (a ~1 % effect on the median disparity, findings §4); the distortion
leaves the median essentially unchanged; after the focal correction the estimate is
~18 % below the lower reference layer (~1966 m vs ~2400 m) and ~48 % below the
upper.

**Hypothesis.** After R1–R3 the offset reduces to one signed term: the geometric
cloud boundary sits *below* the ceilometer's backscatter threshold, so
stereo < ceilometer by a cloud-type-dependent amount (the mechanism in
literature.md §1, attributed in the AMT discussion to unmodelled 3-D cloud-side
geometry).

**Falsifiable prediction.** The reference sits *above* the stereo by a non-negative
amount in **every** regime (the geometric cloud boundary is at or below the
ceilometer's backscatter threshold), the magnitude is <15 % of the height, and the
offset does not change sign between regimes. The current data fail this (the stereo
reports *higher* when the reference layer is *lower*, §1.2), so the present offset
is a target error, not a physical offset.

**The test that is unavailable.** The intended discriminator — offset co-varying
with backscatter width `L` — needs `L`, which is absent (§1.4, §4). Substitutes are
weak: `Q` (13 discrete levels, confounded with detection) and the within-frame IQR.
Any offset-vs-`Q` result must be declared a characterisation against a confounded
covariate, not a mechanism test (G3).

**Data and method.** Combine R1–R3 outputs; report the offset per regime with a
block-bootstrap CI and per height bin. Effort M.

**What kills it.** If the offset changes sign between regimes after R1–R3, it is not
an optical-vs-geometric term and the deliverable must report an uncalibrated level
with a stated target ambiguity.

### R5 — Multi-layer retrieval re-test under corrected geometry *(rank 5)*

**Already measured** (`experiments/layers/layer_analysis.py`, findings §12): on 167
frames and 406 289 exact-triangulation matches, **80 % of frames have one
significant mode, 20 % have two, mean 1.20 modes**; the primary mode sits at a
median 1675 m with p10–p90 1525–1845 m for five hours. The single modality is
therefore **not** a clustering artifact — it is in the per-match distribution.

**Hypothesis.** The single mode is real *as a property of the matched population*,
but the population is texture-selective: LoFTR locks onto the most contrasty cloud
available, which need not be the reference's lowest layer. The clear-sky control
supports the population being cloud-driven (2025-05-22 gives single-digit
correspondences per hour against 1100–3700 on 2025-05-23, `w1-calibration.md`
§W1.14), so the matches are cloud, not scene.

**Falsifiable prediction.** After R2 (distortion) and R1 (masks), the mode count
stays ≤1.5 and the primary mode stays below 2000 m; if instead two modes emerge near
the reference's 2400 m and 3800 m, the single-mode finding was geometry-limited and
must be withdrawn.

**Data and method.** Re-run `layer_analysis` on the corrected geometry with the
occluder mask; add mode persistence and support; report the mode-vs-`H2`
distributional comparison when `H2` exists (267 window samples, R6). Effort M.

**What kills it.** If every geometry fix leaves the mode structure unchanged, the
single mode is a property of the matcher's selectivity, and multi-layer retrieval
from a 45 m-baseline rig at this cadence is not supported — report single-layer with
a stated selection bias (G10).

### R6 — Reference refinement with `H2` and `Q` *(rank 6)*

**Hypothesis.** `H2` (present in 29 % of `H1`-valid samples) supports a cleaner
regime conditioning than a fixed 2750/3500 m split, and the stereo's occasional
second mode corresponds to `H2`.

**Falsifiable prediction.** Conditioning the comparison on `H2` rather than `H1`
does not raise the differenced correlation above its block-bootstrap CI, and when
the stereo has two modes its upper mode is closer to `H2` than its primary mode is
(Tier 2, distributional). `Q` gives no additional vocabulary because `Q = 0`
coincides exactly with no detection.

**Data and method.** The DOL-2 reader in `src/clouds/sources/ceilometer.py`
already exposes `h1`, `h2`, `backscatter_l`, `quality`. Effort L.

**What kills it.** If the `H2`-conditioned comparison is not distinguishable from
the circular-roll null, record the null result and stop (G10) — do not re-tune the
regime boundaries.

### R7 — Motion-based internal consistency (sector shear) *(rank 7)*

The imagery gives a 10 s-cadence motion field, already measured: median 3.27 m/s,
image azimuth 213° ± 9°, single-plane fit rms median 3.88 px / p90 8.10 px over 1798
pairs — the rms being comparable to the displacement at some times, consistent with
multiple layers or rapid evolution.

**Hypothesis.** Vertical shear is detectable: two layers moving at different speeds
or directions must make the single-rigid-plane fit fail, and the failure must
co-occur with height bimodality.

**Falsifiable prediction.** Split the field into angular sectors and fit the
translating-plane model per sector. If at a fixed time the sector motions disagree
by more than their bootstrap CI, the layer is not rigid; and those frames must have
a higher per-match height IQR / mode count than frames with consistent sector
motion. If sector disagreement is uncorrelated with height bimodality, the motion
field is dominated by estimator noise and this direction is dead.

**Data and method.** `experiments/temporal_alignment/q2_cloud_motion.py` already
implements the plane fit and its controls; extend to per-sector fits on the same
1798 pairs. Ground truth untouched. Effort M.

**Advection: mostly closed.** The instrument separation is unrecorded, so the lag
`= separation / v` is a function, not a number (`temporal-alignment.md` §3). The
best-shift advection fit gives R² = 0.099 (p = 0.18), the shift sequence jumps
200–250 s between adjacent windows, and a 2–3 px calibration change moves it by
168 s on average. The only defensible use is as a **bound**: at 3.27 m/s the
footprint advances 9–45 decorrelation lengths per window, so a point and a footprint
series cannot be expected to correlate. Do not build an advection correction and
then validate it against the ceilometer — the shift would be fitted to the answer
(G1, G3).

### R8 — The calibration margin: `f`, tilt, translation as declared nuisance terms *(rank 8)*

**Status.** `f` is bounded to 642–665 px/rad (through-origin, four Sun streams,
spread 3.5 %), ratio 1.070 ± 0.018 to the assumed `ξ₀/π = 611.2`, but the
Sun-blob centroid cannot resolve calibration below roughly 4° (~30 px at the crop
corner) — about 30× coarser than the ~1 px the margin needs. The measured roll is
−1.57° ± 0.10° from a scene-based fit and corroborated three ways. The margin
arithmetic is written down: 10 % absolute height at 1.5–3 km needs calibration good
to ~1–2 px.

**What remains genuinely open.** (i) The projection *shape* (equidistant vs
equisolid vs stereographic) is largely settled by the distortion term but should be
fitted jointly with `a`. (ii) The **tilt/translation** terms the affine cannot
express: a target-free relative-pose estimate from the **static foreground**
essential matrix (Crispel & Roberts' route) would bound them independently of the
Sun, and the near-field foreground gives large disparity and therefore a
well-conditioned pose. That is a high-value, moderate-effort measurement, but it
serves R2/R4 rather than being a standalone research direction.

**Falsifiable prediction.** The foreground essential matrix yields a relative
rotation within 0.2° of the affine + measured roll and a translation direction
consistent with `(b, 0, 0)`; if it does not, the fixed affine is wrong by more than
the margin and the deliverable's absolute height is not defensible.

**What kills it.** Near-field foliage moves in wind and the two cameras are ~1 s
apart, so the "static" foreground may not be static. If the essential-matrix
rotation is unstable across frames, report it as a failed route and rely on the
Sun-based bound.

---

## 4. What the data cannot support — directions that must not be attempted

Each item is a direction that would look reasonable from the literature and is
impossible with this evidence. Stated with the number or document that closes it.

1. **Anything requiring a second session.** 2025-05-22 is cloud-free and yields
   single-digit correspondences per hour (`w1-calibration.md` §W1.14). No
   cross-session drift, no tuning/validation split by session, and **no "unseen
   session" runtime validation** (plan W6's falsification test cannot be satisfied).
   Any held-out block is a time block of the same day and is correlated with the
   tuning block.
2. **Confirming the anti-correlation on an independent period**, as
   `evaluation.md` §3 asks. There is no independent period. The honest substitute is
   a pre-registered, gap-safe protocol with a block-bootstrap CI, and the limitation
   stated.
3. **Real meteorology.** `meteo1`, `meteo2`, `mtp`, `rvp` cover 2023–2024 only; there
   is no wind, temperature or radiosonde data for 2025-05-23. Cloud motion must be
   inferred from the imagery, and its conversion to a compass bearing depends on a
   rig handedness convention that was never independently verified
   (`temporal-alignment.md` §6.6).
4. **Site coordinates or instrument separation.** None exist anywhere in the
   repository (`temporal-alignment.md` §6.1). The advection lag can be bounded but
   never computed, and the ceilometer's column can never be located inside the
   camera footprint. This is what makes the spatial mismatch unfixable rather than
   merely large.
5. **A survey-grade reference.** There is no calibration target, no surveyed point,
   no collocated second ceilometer and no lidar/radar. The only absolute geometric
   reference is the Sun, and the Sun-blob estimator is ~30× too coarse; the
   disc-detector that would fix it is an open, hard task (`w1-calibration.md`
   §W1.10). Absolute scale therefore rests on `f` (±1.8 %, 1σ) plus an unmeasured
   calibration residual.
6. **Any use of the ceilometer's backscatter width `L`.** `L` is populated in
   **2 of 5759 slots** on 2025-05-23 and **0 of the 906 validation-window
   detections** (§1.4). The literature-specified test of the optical-vs-geometric
   offset against `L` cannot be run. Do not design an experiment around it.
7. **Anything needing the fixed calibration to be re-derived.** It is fixed by
   constraint (`README.md` §2); the project may measure residuals and apply them as
   declared corrections, never change the matrix. `w1-calibration.md` shows the
   distinction and the roll correction is the worked example.
8. **Star- or Moon-based calibration.** Night frames are pure noise at the
   acquisition exposure and there is no Moon in them (`w1-calibration.md` §W1.4,
   findings §7). Closed, recorded, not to be retried.
9. **Height accuracy above ~3 km, and any claim of "10 % at 5 km".** 1 px of
   disparity error is 5.5 % at 1.5 km and 10.9 % at 3 km; disparity is 18.3 px at
   1.5 km and 9.2 px at 3 km on a 45 m baseline (`stage-audit.md` §1.1). The
   structural limit is the baseline and no downstream method removes it.
10. **Runtime-inference validation on unseen data.** G13 requires the deliverable to
    run end-to-end on a session it has never seen; with one usable session this
    cannot be demonstrated. The closest honest substitute is a held-out time block
    plus the internal consistency checks, with the limitation recorded
    (`w1-calibration.md` §W1.14).

---

## 5. Ordered sequence of work

The order follows the ranking and respects the dependency that R4 (level) is only
meaningful once R1–R3 exist.

```
R1  target identity + wrong-sign mechanism      ─┬─► R4  level decomposition
R2  distortion in a joint geometry refit        ─┤
R3  Tier-1 composite + spatial model            ─┘
R5  multi-layer re-test            (needs R1, R2)
R6  H2/Q reference refinement      (needs R5)
R7  sector-shear motion test       (independent)
R8  foreground essential matrix    (feeds R2, R4)
```

### Step 1 (first, precisely)

**Title.** Gap-safe protocol and target-identity diagnosis on stored artifacts.

**Inputs.** `experiments/layers/heights.npz` (167 frames, per-match exact-
triangulation heights + image coordinates, roll-corrected); the baseline cluster
dump `20261013T092414Z_evaluate_clouds_2025-05-23_63dbc1/clusters.jsonl`; the
roll-corrected `series.json`; `Zve_Sci_250523.txt`.
**Do.** (a) Adopt and record the gap-safe correlation protocol: 30 s grid, only
genuinely adjacent pairs, block bootstrap at a stated block length; recompute the
level and difference correlations and publish both. (b) On the per-match heights,
compute for each frame the quantile family (min, 10th, 25th, median, mode) and, for
each, the regime shift, its block-bootstrap CI, the within-frame IQR and the match
radius. (c) Restrict to `r ≤ 250 px`, then `r ≤ 200 px`, and repeat.
**Outputs.** A JSON under `experiments/planning/` with those numbers; one figure of
per-frame estimator vs reference regime with the bootstrap band.
**Pre-registration (write before running).** Metric: regime shift of the per-frame
estimator, KS test between regimes, block-bootstrap CI. Failure condition: the
regime shift under a mode/quantile estimator is not distinguishable from zero while
the within-frame IQR is not reduced — that kills M1.
**Kill.** If no internal estimator (min, quantile, mode) removes the inversion while
improving Tier 1, move effort to R2 and the matcher, and record the null.

### Step 2 (precisely)

**Title.** Joint `(a, roll)` geometry refit and re-run of the internal criteria.

**Inputs.** The per-match rays and the geometry model; `f = 654 ± 11` as the baseline
constant; the roll optimum as the starting point.
**Do.** Implement `θ = (r/f)(1 + a (r/f)²)` in the ray-direction map used by
triangulation (the geometry module is owned by W2; this step specifies the change
and measures it). Refit `a` and the roll jointly on the radial-flatness criterion
with a temporal hold-out (tune on 12:00–14:00 MSK, report on 15:00–17:00 MSK).
Re-measure: radial flatness, within-frame height IQR, frame median, ray-intersection
residual, mode count, and the Step-1 regime shift.
**Pre-registered predictions** (from R2): IQR falls >20 %; median moves <3 %;
level anti-correlation changes by |Δr| < 0.15; optimal roll moves <0.1°. A median
move >10 % reclassifies distortion as a level term.
**Kill.** If no single `a` flattens the profile under both exact triangulation and
the roll correction, or `a` is unstable across the hold-out, attribute the radial
trend to the affine's inability to express tilt and report an uncalibrated radial
term.

### Step 3 (precisely)

**Title.** Occluder/field mask and selection-rule audit, on Tier-1 criteria only.

**Inputs.** The same per-match heights and coordinates; the occluder geometry in
`stage-audit.md` §5 (trees intruding to `r ≈ 400–500 px` on one side, a building and
mast at the bottom, a wire crossing the upper field; the crop is a 640 px square
whose corners reach `r = 452 px`).
**Do.** Apply a circular field mask plus an explicit occluder mask. Then compare
`lowest`, `largest`, size-weighted median, and dominant mode on Tier-1 criteria
(radial flatness, cross-region agreement, ray residual, temporal coherence, layer
stability) **and** report coverage and rejection counts for each. Do not consult the
ceilometer for the choice (G1, G4).
**Pre-registered prediction.** The mask removes the low tail near the top edge
(matches with `v < 64` have median height 1181 m vs 1911 m elsewhere), but §1.3
predicts it will *not* by itself fix the level: excluding the top 128 px lifts the
selected median only to 1463 m. Failure condition: after masking, the selected-lowest
median changes by less than its across-frame IQR (88 m) — the low tail is then not
localised, and the selection rule rather than an occluder is the cause.
**Kill.** If the low cluster persists after masking and after R2, the "lowest"
rule's tail is intrinsic to the match field; report it and move to the matcher.

---

## 6. Pre-registration record

Before each step's first run, append to this section: the metric, the period, the
config, the exact grid/protocol, the null, the significance threshold, and what
counts as failure (G2). A metric chosen after seeing a result is not evidence.

| step | date written | metric | period | null | failure condition | status |
|---|---|---|---|---|---|---|
| Step 1 | — | regime shift of per-frame estimator; KS; block-bootstrap CI | 2025-05-23, 12:00–17:00 MSK | circular-roll / block bootstrap | shift not distinguishable from zero, IQR unchanged | not started |
| Step 2 | — | radial flatness, within-frame IQR, frame median | tune 12:00–14:00, report 15:00–17:00 MSK | `a = 0` | no single `a` flattens; `a` unstable | not started |
| Step 3 | — | Tier-1 criteria + coverage | 2025-05-23 | unmasked configuration | median immobile after masking | not started |

---

## 7. The honest summary

The project's papers say it has a method with a level offset. The measurements in
this workplan say something more specific and less comfortable: **the estimator
reports a low-tail order statistic (≈1156 m) of a heavy-tailed height field whose
level is nearly constant for five hours, while the reference moves between 2480 m
and 3930 m; the estimator's small regime-dependent response is real but inverted
(−105 m to −154 m), and its frame-to-frame changes are uncorrelated with the
reference once the differencing is made gap-safe.**

That recasts the central question from "how do we correct the offset" to "what is
the matched population, and does any internal estimator of it measure the cloud
base". R1–R3 are the measurements that answer it; R4 can only be attempted after
them; R5–R8 are bounded by the data limitations in §4. The one session is the
binding constraint on everything that involves validation, and the absent `L`, site
coordinates and meteorology are the binding constraints on every mechanistic
explanation of the level.

---

# R2 — DONE (2026-10-13)

**Applied the measured lens distortion end-to-end and the pre-registered prediction
held.** Implemented as `camera.distortion_a` and `camera.exact_triangulation`, both
defaulting to the historical behaviour so every earlier result stands
(`src/clouds/geometry/triangulation.py`, `src/clouds/pipeline/analyze.py`,
`configs/ablations/_distortion.yaml`).

Three 300-frame runs on 2025-05-23 with RoMa, differing only in the geometry. The
metric is the pipeline's **own** per-cluster heights binned by distance from the
principal point (162,000 clusters each), summarised by the profile's interquartile
range — a flat cloud layer must give a flat profile:

| configuration | inner (0–100 px) | outer (350–450 px) | ratio | profile IQR |
|---|---|---|---|---|
| as shipped (paraxial, ideal lens) | 1911 m | 1707 m | 0.89 | 4.1 % |
| exact triangulation, ideal lens | 1815 m | 1266 m | 0.70 | **19.1 %** |
| **exact + distortion a = −0.30** | 1873 m | 1782 m | **0.95** | **2.4 %** |

- **Flatness improves from 19.1 % to 2.4 %** — an 87 % reduction, against a
  pre-registered threshold of >20 %. ✓
- **The median moves −2.0 %**, against a pre-registered threshold of <3 %. ✓

## The finding that came out of it

**Exact triangulation without the distortion is *worse* than the paraxial formula**
(19.1 % against 4.1 %). The paraxial approximation's off-axis bias and the lens
distortion's radial effect have opposite signs and partially cancel. Removing one
without the other destroys the compensation.

Two consequences for the plan:

1. **W2 and R2 must be applied together.** A "more correct" height formula on its own
   is a regression here, which is the kind of result that would be read as a modelling
   failure if only the endpoint were reported.
2. **This is why the coordinate-descent instinct is dangerous.** Improving one stage and
   measuring the same metric would have rejected exact triangulation — correctly, given
   the other error still present.

R2 therefore also partly answers R1: the field's radial structure was an instrument
signature, and the ~1000–1400 m within-frame scatter remains the dominant unexplained
error.

---

# Owner's additions (2026-10-13) — R9 to R12

Four testing directions added on review. They are not independent of R1; three of them
are ways of attacking the same question from different sides, and R10 is the sharpest
statement of it yet.

## R9 — Re-calibrate on the cloudless day

**The opportunity is real and was missed.** The Sun is the only object at infinity, and
the earlier attempt to use it for calibration failed because the detected blob is the
whole saturated *region* around the Sun, whose centroid is displaced by whatever bright
cloud sits beside it — the measured bias was large enough to make the method useless
(4° resolution against a 1 px target).

On **2025-05-22 the sky is cloudless**. There is no cloud glare, so the saturated region
should be the Sun plus its atmospheric glow alone, and therefore far more symmetric. The
session was dismissed as "no stereo signal" — correct for stereo, and wrong for
calibration.

**Hypothesis:** on the clear day the Sun-blob centroid is an accurate estimate of the
Sun's position, and the residual of the fixed calibration can be measured to well below
the ~4° floor that killed the method on the cloudy day.

**Falsifiable prediction:** the frame-to-frame scatter of the inferred Sun position on
2025-05-22 is substantially smaller than on 2025-05-23, and the calibration residual
measured from it is consistent between the two cameras and with the roll already
measured from the disparity field (&minus;1.6°).

**Data and method:** the Sun ephemeris and detector already exist
(`experiments/w1/sun_reference.py`); the clear-day frames need only be substituted.
Then fit the 2-D similarity (rotation, translation, scale) that maps camera 1's Sun
positions onto camera 2's, and compare it with the shipped affine.

**Effort:** low. **Risk:** the glow may be asymmetric even without cloud.
**Kill criterion:** if the clear-day scatter is no better than the cloudy day's, the
Sun-blob method is dead for this rig and that is the finding.

## R10 — Why is the estimate's variance so low when the cloud changes so much?

**This is the sharpest open question in the project.** It is not merely that the estimate
is offset; it is that the estimate is *almost constant* while the reference moves through
~1450 m, and no selection rule changes that.

There is a paradox here that R1 has stated only loosely:

| quantity | magnitude |
|---|---|
| within-frame scatter of the height field | **1000–1400 m** |
| between-frame variation of the reported height | **~130 m** |
| actual change in cloud base | **~1450 m** |

The field is broad and noisy, yet a statistic extracted from it is stable to ~130 m. A
broad, *changing* field should produce a less stable statistic, not a more stable one.
So either the field's shape barely changes, or the reported statistic is insensitive to
the part of the field the cloud controls.

**Candidate mechanisms, each with its discriminator:**

1. **A persistent artefact dominates the field**, so the low tail — which the `lowest`
   rule selects — is set by fixed structure rather than by cloud. *Discriminator:* does
   the selected cluster's *position* stay in the same image region across the whole
   session? The earlier probe found 22.6 % of frames selecting a cluster inside one
   60 × 60 px bin, which would be extraordinary for a moving cloud.
2. **The selected height is an order statistic whose level is set by the field's spread
   and the match count**, both of which are stable. *Discriminator:* regress the reported
   height on the field's spread and the match count; if they explain it, the retrieval is
   reporting a property of the estimator, not of the sky.
3. **The Kalman filter removes what little variation there is.** The raw series has a
   28 m median jump and the filtered series 6 m; but both are far below 1450 m, so
   filtering alone cannot be the whole cause. *Discriminator:* compare raw variance to
   filtered variance — already partly known, so this is a control, not a candidate.
4. **The cloud genuinely does not change over the window** and the reference is
   reporting a different layer. *Discriminator:* the reference's own regime structure —
   it switches between two populations in long stretches, which is not a cloud holding
   still.

**Effort:** low to medium, and it needs no new data.

## R11 — Temporal verification

Never attempted. The question is not "does the series correlate with the ceilometer"
— that has been answered, and the answer is no — but **does the estimate behave
coherently in time for a single identifiable cloud?**

- Track one cloud through consecutive frames and ask whether its estimated base is
  *consistent*: does the same cloud keep the same height, and does it change only when
  the imagery says it should?
- Use the measured cloud motion (median 3.27 m/s at the estimated height, azimuth ~213°)
  to predict where a feature should be in the next frame, and check that the height
  attributed to that track stays continuous.
- This is a Tier-1 internal criterion: it needs no reference at all, so it has thousands
  of independent samples rather than the ~19 the filtered output supports.

**Discriminator for R10:** if a single tracked cloud's height is stable while the cloud
visibly evolves, mechanism 1 or 2 is confirmed. **Effort:** medium.

## R12 — Spatial verification

Also never attempted, and the owner's framing is the right one: **for one distinct,
clearly bounded cloud, the estimated base height should show a realistic spatial
variance** — the base of a real cumulus is not a plane to within a few metres.

Today that check is vacuous, because the field is nearly planar by construction and the
per-frame estimate carries almost no variance of its own. So R12 is conditional on R10:
once the low variance is explained or fixed, the spatial distribution of heights across
a single cloud becomes a *test* rather than a *tautology*.

Concretely, once R10 is resolved:

- segment one cloud in the image, and report the distribution of per-match heights
  inside it — mean, spread, and how the spread compares with the disparity uncertainty;
- compare that spread against the within-frame scatter of the *whole field*: if one
  cloud's spread is a small fraction of the field's, the field is not one cloud and the
  retrieval is mixing objects;
- check that the distribution is unimodal inside one cloud. A multi-modal distribution
  within a single visually distinct cloud would mean the heights are not measuring the
  cloud surface.

**Effort:** low, once R10 is answered. **Blocked on:** R10.

## Order of work, revised

R9 is cheap and could change the height scale for everything downstream, so it runs
first. R10 is the crux and needs no new data, so it runs second and in parallel with
R11. R12 follows R10. R2 is done; R1 remains the framing question that R10, R11 and R12
each attack from a different side.

---

# R9 — RESULT (2026-10-13): half-supported, and useful either way

`experiments/w1/r9_clear_day.py` ran the same Sun measurement on both sessions.

| | clear (2025-05-22) | cloudy (2025-05-23) |
|---|---|---|
| frames used | 46 | 42 |
| median saturated-blob area | **14,970 px** | 6,266 px |
| azimuth residual, median | **0.73°** | 1.51° |
| azimuth residual, p90 | **1.34°** | 4.69° |
| radial fit rms | 74.7 px | 67.5 px |
| through-origin `f` | **683.3 px/rad** | 662.8 px/rad |

## What holds

**The azimuth — the orientation measurement — is 3.5× tighter on the clear day**
(p90 1.34° against 4.69°). The hypothesis was right about the mechanism: with no cloud
glare the saturated region is more symmetric about the Sun, so the centroid's *direction*
is a much better estimate. For measuring orientation, R9 works.

## What does not

**The radial measurement is not better, and the blob is 2.4× LARGER, not smaller.** The
prediction was that removing cloud glare would shrink the saturated region. It does the
opposite: with no clouds to occlude or dim the sky around the Sun, the glow saturates a
*wider* area (14,970 px). The centroid's *radial* position is therefore biased as much as
before, and the radial rms is unchanged (74.7 against 67.5 px).

So the failure mode is not cloud glare — it is that the saturated region around the Sun
is large under any clear-ish sky, and its centroid is a poor estimator of the Sun's
*radius* even when it is a good estimator of its *azimuth*. The two coordinates have
different quality, which the earlier attempt treated as one number.

## An independent result worth more than the test

Both sessions give a through-origin focal constant of **663–683 px/rad** against the
pipeline's assumed `ξ₀/π = 611`. Combined with the earlier four-stream fit (654 ± 11) and
the image-circle measurement (686–736), **five independent estimates now sit between 654
and 736, and none is near 611.**

The conclusion that heights are ~10 % low no longer rests on any single method. It is
worth restating because it is a scale error that no amount of downstream modelling
removes: it multiplies every height.

**R9 status: closed.** Orientation is better on the clear day and that measurement should
be used; the radial method is dead for reasons now understood, and the clear day does not
rescue it. R10 is next and needs no new data.

---

# R10 — RESULT (2026-10-13): the low variance has two causes, one of which is the filter

`experiments/layers/r10_variance.py`, 300 frames, selecting exactly as the pipeline does
(size &ge; 8, then lowest):

| quantity | value |
|---|---|
| reported (selected) height | median **1509 m**, std **176 m**, p5&ndash;p95 **574 m** |
| the field's own median | median 1911 m, std 165 m |
| reference H1 over five hours | std **743 m**, p5&ndash;p95 **1720 m** |
| Kalman-filtered estimate (same window) | std **91 m**, p5&ndash;p95 293 m |

## The answer, in two parts

**1. The raw retrieval under-responds by about 4&times;.** The per-frame selection moves
with a standard deviation of 176 m against the reference's 743 m — **24 % as much**.

**2. The Kalman filter then removes half of what is left.** Filtering takes the standard
deviation from 176 m to 91 m, a **48 % reduction**, leaving the delivered series at
**12 %** of the reference's variation.

So the flatness is not one defect but two multiplicative ones: a retrieval that responds a
quarter as much as it should, and a smoother that halves it again. Neither is visible from
the delivered series alone, which is why the earlier "the estimate is flat" statement was
true but not diagnostic.

## The two candidate mechanisms, adjudicated

**M2 — "the reported value is an order statistic whose level is set by the field's spread
and match count" — REFUTED.** Regressing the reported height on the field IQR and the
match count gives **R&sup2; = 0.001**, with coefficients of &minus;0.0002 m per metre of
spread and &minus;0.004 m per match. Both are indistinguishable from zero. The reported
level is not a property of the estimator's statistics.

**M1 — "a persistent artefact dominates the low tail the rule selects" — PARTLY
SUPPORTED.** The selection is concentrated: the most populated 60&times;60 bin holds
**14.5 %** of all frame selections, and that is **5.5&times; more concentrated than the
field as a whole**. The selected cluster sits at a **median radius of 350 px** in a
640 px crop — that is, near the periphery, in the annulus where the radial geometry error
was largest. This is consistent with the low tail being partly set by fixed structure
rather than by cloud, but it does not prove it, because a cloud with a low base genuinely
occupying one part of the sky would look the same.

## What follows

- **The filter is a candidate for removal or re-tuning on its own merits.** It halves a
  signal that is already 4&times; too weak. Its justification was smoothness, and
  smoothness of a signal that should be varying is not a virtue.
- **Re-run R10 under the corrected geometry.** The selected cluster sits at radius 350 px,
  exactly where the distortion correction (now applied, R2) has most effect. If the
  concentration and the under-response both come from edge geometry, this will show it.
- **The target-identity question is still open.** M1 is neither confirmed nor excluded,
  and R11 (temporal) is the discriminator: if a tracked single cloud moves in the imagery
  while its reported height does not, the selection is not following the cloud.

## An unresolved inconsistency to chase

The per-match height field and the per-cluster height field disagree by about 7&times;:
the pooled per-match interquartile range is ~1400 m, the per-frame per-cluster
interquartile range is ~197 m. Part of this is pooled-versus-per-frame, which the earlier
probe's own numbers do not fully explain. **Until it is reconciled, the width of the field
— which sets how much the low tail can vary — is not known to better than a factor of
several.** That is the next thing to pin down, because R11 and R12 both depend on it.

---

# RECONCILIATION (2026-10-13): the field width, and a correction to the record

The 7× inconsistency flagged in R10 is resolved, and the resolution **corrects an
earlier headline in the wrong direction — favourably.**

## What happened

The ~1400 m per-frame field width came from a **dump of the uncorrected configuration**
(LoFTR, no roll correction). The ~197 m came from a **dump of the corrected one** (RoMa,
roll corrected). Both are real; they are different instruments.

Isolating it on **identical matches** (`experiments/w1/matches_cache_2025-05-23.npz`, 38
frames, exact triangulation throughout):

| configuration | per-frame height IQR | median height |
|---|---|---|
| **uncorrected (as shipped)** | **689 m** | 1441 m |
| roll correction only | 341 m | 1716 m |
| distortion only | **792 m** | 1620 m |
| **roll + distortion together** | **235 m** | **1886 m** |

## What this says

- **The roll correction halves the height spread** (689 → 341 m). This is the same effect
  already measured on the disparity field, seen now in metres.
- **The distortion alone makes the spread slightly worse** (689 → 792 m) — the same
  compensation effect found in R2, now visible as a variance cost as well as a bias.
- **Together they reduce the spread 2.9× and raise the level 31 %.**

So the statement that "the dominant error is a 1000–1400 m within-frame scatter" was true
of the **uncorrected** instrument. With both corrections applied it is **~235 m** over all
matches, or ~197 m over the size-filtered clusters the pipeline actually selects from.

## The level moves too, and in the right direction

| | median height |
|---|---|
| uncorrected | 1441 m |
| roll + distortion | **1886 m** |
| ceilometer's lower layer | ~2400 m |

The gap to the lower reference layer **halves**, from ~960 m to ~514 m — before the ~7 %
focal correction, which would take the estimate to ~2019 m and the gap to ~380 m.

## Consequence for R10

The variance paradox was stated as "a broad, noisy field yielding an almost constant
statistic". That premise was built on the uncorrected field. With the corrections the
field is **2.9× narrower**, so the puzzle is smaller than it appeared: the estimator's
4× under-response to the reference remains, but the "broad noisy field" it must be
insensitive to is now ~235 m wide, not ~1400 m.

**R10's two conclusions stand** — the filter still removes 48 % of the raw variation, and
M2 is still refuted — but the framing must carry the corrected field width. R11 and R12
should be run **on the corrected configuration**, which is now the default.

---

# R11 — RESULT (2026-10-13): the estimator is dominated by a STATIC spatial pattern

`experiments/layers/r11_identity.py`. The test needs no reference: if the retrieval
assigns a height to *cloud*, the height map must move as the cloud moves; if it assigns a
height to a *place in the image*, the map is the same all session. The measured cloud
motion moves air through the field many times over in five hours, so a static map cannot
be explained by the cloud standing still.

**Correlation of the height map between the session's first and second halves** — two
disjoint sets of frames, ~2.5 hours apart, through which entirely different air has passed:

| configuration | r between halves | shuffled null | spatial spread | per-frame spread |
|---|---|---|---|---|
| roll correction only | **+0.981** | −0.002 | 173 m | 215 m |
| **roll + distortion** | **+0.912** | −0.002 | 92 m | 231 m |

## What this says

**A large part of the estimated height is determined by WHERE in the image a match lies,
not by WHEN it was measured.** The map is reproducible to r = 0.98 without the distortion
correction and r = 0.91 with it, against a null of zero. Applying the distortion halves
the static pattern's amplitude (173 → 92 m), so the distortion is roughly half of it —
**and the rest is something else.**

Also from the same run: the median per-cell *temporal* interquartile range is 63 m against
a *spatial* spread of 173 m, a space-to-time ratio of 2.8 (1.4 after the distortion
correction). Space dominates.

## Why this explains everything observed so far

One mechanism accounts for the whole set of failures found earlier:

- **the estimate is flat** — a static map has a constant average, so any per-frame summary
  of it is nearly constant, which is exactly the 130 m the delivered series moves against
  the reference's 1450 m;
- **no selection rule helps** — `lowest`, `largest` and median-of-candidates all sample
  the same static map, which is why all three are flat over five hours;
- **the inverted response** — the residual cloud signal sits on top of a pattern that does
  not move with the cloud, so its sign carries no information;
- **more matches do not help** — extra coverage widens the static pattern rather than
  adding cloud signal.

## The consequence for the project

**The estimator is not currently measuring cloud base height.** The dominant signal in its
output is an image-position-dependent artefact that is stable across 2.5 hours of
completely different weather. Until that pattern is characterised and removed, neither the
level offset, nor the layer question, nor any comparison against the reference can be
interpreted.

**The static pattern is a calibration signature, and this makes R8 the priority.** A
spatially fixed height pattern is what a spatially varying calibration error produces —
the distortion was one component, and the 2-D affine's inability to express a relative
tilt is a candidate for the rest. R12 is now conditional on R8, not R10.

## R12 — how it changes

The owner's spatial check ("for one distinct cloud we should expect realistic variance in
its base") is now the *same* test as R11 run at one instant instead of across the session.
It remains the right check, and it now has a name: it detects exactly the artefact R11
found. It should be run after R8.

---

# R8 — FIRST RESULT (2026-10-13): the static field is large, non-radial, and not plane geometry

`experiments/layers/r8_static_field.py`, 167 frames, 406,289 matches. The static part of
the disparity field is the part that does not change while the cloud moves, so it is not
cloud — it is instrument.

## What the static field looks like

| quantity | value |
|---|---|
| static disparity field | mean **15.22 px**, std **0.84 px**, range 14.02–17.37 |
| per-cell temporal IQR | median **1.17 px** |
| static spread / temporal spread | **0.71** |

## What geometric form it takes

Fitting the static field to candidate error models, as a fraction of the field's own
0.84 px standard deviation:

| model | rms residual | explains |
|---|---|---|
| constant | 0.837 px | 0 % |
| **linear in u** | **0.486 px** | **41.9 %** |
| linear in v | 0.813 px | 2.9 % |
| radial (r) | 0.726 px | 13.3 % |
| radial (r², r³) | 0.725 px | 13.4 % |
| roll | 0.817 px | 2.4 % |
| **radial + linear + roll** | **0.168 px** | **79.9 %** |

**The static error is dominated by a LINEAR GRADIENT IN u, not by a radial term.** Radial
terms alone explain only 13.5 %. The full model explains 79.9 %.

The linear coefficient is **+0.00394 px per px**, which over the 640 px crop is a
**+16.6 % change in disparity end to end** — the signature of a horizontal **scale** error
between the two cameras, not a translation and not a distortion.

## The field is inconsistent with a plane

The exact geometry of a horizontal plane at the median height predicts a disparity that
*decreases* outward; the observed static field *increases* outward. Subtracting the plane
model leaves a residual of **+1.88 px mean, 1.57 px std — about 11 % of the disparity**,
larger than the static field itself. So the observed field is not a plane seen through a
distorted lens; it is a plane seen through a *misaligned* pair with an additional
position-dependent error.

## What this means, and what it does not yet pin down

- **A ~16 % horizontal scale error across the crop is the single largest identified
  defect.** At 1.5 km, where the whole disparity is 18.3 px, a scale error of this size
  moves the height by more than 15 % from one side of the image to the other.
- **It is not the lens distortion.** The distortion is radial; this is linear in u.
  Applying the distortion already removed about half the static pattern's amplitude
  (R11), which is consistent: distortion is present, and there is a separate scale error
  on top of it.
- **The residual's sign structure is not yet explained.** The plane-model subtraction
  leaves a residual *larger* than the field it was subtracted from, which means the model
  and the data disagree systematically rather than the model being nearly right. Either
  the assumed principal point is wrong, the two cameras differ in focal length, or the
  comparison is confounded by which clouds occupy which image regions. **These three are
  distinguishable and should be separated before any correction is fitted**, because
  fitting a correction to a confounded field would encode the confound.

**This is now the gating measurement for the project.** It is measurable directly — the
static field *is* the error field — and nothing downstream is interpretable until it is
accounted for.

---

# Temporal filter — UNWIRED (2026-10-13)

On review the Kalman filter is bypassed while the retrieval is diagnosed, because it was
measured to remove **48 % of the raw variation** of a signal already ~4× less responsive
than the reference. Confounding the retrieval's variability with a smoother's makes both
unreadable.

Implemented as `temporal.filter_enabled` (default `True`, so every recorded result
stands); `configs/ablations/_nofilter.yaml` disables it. Verified: with the filter on the
delivered series has std 91 m, with it off 182 m against a raw 176 m — i.e. the delivered
series is the selection, as intended.

---

# R8 — VERIFICATION OF THE "COORDINATE-FRAME BUG" CLAIM (2026-10-14)

A claim was raised that the pipeline has a coordinate-frame bug: `prepare_pair` warps
`img1` by the calibration affine (translation 1972.74 px) and `package_matches` then feeds
`kp1` to `heights_from_matches` with a common centre, so camera 1's rays are computed about
the wrong principal point — and that this explains R8's linear-in-u field.

**It was tested rather than accepted, because if it were true it would invalidate R8 and
R11.** `experiments/layers/frame_bug_test.py`.

## What is true

- **The code fact is confirmed.** `kp1` is in the affine-warped frame and is passed with
  `centre=(320,320)`, the crop centre.
- **The affine has a 54.34 px residual** beyond a rotation about the image centre:
  `t − (w − R·w) = (+54.10, +5.03) px`, which is **4.76° of angle** at f = 654 px/rad.
  |det R| = 1.000000, so the matrix is a pure rotation plus translation, and this residual
  is a real, unexplained offset.
- **The affine cannot separate the two principal points.** It fixes only the relation
  `c2 = R(c1 − w) + w + delta`; the absolute position of either is unobservable from it.
  The pipeline implicitly assumes both are at the crop centre.
- **A principal-point error does produce a static height gradient.** Rendering a plane
  with the rays anchored at `c2` and triangulating about the crop centre gives
  **−16.6 % across the crop at 1800 m** — the same amplitude as the observed static field.

So the concern identified a genuine structural gap.

## What is not true

- **It is not automatically a bug.** For a rectified pair a *common* centre is correct —
  that is what rectification means — provided the centre is the common principal point.
  The question is only whether the crop centre is that point.
- **The residual 54 px is not "a 52.74 px translation of camera 1".** The translation
  1972.74 px is what a 89.92° rotation about the ORIGIN produces by itself; a rotation
  about the image centre needs 1918.64 px. The difference, 54.10 px, is the meaningful
  quantity and has a different physical meaning than the claim assigns it.

## The decisive test: the proposed fix makes the static field WORSE

If the pipeline's centre were wrong, using the correct one should reduce the static field.
Applying the fix — centre = `c2` = crop centre + delta = (374.10, 325.03) in crop
coordinates — measured on the real 167-frame data:

| centre used for the rays | r between session halves | spatial spread |
|---|---|---|
| crop centre, distortion (**current**) | **+0.912** | **92 m** |
| principal point c2, distortion (**the proposed fix**) | **+0.939** | **115 m** |
| crop centre, no distortion | +0.981 | 173 m |
| principal point c2, no distortion | +0.982 | 169 m |

**The fix does not work.** It leaves the static field as strong or slightly stronger.

## What the evidence actually points to: a relative focal ratio

R8's linear-in-u term in the DISPARITY field is independent of any centre choice, because
`du = kp1 − kp2` and a common centre cancels in the difference. It is therefore not a
principal-point effect at all. A linear-in-u disparity is the signature of a **scale**
difference between the two images, and the fitted slope (+0.00394 px/px) implies a
**relative focal ratio of ~0.4 %**.

Scaling camera 2's coordinates and measuring the static field directly:

| scale applied to kp2 | r between halves | spatial spread |
|---|---|---|
| 1.0000 (current) | +0.912 | 99 m |
| 1.0010 | +0.884 | 88 m |
| 1.0020 | +0.854 | 80 m |
| **1.0030** | **+0.834** | **77 m** |
| 1.0039 (R8's independent estimate) | +0.835 | 79 m |
| 1.0050 | +0.861 | 85 m |
| 1.0080 | +0.937 | 121 m |

**Two independent estimates of the same physical parameter agree**: the R8 linear-u fit
gives 0.394 %, and minimising the temporal instability gives 0.30 %, with a clear minimum
and a smooth rise on both sides. That is a genuine cross-validation, not a fitted
parameter.

**But the static field survives it** — r falls 0.912 → 0.834 and the spatial spread
99 → 77 m, a 22 % reduction, leaving the pattern clearly dominant. So a relative focal
ratio is **part** of the static field, worth about a fifth of it, and something else
remains.

## Conclusion

- **The proposed coordinate-frame fix is rejected on measurement.** It is recorded here
  with its numbers so it is not re-proposed.
- **A relative focal ratio of ~0.3–0.4 % is confirmed by two independent methods** and
  should be applied — it is a real, physically meaningful calibration parameter that the
  pipeline currently ignores (it forces scale 1.0).
- **~78 % of the static field remains unexplained**, and it is not the principal point and
  not the focal ratio.

## A caution this raises for the plan

The claim arrived with a specific mechanism, a specific line range, and a specific
predicted magnitude that matched the observation. It was still not the cause. A prediction
that matches a magnitude is not evidence of the mechanism — the same magnitude here had
**three** candidate explanations (principal point, focal ratio, cloud climatology), and
only direct testing separated them.

---

# STEPS 3 AND 4 — IMPLEMENTED (2026-10-14)

## Step 3 — circular field mask, foreground mask, measured focal constant

Implemented as three explicit `CameraCfg` knobs (`f_px_per_rad`, `field_radius_px`,
`mask_foreground` / `foreground_radius_px`), all defaulting to the historical behaviour
so every recorded result stands. The masks are applied to the correspondences before
clustering, and are the identity when disabled (verified). Config:
`configs/ablations/_masks_f654.yaml`.

## Step 4 — `lowest_mode`: a prominence rule instead of a minimum

`analyze.lowest_prominent_mode`. The shipped chain runs
`HDBSCAN(min_cluster_size=3)` and takes the lowest size≥8 fragment, which reports an
extreme low-tail **order statistic** of the measurement noise — it depends on how many
points there are, so a larger sample yields a lower minimum. That is the mechanism behind
the inverted regime response. The new path bypasses clustering and the 300 m merge
entirely, weights each match by the inverse ray-intersection residual, and reports the
**lowest prominent mode** of the height distribution: a mode must carry ≥10 % of the
weight and ≥50 matches.

Its defining property, verified on a synthetic two-layer field with a thin low tail
(60 % at 2400 m, 30 % at 3900 m, 40 outliers at 900 m):

| | value |
|---|---|
| lowest prominent mode | **2398 m** — the 900 m tail correctly ignored |
| the same field with 3× the points | **2394 m** |
| what a minimum would do | fall, by construction |

## Measured effect, 300 frames, 2025-05-23

| configuration | raw median | raw std | frame-to-frame jump (median) |
|---|---|---|---|
| lowest, f=611, no masks | 1550 m | 272 m | 53.0 m |
| lowest + masks + f=654 | 1705 m | 336 m | 31.6 m |
| **mode + masks + f=654** | **1994 m** | 247 m | **12.9 m** |
| *reference H1 over the hour* | *3150 m* | *649 m* | — |

- **Frame-to-frame stability improves 2.4×** (31.6 → 12.9 m) — a Tier-1 criterion needing
  no reference.
- **The level rises 17 %** (1705 → 1994 m), narrowing the gap to the reference from 1445 m
  to **1156 m**.
- **8 of 300 frames (2.7 %) produce no prominent mode** and abstain, rather than reporting
  a fragment. That is the intended behaviour of a prominence rule, not a failure.
- The raw spread falls (336 → 247 m) while the stability improves — the two move together
  only because the mode is not an extremum.

## What this does NOT do

It reduces the static-field problem; it does not remove it. The estimator is still flat
relative to the reference (std 247 m against 649 m), and R8's ~78 % unexplained static
component is untouched by Steps 3 and 4. Those are selection and calibration changes
respectively, and only the calibration side can address the static field.

---

# W2 — RESULT (2026-10-14): the foreground control is NOT available on this data

The advisor's directive proposed using the rigid foreground (mast, building, roof panels)
as a cloud-independent geometric control to attribute the R-15/R-16 static pattern to
intrinsics, extrinsics or cloud climatology. The experiment was implemented to the
directive's protocol — pre-registration, gates, fail-fast — and **halted at its first
gate**.

## Outcome: `INSUFFICIENT_FOREGROUND`

| gate | condition | outcome |
|---|---|---|
| G-A foreground | ≥50 inliers, epipolar RMS < 2.0 px, on ≥5 frames | **FAILED** |
| G-B sun comparison | — | not reached |
| G-C drift | — | not reached |
| G-D collapse | — | not reached |

## The measurement that failed it

Two matchers were tried. **SIFT**: 11 matches at ratio 0.75, 109 at 0.85 — almost all
ambiguous at any defensible threshold. **RoMa**: **zero** matches on the foreground band
at certainty 0.9, 0.7 or 0.5, and 35 at 0.2; on the building alone, **zero** at 0.5 or
0.3, and 246 at 0.15 with a disparity **interquartile range of 91 px** on a ~247 px
displacement.

**The measured foreground disparity is 87–247 px.** At that displacement the two cameras
view the building from ~20–40° apart, so local appearance changes too much for either a
descriptor or a dense matcher. The essential matrix built from it returned 26 inliers and
an impossible epipolar residual, which is how the failure first announced itself.

## Corrections to the directive, verified before implementing

| directive | measured |
|---|---|
| `img-{stamp}-devID{1,2}.jpg` | **no dash: `img-{date}T{HH}-{MM}-{SS}devID{n}.jpg`** |
| cameras desynced ~1–2 s | **median 0 s**, p90 0 s, max 5 s; 95 % of frames share a stamp |
| `dv_struct_rms_px` already exists in the pipeline | **no such metric or `epi` block exists**; it had to be built |
| the foreground is available as a control | **it is outside the 640×640 crop**, and when reached it cannot be matched (§ above) |
| "the affine is the wrong MODEL; the essential matrix is the correct model" | not well-posed: the affine is the rectification, the baseline enters triangulation separately, and the pipeline already triangulates with it |

## Why Phases 3–5 were not run

The pre-registered protocol halts at G-A. A cloud-collapse ratio computed from a
26-inlier degenerate essential matrix would have been worse than no number, and reporting
one would have violated the directive's own fail-fast principle.

## Consequence

**The R-15/R-16 static pattern remains a measured fact with an unattributed cause.** The
recommendation is to stop pursuing the foreground route on this dataset — the limit is
the 45 m baseline against a close, oblique foreground, not the choice of matcher — and
either obtain a controlled calibration target or instrumented session, or fall back to a
parameter estimate (the ~0.3 % focal ratio of R-16) which is a fit rather than a control.

Full record: `experiments/w2/PREREGISTRATION.md`, `experiments/w2/RESULT.md`,
`experiments/w2/foreground_pose.py`, `experiments/w2/foreground_pose.json`.

---

# W3 — RESULT (2026-10-14): both, and the inversion is not the calibration's

Continuation after W2 closed the foreground route. Full record:
`experiments/w2/W3_RESULT.md`.

| test | result |
|---|---|
| **H3 (climatic) vs H1/H2 (instrumental)** — partition the session by the ceilometer's own regime and compare the per-cell disparity maps | **BOTH.** Cross-regime r = +0.371 against a within-regime reproducibility ceiling of +0.718. Variance decomposes to **66 % instrumental, 34 % regime-dependent** |
| **does removing the instrumental component fix the sign?** | **NO — it strengthens the inversion** (−129 m → −164 m; ρ −0.547 → −0.581, p = 5.6e−16). The correction is not fitted to the reference |
| **is the inversion an edge effect?** | no — ρ = +0.540 at r ≤ 100 px and +0.547 at r ≤ 452 px |
| **is it a diurnal confound?** | no — partial ρ(reference, disparity \| hour removed) = **+0.498, p = 1.6e−11** |
| **what carries it?** | the **match count**: ρ(matches, disparity) = +0.559, ρ(matches, reference) = +0.425; the high regime returns ~2× the matches |
| **do the shipped configurations show it?** | **no** — all six give a shift of ≤30 m with p > 0.05. The low-tail statistics are flat and hide it |

## Consequences

1. **The directive's D2 hypothesis is refuted.** An essential-matrix pose cannot fix the
   retrieval: the inversion survives removing the entire instrumental component and grows
   when it is removed.
2. **The failure is in the matched population (R1), now measured.** The regime changes how
   many correspondences the matcher returns, and the extra population has a systematically
   different disparity.
3. **Calibration work should stop.** Two independent tests have now shown geometric
   corrections move the error, not the signal.
4. **The next experiment is about the matcher's population**, not the geometry: does the
   extra population in the high regime survive any internal consistency test?

---

# W4 — GATES 1 AND 2 (2026-10-14)

Full record: `experiments/w4/W4_RESULT.md`. Raw per-match evidence:
`experiments/w4/matches_roma.npz` (386 frames, 4,499,068 correspondences).

## A correction to W3.3

W3.3 reported "the high regime returns ~2× the matches (3630 vs 1751)". That was measured
on the **clustered** dump. On the raw matcher output the rate is **identical — 11,655.1 vs
11,655.8 per frame** — because RoMa emits a fixed dense grid. **The population size cannot
differ; only the disparity assigned to the same grid points differs.** The apparent
population change was created by the clustering, not by the matcher.

## Gate 1 — certainty refuted as the cause

Within every one of ten certainty deciles the high regime carries the higher disparity,
differences +0.82 to +2.50 px, all the same sign, weighted mean **+1.59 px** against a
pooled +1.80 px. And **the surplus population's median certainty is 0.782** — the extra
matches are *high* confidence, not low.

## Gate 2 — coherence refuted as the cause

Local residual against the 25 nearest neighbours: low **0.047 px**, high **0.041 px**,
ratio **0.865**. The disparity difference being explained is **+1.785 px**; the coherence
difference is **−0.006 px**. Both regimes produce equally smooth fields.

## What is now excluded

| candidate | status |
|---|---|
| certainty / hallucination | refuted |
| admitted noise / incoherence | refuted |
| calibration / geometry | refuted (W3) |
| edge or diurnal confound | refuted (W3) |
| population size | refuted — identical to 0.01 % |

**What remains is a target-identity mismatch or an unidentified physical effect.** The
matcher returns confident, coherent fields, and the high regime's sits 1.8 px (~350 m) away
in the direction of a *lower* cloud. Nothing internal to the matching explains it.

Gates 3 (epipolar residual) and 4 (LoFTR cross-check) are next.

---

# W4 — GATES 3 AND 4, AND THE MULTI-LAYER TEST (2026-10-14)

Full record: `experiments/w4/GATE34_RESULT.md`.

## Gate 3 — FAILS (my own tautology prediction was wrong)

| component | low | high | shift | KS p |
|---|---|---|---|---|
| along (du) | 12.273 px | 14.070 px | **+1.798 px** | ≈0 |
| **across (dv)** | 0.866 px | **1.885 px** | **+1.019 px** | ≈0 |

I predicted the gate would pass tautologically because a horizontal shift is along a
horizontal epipolar line. **The perpendicular residual nearly doubles**, so the high regime
satisfies the stereo geometry *less* well. Gate 3 fails.

**The decisive comparison:** the reference predicts a shift of **−5.315 px** (2250 → 3790 m
in parallax); the observation is **+1.798 px**. Opposite sign, 15 % of the magnitude. The
observed displacement corresponds to 2398 → 2092 m while the reference says 2250 → 3790 m.

## Gate 4 — same sign, 6.5× smaller

LoFTR: **+0.275 px, 95 % CI [+0.112, +0.401]**, bootstrapped over frames. Same direction as
RoMa's +1.798 px and the CI excludes zero, so the effect's *existence and direction* are
matcher-independent; its *magnitude* is representation-dependent.

## The multi-layer hypothesis — level explained, inversion not

| regime | camera | ceilometer H1 | difference |
|---|---|---|---|
| LOW (H1 < 2750) | **2361 m** | **2250 m** | **+111 m** |
| HIGH (H1 > 3500) | 2064 m | 3790 m | −1726 m |

**In the low regime the instruments agree to 111 m** — the first genuine level agreement in
the project. The camera's distribution (median 2244 m, IQR 522 m) sits inside the
ceilometer's lower population.

**But within the low regime the correlation is still −0.473 (p = 2.3e−13).** The layering
explains the LEVEL; it does not explain the INVERSION.

## Next

Not another consistency check. The direct question: **within the low regime, does the
camera's residual variation (~300 m) track anything in the reference at matched temporal
resolution, or is it below the instruments' agreement threshold?** And the level agreement
must be checked against the cloudless day, where nothing should agree.

---

# W5 — THE THREE CONTRACTS (2026-10-14)

Full record: `experiments/w4/W5_RESULT.md`.

## Premise correction
The desync hypothesis rests on a 1–2 s camera offset. **89.9 % (05-22) and 91.0 % (05-23)
of frames stamp both cameras identically**, median offset 0 s, maximum 5 s.

## 1. Cloudless Null — PASS
| | cloudy | cloudless |
|---|---|---|
| implied height | **2242 m** | **655 m** |
| median disparity | 13.13 px | **44.91 px** |
| median certainty | 0.0084 | 0.0012 |

KS p = 0.000; medians 1586 m apart. **The level is content-driven**, so the +111 m
agreement is not a fixed artefact. Caveat: the match COUNT is identical on both days
(11,653 vs 11,656) — the clear sky is a *different* signal, not no signal.

## 2. Desync probe — REFUTED
ρ(speed, dv) = **+0.002** (p = 0.97) against ρ(regime, dv) = **+0.447** (p = 1.2e−19).
Speed explains **0.0 %** of dv, regime **20.0 %**. Motion is identical across regimes
(0.68 vs 0.67 px) while dv doubles. **The exposure offset the observation would need is
18.14 s against an archive maximum of 5 s.**

## 3. Low-regime tracking — the inversion survives within one layer
215 frames, no shift search: **ρ = −0.473 per-frame, −0.525 gridded (p = 0.030, 17 points,
n_eff = 7)**. The estimate's spread in the low regime is **1225 m, twice the reference's
620 m** — the estimate is not flat, it is strongly varying and anti-correlated.

**Layering explains the LEVEL, not the SIGN.** Combined with W4 gates 1–4, every
hypothesis tested so far is refuted.

## The pivot
After nine refuted explanations, the productive question is no longer "what is wrong with
the pipeline" but **"what physical quantity does a dense matcher on a fisheye sky actually
return, and what does it correlate with?"**

---

# W6 — PHENOMENOLOGICAL CHARACTERISATION (2026-10-14)

Full record: `experiments/w6/W6_RESULT.md`. 120 frames, ground-truth-free.

| probe | result | reading |
|---|---|---|
| \|gradient\| vs dv, within frame | **+0.117**, positive in 82 % of frames | a real but weak photometric coupling to the perpendicular residual |
| \|gradient\| vs du | +0.002 (51 %) | none |
| **ρ(\|cos(gradient angle)\|, \|dv\|)** | **−0.000 (p = 0.91)** | **the photometric mechanism's distinguishing prediction FAILS** |
| brightness vs du, **within frame** | **−0.176** (negative in 73 % of frames) | a real coupling |
| brightness vs dv, within frame | +0.240 (69 %) | a real coupling |
| within-frame brightness span of du | **2.709 px** | larger than the effect under study (1.798 px) |

## The decisive point: the sign is wrong

The coupling is **negative** — brighter regions carry a *lower* disparity. The high regime
is **brighter** (median intensity 124 against 107). So photometry predicts the high regime
should have a *lower* `du`, whereas the observation is **+1.798 px higher**.

**The photometric hypothesis is refuted as an explanation of the inversion**, on three
grounds: opposite sign, a failed distinguishing prediction, and the wrong magnitude
ordering.

It does explain part of Gate 3 — the `dv` inflation has a real photometric component — but
not the systematic regime shift and not the inversion.

## Standing: ten refuted explanations

certainty · incoherent noise · calibration geometry · population size · pure depth change ·
a RoMa-specific artefact · target-identity layering · camera desync · a fixed pipeline
artefact · non-Lambertian photometry.

## Recommendation

Stop generating hypotheses. Every candidate has been a property of the scene or of the
pipeline, and each is measurable in isolation — and each is absent or wrong-signed. The
productive next step is a **controlled input**: a synthetic sequence with known height,
known motion and a known phase function, rendered through the same fisheye geometry, on
which the estimator must recover a height it cannot know. Failing that: a target, or an
instrumented session. **The analysis side of this question is, on present evidence,
exhausted.**

---

# W7 — SYNTHETIC CONTROL: THE MATCHER RECOVERS A KNOWN HEIGHT (2026-10-14)

Full record: `experiments/w7/W7_RESULT.md`.

A cloud layer at an exactly known height, rendered through the measured fisheye geometry,
matched by the production matcher, compared against a **per-pixel** ground-truth map.

| rendered H | ground truth | recovered | bias |
|---|---|---|---|
| 1000 m | 29.421 px | 28.633 px | −0.361 px (−1.23 %) |
| 1500 m | 19.599 px | 19.037 px | −0.085 px (−0.43 %) |
| 2000 m | 14.665 px | 14.406 px | −0.025 px (−0.17 %) |
| 3000 m | 9.747 px | 9.522 px | +0.030 px (+0.31 %) |
| 4000 m | 7.285 px | 7.205 px | +0.017 px (+0.23 %) |

**ρ(true, recovered) = +1.000, slope 0.968, bias spread 0.71 px** — against a real-world
cross-regime effect of +1.798 px.

## The pre-registered interpretation matrix selects its first row

Bias < 0.5 px at every height and ρ = +1.000: **the matcher and geometry are fundamentally
sound; the real-world inversion is caused by an unmodelled physical effect.** This is the
project's first positive result after ten refutations, and it says the instrument is
capable of the measurement.

Ruled out cleanly: the matcher architecture, the fisheye geometry and distortion model, the
crop and scale path, and a receptive-field/scale-aliasing defect.

## Two silent bugs, and the check that caught them

The first render gave a **negative** ground-truth disparity. Two independent errors, neither
raising an exception: a projection written for a **downward**-pointing camera (these point at
the zenith), and 3-D points built at `z = 0` instead of `z = height`. The check that caught
them is the simplest possible and belongs in any synthetic harness: **triangulating the
ground-truth disparity must return the height it was rendered at.**

## The productive next step: bisection, not hypothesis

The estimator is sound in simulation and fails on the real scene, so the question becomes
**which ingredient of the real scene is missing from the renderer.** Add each in turn and
re-test:

1. **3-D relief** — the render is a flat plane; a real cloud base is rough.
2. **Sensor and ISP realism** — noise, the magenta cast, JPEG artefacts.
3. **Motion between exposures.**
4. **Real texture** — procedural noise may not match a real cloud base, especially in the
   low-gradient interiors.

**Finding which one breaks the recovery is now the highest-value experiment available**,
because there is finally a working control to break.

---

# W7 — WITHDRAWN (2026-10-14)

**The W7 synthetic control is retracted.** Full record and reasoning:
`experiments/w7/W7_RETRACTION.md`.

The renderer never modelled the **90° relative sensor rotation** between the two cameras —
which is precisely what the calibration affine exists to undo. Both synthetic images were
produced in a shared orientation, so:

- the pair was never in the pipeline's convention;
- the affine (a 90° rotation) was applied to an image that did not need it;
- with that corrected the frames are misaligned by ~90° and matching fails catastrophically
  (recovered disparity IQR **688 px**, ρ = −0.600).

W7 therefore tested a rig in which both cameras point the same way. **That rig does not
exist, and the "bias < 0.4 px" result describes an un-rectified, identically-oriented pair
— not the production configuration.**

## How the flaw survived
The gate I applied — *triangulate the ground truth and check it returns the rendered
height* — passed **in the native frame**. The warped frame, which is what the matcher
consumes, was never validated.

## After correction, the renderer is still not right
| origin₂ | H=1000 | H=2000 | H=4000 | expected |
|---|---|---|---|---|
| (45,0,0) | 54.15 px | 54.13 px | 54.12 px | constant — **no parallax** |
| (0,45,0) | 24.63 px | 39.35 px | 46.72 px | increasing — **inverted** |

A physical parallax must fall as 1/H. Neither does.

## Standing
| claim | status |
|---|---|
| "RoMa recovers a known height on synthetic cloud texture" | **withdrawn** |
| the matcher architecture is cleared | **not established** |
| the fisheye geometry is cleared | **not established** |
| the analytic round trip (a plane at 1500 m triangulates to 1500.00 m) | **stands** — analytic, no renderer |
| Gate 0's finding that `lowest` reports 37–278 m for 1000–2000 m clouds | **stands as a measurement**, from the same flawed renderer; must be re-run |

**The project reverts to: ten refuted explanations and no positive control.** The matcher has
not been cleared.

## The gate any corrected renderer must pass first
The ground-truth disparity, expressed in the **warped** frame, must (a) fall as 1/H across
at least three heights and (b) triangulate back to the rendered height through
`heights_from_matches`. Until both hold, no result from the renderer may be reported.

---

# W7 (REDONE) — VALIDATED SYNTHETIC CONTROL (2026-10-14)

Full record: `experiments/w7/W7_RESULT_FINAL.md`. Supersedes the withdrawal.

## The renderer, derived from the calibration and gated
Baseline along world **−Y**; camera 2's sensor rotated **−90°**; **camera 2's principal
point at (1014.10, 965.03)**, which is `R·c1 + t` from the affine. That last term's absence
was what made the first renderer's parallax negative and height-increasing.

**Gate passes exactly**: `du(zenith)` = 29.428 / 14.715 / 7.357 px against
`f·atan(B/H)` = 29.410 / 14.713 / 7.357 px.

## 1. The matcher recovers the known height
| H_true | truth | RoMa | bias |
|---|---|---|---|
| 1000 m | 29.56 px | 28.72 px | −0.914 px |
| 1500 m | 19.73 px | 19.58 px | −0.239 px |
| 2000 m | 14.80 px | 14.85 px | +0.060 px |
| 3000 m | 9.87 px | 9.80 px | −0.071 px |
| 4000 m | 7.40 px | 7.48 px | +0.072 px |

**ρ = +1.000, median |bias| = 0.072 px.** The matcher and triangulation are **cleared** —
this time on a renderer that passed its own gate.

## 2. The shipped selection rule is the defect
| selector | ρ(true, reported) | p | error range |
|---|---|---|---|
| `median` | +1.000 | 0.000 | −56 to +24 m |
| **`lowest` (shipped)** | **+0.700** | **0.188 (n.s.)** | **−1471 to −325 m** |
| **`lowest_mode` (new default)** | **+1.000** | **0.000** | **−64 to +25 m** |

The shipped rule reports **675 m for a 1000 m cloud, 722 m for 1500 m, 529 m for 2000 m**.

## What this settles
The selection rule is a major defect, now **demonstrated on ground truth** rather than
inferred; the matcher and geometry are cleared; and `lowest_mode` — already the project
default — removes the defect.

## What it does not settle
The real-world inversion survives both. Replacing `lowest` with `lowest_mode` on real data
moved the regime shift only from ≈0 to −30 m (p = 0.058). **Two stages are now cleared on
ground truth, and the remaining failure is demonstrably in neither.**

---

# W8 — PHYSICAL BISECTION, INJECTION 1 (2026-10-14)

Full record: `experiments/w8/W8_RESULT.md`. A validated wind tunnel (W7 renderer, gate
passing exactly) with one real-atmosphere property added at a time.

**Pre-registered trigger:** the recovered height must fail to increase with the true height
— inverting, or compressing >50 % — across ≥3 heights. A bias is not an inversion.

## Injection 1 (texture scale) — CLEARED before W8 began
The renderer indexes its texture by **world coordinates** (`extent = 4000 m`), so the noise
is fixed in **metres**: a 352 m feature subtends **230 px at 1000 m and 57 px at 4000 m**.
The matcher recovered the truth at every scale.

## Injection 2 (3-D relief) — CLEARED
| relief | matcher ρ | `lowest_mode` ρ | triggered |
|---|---|---|---|
| 0 % | +1.000 | +1.000 | no |
| 5 % | +1.000 | +1.000 | no |
| 10 % | +1.000 | +1.000 | no |
| **20 %** (2× physical) | **+1.000** | **+1.000** | no |

At 20 % relief, errors are within 75 m across 1000–4000 m, ordering exactly monotonic.

**Worth running despite being negative:** it closes a mechanism no flat-plane control can
exercise — with relief the two cameras sample genuinely different surface points, which is
exactly the class of effect suspected.

## Order revised
Haze moves **last**, not first: W6 measured the photometric coupling and it runs the wrong
way (ρ(brightness, du) = −0.176 while the high regime is brighter). **Next: the broken cloud
field**, the only remaining mechanism that touches what each instrument is actually looking at.

---

# W8 INJECTION 3 — REFUTED ON REAL DATA, AND A SHARPER TARGET FOUND (2026-10-14)

Full record: `experiments/w8/W8_INJECTION3_TEST.md`.

The broken-field / sky-hallucination hypothesis (a clear-sky mode at ~44.9 px captured by
`lowest_mode`) was tested **before** building a renderer, using the 3.9 M real
correspondences already in hand.

| regime | median du | share > 30 px | share > 38 px |
|---|---|---|---|
| low | 12.27 px | 1.37 % | 0.84 % |
| high | 14.07 px | **3.55 %** | **2.57 %** |

**Refuted on two quantitative grounds:** 3.55 % is below `lowest_mode`'s own **10 %
prominence floor**, so the sky population cannot become the reported mode; and a 2.6 %
population cannot move a **median** by 1.80 px.

## What the test surfaced instead

| percentile | low du | high du | shift |
|---|---|---|---|
| 1 % | −1.88 | −9.07 | **−7.19** |
| 50 % | 12.27 | 14.07 | **+1.80** |
| 99 % | 35.16 | 72.67 | **+37.51** |
| 1–99 span | 37.0 px | **81.7 px** | **×2.2** |

Two distinct effects, with different likely causes:

1. **The central mass shifts by ~+1.5 px** (25th–90th percentiles all move +0.9 to +1.8 px).
   Not a tail artefact — the sky population cannot produce it.
2. **The dispersion roughly doubles**, with both tails extending.

The dispersion growth **independently corroborates Gate 3**, where the perpendicular epipolar
residual also nearly doubled (0.866 → 1.885 px). Two statistics on two components agree.

## Next target

**Dispersion, not height.** Any mechanism reproducing the inversion must reproduce both the
+1.5 px central shift and the 2.2× dispersion growth. Dispersion is the easier to measure and
falsify, and it is now the better-posed question.

---

# W8 INJECTIONS 4/5 — SUPERPOSITION: ONE TRIGGER, UNRESOLVED INCONSISTENCY (2026-10-14)

Full record: `experiments/w8/W8_SUPERPOSITION_RESULT.md`.

A synthetic cloud alpha-blended over a **real 2025-05-22 cloudless frame**, swept over
cloud weight α ∈ {1.0 … 0.2}. Trigger: 1–99 span ×≥1.8 with IQR ×<1.4.

| H | α | span × | IQR × | triggered |
|---|---|---|---|---|
| 2000 | 0.80 / 0.60 / 0.40 | 1.00–1.11 | ~1.0 | no |
| **2000** | **0.20** | **1.80** | **1.34** | **YES** |
| 4000 | 0.80–0.40 | 1.01–1.07 | ~1.0 | no |
| 4000 | 0.20 | 1.42 | 0.99 | no |

At H=2000, α=1.0 → 0.2 pulls `lowest_mode` from 1433 to 1260 m (**−173 m**, against a
real-world −297 m), so the mechanism is not obviously absent.

## Why this is NOT reported as a result

**At α = 1.0 the blend is byte-identical to the W7 control's input at H = 2000, and the two
disagree:**

| run | `lowest_mode` reports |
|---|---|
| W7 control | **1984 m** |
| this run, α = 1.0 | **1433 m** |

A **551 m** difference on identical input, through the same matcher and selector. Until that
is explained — a script bug, or non-determinism in the estimator — **no number in the table
can be interpreted, including the trigger.**

## Next step
Re-run the W7 control and this script's α = 1.0 side by side and find the discrepancy.
Reporting a trigger from a table whose control does not reproduce would repeat the exact
error of the withdrawn first W7.

---

# W8 INJECTIONS 4/5 — SUPERPOSITION REFUTED (2026-10-14)

Full record: `experiments/w8/W8_SUPERPOSITION_RESULT.md`.

## The 551 m discrepancy — found, and it was mine
**A call-site default mismatch, not a physical effect.** `w7_control` called
`render_pair(H, tex, tex_extent_m=4000.0)` → `roll_deg=0.0` (the default);
`w8_superposition`'s `cloud_layers` passed `roll_deg=1.5`. Rendering the same cloud 1.5°
apart gives `lowest_mode` = **1984 m** vs **1433 m** — the whole discrepancy.

All four proposed suspects were checked and cleared: the alpha-blend clip is a no-op (the
renderer's output lies in [0.113, 0.957]); **RoMa is deterministic** (5 calls, identical to
4 decimals); no images are written to disk; both scripts use the same frame convention.

## The corrected sweep — trigger NOT MET anywhere
| H | α | span × | IQR × | triggered |
|---|---|---|---|---|
| 2000 | 0.20 | 1.71 | **9.19** | no |
| 4000 | 0.20 | **1.86** | **2.87** | no |
| 4000 | 0.80 | 1.29 | 1.00 | no |

**Superposition widens the CORE; the real atmosphere widens the TAILS** (real: span ×2.2,
IQR ×1.11). At H=2000, α=0.20 the IQR grows ×9.19.

**And it needs 80 % clear sky.** The effect only appears at α=0.20; the real high regime
carries **2.6–3.6 %** sky-like population.

## Standing
| injection | status |
|---|---|
| 1. texture scale | cleared |
| 2. 3-D relief | cleared |
| 3. broken field / sky mode | refuted |
| **4/5. photometric superposition** | **refuted** — wrong dispersion signature, needs 80 % sky |
| 6. atmospheric haze | **not run — low value**, it is a contrast-reduction variant of the same refuted mechanism |

**The dispersion signature is the discriminating measurement**, and it has refuted the
photometric route.

---

# W8 INJECTION 7 — PHASE ALIASING REFUTED (2026-10-14)

Full record: `experiments/w8/W8_PHASE_ALIASING.md`. Refuted on real data before any renderer
work, as with the sky route.

**Two grounds:** a **+1.8 px** jump needs a ~1.8 px texture wavelength, while 100–1000 m cloud
features at 2–4 km subtend **100–1000 px**; and **no periodic structure exists in the
disparity histogram at all** — no period peaks above the 0.05 autocorrelation floor in either
regime.

**What survives:** the tail is **6× heavier** in the high regime (1.245 % vs 0.208 % beyond
median+50 px). Something generates it, but not periodically.

## Twelve refuted hypotheses
certainty · incoherent noise · calibration · population size · pure depth change · a
RoMa-specific artefact · target-identity layering · desync · a fixed pipeline artefact ·
non-Lambertian photometry · photometric superposition · phase aliasing.

**The cause of the inversion remains unattributed.** No analysis-side mechanism tested
accounts for it.

---

# W9 — STATIC FIELD DIAGNOSED, AND A CROSS-VALIDATED REMOVAL SEARCH (2026-10-15)

Full record: `experiments/w9/W9_RESULT.md`. Scripts: `experiments/w9/w9_cv.py`.

## The diagnostic: u, not radius
| predictor | r | R² |
|---|---|---|
| **u** | **+0.814** | **0.663** |
| cos(angle) | +0.767 | 0.588 |
| **radius r** | **+0.499** | **0.249** |

**A linear gradient in u accounts for 66 % of the static field's variance; radius accounts
for 25 %.** The radial profile is monotone (14.51 → 16.57 px, +14 %), but it is largely a
consequence of the u-gradient. And the radial structure **largely cancels in height**:
after the measured corrections, corr(radius, height) = **−0.146**, with a **3 %** median-height
spread across the radius against the disparity's 14 %.

## The search: 6 corrections × 3 estimators × 5-fold, out of sample
| correction | estimator | LEVEL ρ | DYNAMICS ρ |
|---|---|---|---|
| **per-cell+linear** | **mode** | **+0.287** | −0.034 |
| none (baseline) | mode | +0.171 | +0.061 |
| none | median | **−0.409** | +0.126 |

### Finding 1 — the sign depends on the estimator, and this corrects the record
The **mode** estimator is **correctly signed** out of sample (+0.171 → +0.287). The **median**
and p25 are **inverted** (−0.41, −0.30). The project's "inversion" was measured on
median-like statistics; it is a property of *those* statistics, not of the disparity field as
a whole. **This must be carried into `final-report.md` §2.**

### Finding 2 — no correction recovers the dynamics
Across all eighteen combinations the dynamics ρ lies in **[−0.166, +0.168]** with no
consistent sign. The best level correction *worsens* the dynamics (−0.034 vs +0.061
uncorrected).

**The static field is partly removable; removing it does not recover the dynamics.** The
anomaly cannot be described as "a static field masking a real signal" without qualification —
an expressive correction does not reveal the signal, and the possibility that the disparity
variation carries no information at this footprint ratio is not excluded.
