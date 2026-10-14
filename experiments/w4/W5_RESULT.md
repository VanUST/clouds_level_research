# W5 — the three contracts: cloudless null, desync probe, low-regime tracking

Executed in the order requested: Cloudless Null first, then the desync probe (because its
own premise was already measured and contradicted), then low-regime tracking.

---

## 0. A premise correction, before any of it

**The desync-motion hypothesis rests on "the two fisheye cameras are desynchronized by
1–2 seconds". That is not in the archive**, and it was corrected in the W2 pre-registration
before this round:

| measurement | value |
|---|---|
| 2025-05-22 frames with BOTH cameras on one stamp | **8,173 / 9,091 = 89.9 %** |
| 2025-05-23 frames with BOTH cameras on one stamp | **8,224 / 9,041 = 91.0 %** |
| median offset | **0 s** |
| p90 offset | **0 s** |
| max offset | 5 s (05-22), 2 s (05-23) |

Identical filenames do not strictly prove identical exposures, so the hypothesis was tested
independently rather than dismissed. It is refuted below on its own arithmetic.

---

## 1. Cloudless Null — **PASS**

`experiments/w4/matches_roma_20250522.npz` (244 frames, 2,843,421 correspondences),
identical preparation, compared against 2025-05-23.

| | cloudy (05-23) | cloudless (05-22) |
|---|---|---|
| matches per frame | 11,656 | **11,653** |
| **implied height, median** | **2242 m** | **655 m** |
| median disparity | 13.13 px | **44.91 px** |
| median certainty | 0.0084 | **0.0012** |
| KS test on the height distributions | — | **p = 0.000** |
| difference of medians | — | **1586 m (70.8 %)** |

**The camera does not return cloud heights on a cloudless sky.** The level is
content-driven, so the +111 m agreement in the low regime is not a fixed pipeline artefact
reproduced regardless of scene.

**One caveat worth carrying:** the match *count* is identical on both days (11,653 vs
11,656), because RoMa emits a fixed grid. The cloudless day is not "no signal" — it is a
*different* signal, at 655 m. The null therefore shows the level is scene-sensitive; it
does not show the estimator is inert without cloud.

---

## 2. Desync-Motion Probe — **REFUTED on three independent grounds**

`experiments/w4/w4_desync.py`. Motion measured by phase correlation between consecutive
crops — no matcher, no model, so it cannot share a failure mode with the matcher whose
residual it is explaining. 371 frames.

The hypothesis's signature: contamination must scale with cloud **speed**
(`dv ∝ f·v·Δt/H`), not with the height regime per se.

| test | result |
|---|---|
| **ρ(speed, dv)** | **+0.002** (p = 0.97) — **no relationship at all** |
| ρ(speed, du) | −0.022 (p = 0.68) |
| **ρ(regime, dv)** | **+0.447** (p = 1.2e−19) |
| variance of dv explained by **speed** | **0.0 %** |
| variance of dv explained by **regime** | **20.0 %** |
| motion, low vs high regime | 0.68 px vs 0.67 px — **identical** |
| **exposure offset required to explain dv** | **18.14 s** (median) |
| archive's measured offset | 0 s median, 5 s maximum |

**The epipolar violation tracks the cloud's HEIGHT and not its SPEED, and the exposure
offset the observation would require (18 s) is fourteen times the archive's maximum.** The
hypothesis is refuted by its own arithmetic.

A detail that cuts against it further: the measured motion is *smaller* in the high regime
(0.67 px) than in the low regime (0.68 px), while dv is *more than twice as large*
(1.912 vs 0.862 px). Desync-motion predicts the opposite ordering.

---

## 3. Low-Regime Tracking — the inversion survives within a single layer

`experiments/w4/w4_lowregime.json`. Low regime only (H1 < 2750 m), 215 frames over 133 min,
**no time-shift search of any kind**.

| quantity | value |
|---|---|
| per-frame rank correlation | **ρ = −0.473 (p = 2.3e−13)** |
| lag-1 autocorrelation, estimate / reference | 0.937 / 0.934 |
| **effective sample size, estimate / reference** | **7.0 / 7.3** (nominal 215) |
| resampled to the slower series' decorrelation time (262 s) | **17 points** |
| **gridded rank correlation** | **ρ = −0.525 (p = 0.030)** |
| estimate spread in the low regime | 2002–3228 m (1225 m) |
| reference spread in the low regime | 2080–2700 m (620 m) |

Two conclusions:

**The inversion persists with both instruments looking at the same layer.** This is the
decisive test of the multi-layer hypothesis, and the hypothesis does not survive it. What
layering explains is the *level* (+111 m in the low regime); it does not explain the
*sign*.

**And a correction to the "flat estimate" framing.** Within the low regime the estimate's
spread is **1225 m** — *twice* the reference's 620 m. The estimate is not flat here; it is
strongly varying and anti-correlated. The flatness seen earlier was a property of the
low-tail statistics the pipeline reports, not of the underlying field (which is exactly
what W3.4 found).

**Power warning, stated plainly:** n_eff is **7**. The per-frame p-value is inflated by
autocorrelation, and the honest figure is the gridded one, ρ = −0.525 at p = 0.030 over 17
points. The effect is significant but the study is underpowered, and no finer structure
should be claimed from it.

---

## Where the project now stands

| hypothesis | verdict | evidence |
|---|---|---|
| certainty / low-confidence hallucination | refuted | gate 1: difference at every decile; surplus is high-certainty |
| incoherent noise | refuted | gate 2: both regimes equally smooth |
| calibration / geometry | refuted | W3.2: removal strengthens the inversion |
| population size | refuted | identical rate, 11,655 vs 11,656 |
| purely along-epipolar depth change | refuted | gate 3: the perpendicular residual nearly doubles |
| a RoMa-specific artefact | partly refuted | gate 4: LoFTR agrees in sign, 6.5× smaller |
| **target identity / layering** | **refuted as an explanation of the sign** | W5.3: the inversion persists within the low regime |
| **desync-motion** | **refuted** | ρ(speed, dv) = +0.002; required offset 18 s against a 5 s maximum |
| a fixed pipeline artefact | **refuted** | W5.1: the cloudless day gives 655 m, not 2242 m |

**Nothing tested explains the inversion.** The estimator tracks the cloud's height regime
with the wrong sign, robustly, coherently, confidently, at every scale, in both matchers,
under both layers, and independently of speed.

That is now itself the finding, and the honest next move is to stop generating hypotheses
and instead **characterise what the estimator IS measuring** — because after nine refuted
explanations, the productive question is no longer "what is wrong with the pipeline" but
"what physical quantity does a dense matcher on a fisheye sky actually return, and what
does it correlate with?"
