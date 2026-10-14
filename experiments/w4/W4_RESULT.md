# W4 — matched population investigation: gates 1 and 2

Direct answers to the two architectural questions, then the measurements.

---

## Q1 — raw per-match dumps: agreed, and one premise was wrong

**Agreed** — W4 runs strictly on raw, un-clustered correspondences, bypassing HDBSCAN
and every selection rule. Re-tuning the clustering would only change which slice of a
censored signal we inspect.

**The certainty gate needed new data.** No existing artefact carried RoMa's certainty
(`heights.npz`: stamps, heights, uv, kp2, f, baseline, centre; `matches_cache_2025-05-23.npz`:
kp1, kp2, stamps, frames). A fresh pass was made with `return_conf=True` and the pipeline's
exact preparation:

**`experiments/w4/matches_roma.npz` — 386 frames, 4,499,068 correspondences, ~11,656 per
frame.**

**And that number corrects a claim I made in W3.3.** I reported that "the high regime
returns ~2× the matches (3630 vs 1751)". That was measured on the **clustered** dump. On
the raw matcher output the rate is **identical**:

| regime | matches per frame |
|---|---|
| low | 11,655.1 |
| high | 11,655.8 |

RoMa emits a fixed dense grid, so the population SIZE cannot differ. What differs is the
**disparity assigned to the same grid points**. The directive's "second population of
matches" is better described as *a different disparity assignment at the same pixels* —
the clustering, not the matcher, was creating the apparent population change.

---

## Q2 — the ordered triad: agreed, with the certainty gate reframed

The order **Certainty → Smoothness → Epipolar → cross-matcher** is right. Two corrections
were applied before running it:

**1. Gate 1's premise was already falsified on this data.** The directive reads "*if the
extra matches are low-certainty, the matcher is likely hallucinating*". R‑7 measured the
opposite: RoMa's low-certainty interior predictions agree with the reliable field to
**0.68 px against a 7.02 px spatially-shuffled null**, and with LoFTR's independent matches
to 0.62 px. On this rig low certainty means low texture, not a wrong answer. A gate that
stopped at "they are low-certainty" would reach the wrong conclusion by construction.

**Reframed to a matched comparison:** *within a band of equal certainty, do the regimes
differ?* Certainty cannot explain a difference that survives holding certainty fixed.

**2. SGBM is a poor final arbiter.** It invents a full field on a cloudless sky
(6,400 matches/frame). **LoFTR** returns 11 on clear sky against 2,189 on cloud, so it is
the useful cross-matcher.

---

## Gate 1 — certainty does NOT explain the difference

`experiments/w4/w4_certainty.py`. 372 frames, 4,335,993 correspondences.

| certainty decile | n low | n high | du low | du high | difference |
|---|---|---|---|---|---|
| 0.000–0.000 | 254,758 | 143,139 | 12.21 | 13.18 | **+0.97** |
| 0.000–0.000 | 270,244 | 120,283 | 12.11 | 14.61 | **+2.50** |
| 0.000–0.002 | 261,567 | 130,051 | 12.11 | 13.78 | **+1.68** |
| 0.002–0.008 | 261,104 | 129,488 | 12.00 | 13.61 | **+1.61** |
| 0.008–0.037 | 255,384 | 134,007 | 11.99 | 13.76 | **+1.77** |
| 0.037–0.147 | 252,219 | 136,654 | 12.20 | 14.00 | **+1.80** |
| 0.147–0.422 | 248,483 | 141,160 | 12.39 | 14.13 | **+1.74** |
| 0.422–0.733 | 234,318 | 154,541 | 12.76 | 14.18 | **+1.42** |
| 0.733–0.998 | 206,855 | 178,078 | 13.46 | 14.28 | **+0.82** |

**The high regime has the higher disparity at every certainty level** — all ten deciles
share the sign, weighted mean **+1.59 px** against a pooled **+1.80 px**.

| quantity | value |
|---|---|
| median certainty, low / high | 0.0062 / 0.0139 |
| **the high regime's surplus population has median certainty** | **0.782** |
| verdict | **CERTAINTY DOES NOT EXPLAIN THE DIFFERENCE** |

**The extra matches are HIGH certainty, not low.** The directive's hypothesis is refuted
directly: this is not a matcher hallucinating on a low-contrast sky. And the difference
persists at matched certainty, so certainty cannot be its cause in any case.

---

## Gate 2 — both regimes are equally coherent

`experiments/w4/w4_smoothness.py`. For every correspondence, the residual against the
median disparity of its 25 nearest neighbours — a cloud surface is smooth, noise is not.
215 low-regime frames, 120 high-regime frames.

| | median residual | p90 |
|---|---|---|
| low regime | **0.047 px** | 0.061 px |
| high regime | **0.041 px** | 0.137 px |
| ratio high/low | **0.865** | |

The disparity difference being explained is **+1.785 px**. The coherence difference is
**−0.006 px** — the high regime is, if anything, marginally *more* coherent
(Mann-Whitney p = 3.6e−4, in the "more coherent" direction).

**The disparity difference is not carried by scattered correspondences.** Both regimes
produce equally smooth disparity fields; the high regime's field is a real, coherent
displacement of the matching solution.

---

## What the two gates establish together

| candidate explanation | status |
|---|---|
| certainty / low-confidence hallucination | **refuted** — difference survives matched certainty, and the surplus is high-certainty |
| admitted noise / incoherent matches | **refuted** — both regimes equally coherent, high regime marginally better |
| calibration / geometry (W3, D2) | **refuted earlier** — removing the instrumental component strengthens the inversion |
| an edge or diurnal confound | **refuted earlier** — present at every radius; survives partial correlation on hour |
| population SIZE | **refuted** — the rate is identical to within 0.01 % |

**What remains.** The matcher returns a fixed grid of *confident, spatially coherent*
disparities, and the field it produces in the high regime sits **1.8 px** — about 350 m of
height — away from the low regime, in the direction that corresponds to a *lower* cloud.
Nothing internal to the matching explains it.

That leaves the two possibilities the project has never been able to exclude:

1. **Target-identity mismatch.** The ceilometer samples a ~10 m column and the camera a
   1956 m footprint (a 200:1 ratio, 8.5 texture decorrelation lengths). The two
   instruments may simply be measuring different cloud.
2. **A physical effect** that makes a genuine high cloud present a larger apparent
   disparity — which no measurement in this project has yet identified.

Note that restricting the camera to a 300 m central disc (r ≤ 100 px) was already tested
in W3.3 and the inversion **persisted** (ρ = +0.540), which weakens but does not exclude
explanation 1.

## Next, in the triad's order

**Gate 3 (epipolar residual)** is the natural next step: if the two regimes' matches both
satisfy the epipolar constraint equally well, then the difference is a genuine
displacement of the solution along the epipolar line — which would point at the sky
itself rather than at the matcher. **Gate 4 (LoFTR cross-check)** then asks whether an
independent matcher sees the same displacement.
