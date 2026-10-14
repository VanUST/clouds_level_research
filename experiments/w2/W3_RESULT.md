# W3 — instrumental vs climatic, and the origin of the inverted response

Continuation of the advisor's diagnostic after the foreground route (W2) was closed.
Four measurements, each pre-stated here with the outcome. No configuration was changed
after seeing a result.

---

## W3.1 — The static pattern is BOTH instrumental and climatic

**Method.** The static pattern could be an instrument error (H1/H2) or a climatology of
where cloud happens to sit (H3). H3 makes a falsifiable prediction: the pattern must
change when the cloud regime changes. The session was therefore partitioned by the
**ceilometer's own reported state** — 64 frames with H1 < 2750 m, 83 with H1 > 3500 m —
and the per-cell disparity map built separately in each partition. The ceilometer only
*sorts* frames; it never enters the estimate, so this respects guardrail G1.

| quantity | value |
|---|---|
| correlation of the per-cell map between the two regimes | **r = +0.371** |
| shuffled null | +0.001 (p95 +0.056) |
| **control**: the same comparison *within* one regime | **r = +0.718** |
| median disparity, low regime / high regime | 13.75 / 15.42 px (11.4 % apart) |

Two things follow. The between-regime correlation is far above chance, so a large part of
the pattern **cannot** be cloud — clouds in a low regime cannot reproduce a pattern in a
high one. But it is well below the within-regime reproducibility ceiling of 0.718, so a
real regime-dependent component exists too.

**Variance decomposition** over 923 cells:

| component | variance | share |
|---|---|---|
| total spatial variance | 2.3146 px² | — |
| **instrumental** (mean of the two regimes) | **1.0762 px²** | **46 %** |
| **regime-dependent** (half-difference) | **0.5422 px²** | **23 %** |
| ratio instrumental / (instrumental + regime) | **0.665** | |

**The answer to the directive's central question is BOTH, approximately two thirds to one
third.** Neither a pure calibration correction nor a pure climatology explains it.

---

## W3.2 — Removing the instrumental component does NOT fix the sign

**Method.** If the instrumental part is additive and independent, subtracting the
session-mean per-cell disparity (the reproducible component) leaves the regime-dependent
field. If the inversion were the calibration's, the corrected series would respond with
the correct sign. **The correction is not fitted to the ceilometer** — the mean is taken
over the session, and the reference is consulted only afterwards — so a positive result
would not be circular.

| | raw | instrumental removed |
|---|---|---|
| median height | 1992 m | 1961 m |
| regime shift (high − low) | **−129 m** | **−164 m** |
| rank correlation with the reference | −0.547 | **−0.581** |
| p | 5.4e−14 | 5.6e−16 |

**The inversion gets STRONGER, not weaker.** It is not the calibration's, and no
geometric correction of the kind the directive proposes will fix it.

---

## W3.3 — The inversion is robust, and lives in the matched population

The response's shape, by reference decile (162 frames):

| reference H1 | median disparity | median estimated height | matches |
|---|---|---|---|
| 1820–2330 m | 13.94 px | 2111 m | 1906 |
| 2526–2778 m | 14.42 px | 2041 m | 1751 |
| 3625–3770 m | 15.16 px | 1941 m | 2897 |
| 3860–3950 m | 15.09 px | 1951 m | 3630 |
| 4000–6160 m | 15.03 px | 1958 m | 2642 |

**A correct stereo requires a NEGATIVE correlation between reference height and disparity.
The measured sign is POSITIVE: ρ = +0.547 (p = 5.4e−14).**

Three checks that could each have killed it:

| check | result |
|---|---|
| is it an edge effect? | **no** — ρ = +0.540 at r ≤ 100 px and +0.547 at r ≤ 452 px; present at every radial cut, and the *low* regime's disparity is 10.59 px in the centre against 14.71 px in the high regime |
| is it a diurnal confound? | **no** — ρ(hour, reference) = +0.114 (p = 0.15, not significant); partial ρ(reference, disparity \| hour removed) = **+0.498 (p = 1.6e−11)** |
| what mediates it? | **the match count** — ρ(matches, disparity) = +0.559 and ρ(matches, reference) = +0.425. The high regime yields ~2× the matches (3630 vs 1751), and the extra population sits at higher disparity |

So the estimator's response to the cloud regime is **real, significant, robust, and
inverted**, and it is carried by **which correspondences the matcher returns**, not by
the geometry those correspondences are put through.

---

## W3.4 — The pipeline's own statistics hide the inversion

Measured on 300-frame runs, all six shipped configurations:

| configuration | median h | shift (high − low) | p | ρ | sign |
|---|---|---|---|---|---|
| `lowest_mode` + masks + f=654 | 1990 m | −30 m | 0.058 | −0.182 | inverted |
| `lowest` + masks + f=654 | 1707 m | +0 m | 0.76 | +0.009 | flat |
| `lowest`, f=611, no masks | 1551 m | +5 m | 0.32 | +0.064 | flat |
| `lowest`, filter off | 1508 m | +18 m | 0.35 | +0.104 | flat |
| exact triangulation only | 1096 m | +1 m | 0.99 | +0.024 | flat |
| roll + distortion | 1551 m | +5 m | 0.32 | +0.064 | flat |

**The median disparity tracks the regime strongly (ρ = +0.55), but every low-tail
statistic the pipeline reports is flat (|shift| ≤ 30 m, all p > 0.05).** The estimator
discards the signal that does exist — an inverted one — and reports a statistic that
carries no information about the regime at all.

---

## What this changes

1. **The directive's D2 hypothesis is refuted.** Replacing the affine with an
   essential-matrix pose cannot fix the retrieval, because the inversion survives the
   removal of the entire instrumental component and strengthens when it is removed.
2. **The failure is in the matched population (R1), and it is now measured rather than
   suspected.** The regime changes how many correspondences the matcher returns, and the
   additional population has a systematically different disparity.
3. **The next experiment should be about the matcher's population, not the geometry.**
   Specifically: does the extra correspondence population in the high regime survive any
   internal consistency test — the ray-intersection residual, cross-matcher agreement, or
   spatial smoothness — or is it noise admitted by a lower-contrast sky?
4. **Calibration work should stop.** Two independent tests (W3.2 and the R-16 scale
   route) have now shown that geometric corrections move the error, not the signal.

## What is still not known

- **Why the population inverts.** The candidates are a genuine optical difference
  (contrast, illumination, cloud texture) selecting different features, or a matcher
  behaviour that changes with the sky. Not yet separated.
- **H1 vs H2.** Still not separable without a control, which W2 showed is unavailable.
- **Absolute height.** Unchanged: the baseline is assumed and never measured.
