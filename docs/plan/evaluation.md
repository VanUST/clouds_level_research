# Evaluation design

**Status:** proposed, with the measurements that motivate it. Replaces the
windowed-correlation protocol, which was shown to be measuring noise
([temporal-alignment.md](temporal-alignment.md)).

---

## 1. Why the previous protocol failed

The old metric searched a ±300 s time shift inside each 30-minute window and
reported the largest correlation found. Three defects compounded:

| defect | consequence |
|---|---|
| 20 overlapping windows per 5-hour session | the same data counted repeatedly, inflating the apparent number of tests |
| maximising over 121 shifts | the reported value is a maximum of a noise curve, not a measurement |
| no account of autocorrelation | a window reported ~152 paired samples when it held ~3.4 independent ones |

The null test settled it: rolling the estimate against a fixed ceilometer and
repeating the same scan gives a *higher* mean scan-max r (0.346 raw, 0.421 filtered)
than the observation (0.189, 0.209). The reported correlations were below chance.

**Rule that follows:** a metric value produced by maximising over alternatives is
not evidence unless the null is maximised the same way.

## 2. Effective sample size comes first

Measured on the 2025-05-23 roll-corrected run (1,784 frames over 5 hours):

| series | nominal n | lag-1 ρ | **effective n** | decorrelation time |
|---|---|---|---|---|
| algorithm, raw per-frame | 1,784 | 0.339 | **881** | ~20 s |
| algorithm, Kalman-filtered | 1,784 | **0.979** | **18.8** | ~949 s |
| ceilometer H1 | 905 | 0.894 | **51** | ~267 s |

Two consequences that shape everything else:

1. **Any comparison is capped by the slower series.** The grid on which two series
   are compared must be spaced at the *larger* decorrelation time. Using the
   ceilometer's 267 s while the estimate needs 949 s leaves the estimate 0.98
   correlated and understates the significance threshold — the same error in a new
   place. It caught this design on its first run.
2. **The filtered output is nearly impossible to validate.** ρ = 0.979 over five
   hours leaves ~19 degrees of freedom, so the smallest detectable |r| is ~0.49.
   The Kalman filter is destroying exactly the variability any validation depends
   on. **Validation must target the per-frame series or the layers, not the smoothed
   output.**

## 3. What the properly-powered test says

With the estimate and the ceilometer both resampled to a spacing that makes their
samples independent, no shift search, and a circular-roll null:

| comparison | n | r | 95 % CI | verdict |
|---|---|---|---|---|
| **level**, whole session, 30 s grid | 470 | **−0.376** | [−0.653, −0.262] | significant, **negative** |
| differences, gap-crossing (flawed) | 469 | −0.248 | — | artefact, see below |
| **differences, strictly 30 s adjacent** | 439 | **−0.045** | [−0.100, −0.007] | **nil** |
| filtered level, 960 s grid | 19 | −0.517 | — | only 19 dof |

**Correction to an earlier version of this document.** It reported r = −0.248 for the
first differences and concluded that the estimate's dynamics were significantly
anti-correlated. That number is a **protocol artefact**: `np.diff` was applied to the
*masked* grid, and 30 of the 469 differences span a missing epoch, silently pairing
samples that are minutes apart. Restricting to genuinely 30 s-adjacent pairs gives
**r = −0.045, 95 % CI [−0.100, −0.007]** — indistinguishable from zero.

> **The failure is in LEVEL, not in dynamics.** The estimate is offset and inverted
> against the reference; it is not moving in opposition to it.

This matters for where to spend effort: a dynamics defect would call for better
temporal processing, whereas a level defect with a flat estimate calls for
establishing **what the matched population actually is** before anything else
(see [research-workplan.md](research-workplan.md) R1).

Stated with its limit: every number here comes from one session and the checks are
time blocks of the same day, not independent periods. [research-workplan.md](research-workplan.md)
§4 explains why an independent period is not available.

## 4. The metric hierarchy

Ordered by statistical power. Prefer the highest tier that can answer the question.

### Tier 1 — internal, ground-truth-free (highest power, always available)

These need no reference at all and can be evaluated per frame, giving thousands of
independent samples.

| criterion | what it tests | how |
|---|---|---|
| **ray-intersection residual** | do the two cameras agree about a point | returned free by `clouds.geometry.triangulation` |
| **radial flatness** | does height depend on image position | median height per radius bin; this is what found the lens distortion |
| **cross-region agreement** | do disjoint parts of one layer agree | split the field, compare heights with an uncertainty-aware test |
| **temporal coherence** | is the implied motion physically possible | bound the height rate |
| **layer stability** | is a mode persistent or a fluctuation | mode count and lifetime |

Tier 1 is where the real progress came from: the focal constant, the roll, and the
distortion were all found without the ceilometer.

### Tier 2 — external, distributional (level)

Compare the *distribution* of the estimate with the *distribution* of the reference
over the whole session. No time alignment is involved, so autocorrelation does not
matter, and the sample size is the number of measurements rather than the number of
independent epochs.

- two-sample Kolmogorov–Smirnov, and the Wasserstein distance, with a block-bootstrap
  confidence interval;
- **layer-count agreement**: the reference reports two layers much of the time; the
  estimate reports one. That is a quantifiable structural disagreement.

Current state: the ceilometer's lowest layer is bi-modal (40 % at 2250–2750 m, 53 %
at 3750–4250 m); the estimate is single-modal at ~1800 m. The distributions differ
grossly, and this is measurable without any correlation.

### Tier 3 — external, temporal (dynamics, low power)

Only after Tiers 1 and 2 pass, and only with:

- a grid at the **maximum** of the two decorrelation times;
- **no unaccounted maximisation** (if a shift is searched, the null searches too);
- one comparison over the whole session, not overlapping windows;
- pre-registered direction and threshold;
- a block bootstrap or a circular-roll null, reported with the p-value.

## 5. Pre-registration

Every evaluation states, before running: the series compared, the grid spacing and
the decorrelation time it comes from, the effective n, the exact metric, the null,
the significance threshold, and what result would count as failure. Recorded in
`experiments/` alongside the code. A metric chosen after seeing the result is not
evidence.

## 6. Reporting rules

- Report **effective n**, not nominal n.
- Report a **p-value** against a stated null, or say plainly that none was computed.
- Never report a best-of-search value without a null that searched identically.
- Report the **direction** of the effect; a significant negative correlation is a
  finding, not a null result.
- State the number of comparisons run on the same data.
- A null result is recorded, not dropped.

## 7. What is now unfounded

Following §1–§3, the project currently has **no demonstrated agreement** between the
estimator and the ceilometer in level or dynamics, and one significant result in the
opposite direction. Any earlier figure quoting a windowed r is withdrawn
([w1-calibration.md](w1-calibration.md) W1.15).

What remains supported, because it never used this metric: the measured focal
constant, the measured roll, the lens distortion, the calibration-error margin
arithmetic, the ceilometer's layer structure, and the internal geometric
consistency improvements.

## 8. What the corrected protocol reveals about the estimator

Applying the gap-safe protocol to the shipped configuration reclassifies the problem.
From `experiments/planning/anti_correlation_probe.py` on the baseline cluster dump
(1,776 frames):

**The estimate is nearly flat while the reference swings.** Hourly medians
(MSK 12:00→16:00):

| | 12 h | 13 h | 14 h | 15 h | 16 h |
|---|---|---|---|---|---|
| estimate | 1590 m | 1704 m | 1573 m | 1571 m | 1570 m |
| ceilometer H1 | 3150 m | 2520 m | 3920 m | 3930 m | 2480 m |

The reference moves through ~1450 m; the estimate moves through ~130 m.

**Where it does respond, the sign is inverted.** Stratified by the reference's own
regime:

| field | lower regime (H1 < 2750 m) | upper regime (H1 > 3500 m) | shift | KS p |
|---|---|---|---|---|
| full field | 1944 m | 1839 m | **−105 m** | 3 × 10⁻²⁴ |
| r ≤ 250 px | 1993 m | 1839 m | **−154 m** | 10⁻⁴⁶ |

The match-radius distribution is identical across the two regimes (median 227.9 vs
227.7 px), so this is not a radial-geometry artefact. **When the cloud is higher, the
estimate is lower.**

**The flatness is in the height field, not in the selection rule.** Every internal
rule is flat over five hours: `lowest` 1156 m, `largest` 1744 m,
median-of-candidates 1878 m. Changing the rule changes the level, not the flatness.
And the shipped `lowest` rule is reporting a **low-tail order statistic** of a
heavy-tailed field — median 1156 m with an IQR of only 88 m, with the selected
cluster sitting at a median radius of 334 px and 22.6 % of frames selecting a cluster
in a single 60 × 60 px bin. Excluding the top 128 px of the field lifts the selected
median only to 1463 m, so it is not one removable occluder.

**The dominant error is the within-frame scatter**, 1000–1400 m, which is 7–14×
larger than the estimator's entire response to the reference. Optimising a
correlation cannot address that; establishing what the matched population is can.
