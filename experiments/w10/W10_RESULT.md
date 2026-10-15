# W10 — the shape mechanism confirmed, and the dynamics boundary refined

Two questions in one pass: does any statistic of the per-match disparity field carry the
reference's temporal variation, and is the median/mode disagreement explained by a change in
the distribution's **shape**?

## The sign convention, stated because the previous round got it wrong

These statistics are **disparities** and the reference is a **height**. A correct estimator
needs the disparity to **fall** as the cloud rises, so for this table a **negative ρ is the
correct sign** and a positive ρ is the inversion. (W9 worked in a height-ordered convention
where the opposite held; both are stated here so the two are not misread against each other.)

## (A) Signal existence — a weak, correctly-signed signal that does not survive correction

27 bins of 300 s, **n_eff = 17.7** for the binned reference, 22 tests.

| statistic | LEVEL ρ | DYNAMICS ρ | p | sign |
|---|---|---|---|---|
| mean | +0.158 | **−0.357** | 0.074 | correct |
| **median** | +0.159 | **−0.409** | **0.038** | correct |
| mode | −0.238 | −0.374 | 0.209 | correct |
| p10 | +0.260 | −0.338 | 0.092 | correct |
| **p25** | +0.159 | **−0.400** | **0.043** | correct |
| p75 | +0.138 | −0.340 | 0.089 | correct |
| p90 | +0.070 | −0.110 | 0.591 | correct |
| std | −0.036 | −0.052 | 0.800 | correct |
| skew | −0.230 | −0.306 | 0.129 | correct |
| kurtosis | +0.120 | +0.149 | 0.466 | inverted |
| iqr_span_ratio | +0.086 | +0.130 | 0.526 | inverted |

**Nine of eleven statistics are correctly signed on the dynamics**, and the five central ones
(mean, median, p25, p75, p10) average **−0.369**, ranging −0.338 to −0.409.

**But nothing survives correction.** The Bonferroni threshold for 22 tests is p < 0.00227 and
the smallest observed p is **0.038**. Under a null of no signal, nine of eleven statistics
sharing a sign has p ≈ 0.033 by a sign test — suggestive, marginal, and **not independent**,
since these statistics are computed from the same matches and are strongly correlated with
one another.

### This corrects W9's conclusion

W9 reported that no correction recovered the dynamics (|ρ| < 0.17) and I read that as the
disparity field carrying no dynamic information. **That was too strong.** W9 measured
*per-frame first differences on held-out folds*, where per-frame matcher noise dominates; at
300 s binning the same data yields a consistent correctly-signed correlation of about
**−0.37**. The two are not contradictory — binning suppresses the noise that the per-frame
analysis is dominated by — but the earlier statement of the boundary was overstated.

**The correct statement of the boundary:** at a 300 s resolution the disparity field carries a
weak, correctly-signed dynamics correlation of roughly −0.37 which is **not statistically
established** on this session at n_eff = 17.7 with 22 tests. It is neither demonstrated nor
excluded.

## (B) The shape mechanism is CONFIRMED

| quantity | low regime | high regime | change |
|---|---|---|---|
| median | 12.354 px | 13.960 px | **+13.0 %** |
| **mode** | 12.896 px | 11.371 px | **−11.8 %** |
| **skewness** | **2.268** | **0.731** | **−67.8 %** |
| kurtosis | 36.88 | 32.23 | −12.6 % |
| **std** | 4.303 | 4.283 | **−0.5 %** |
| iqr_span_ratio | 0.283 | 0.348 | +22.9 % |
| p10 | 9.417 | 10.477 | +11.3 % |
| p90 | 16.087 | 16.719 | +3.9 % |

**The median and the mode move in opposite directions** (+13.0 % against −11.8 %) **while the
standard deviation is unchanged to within 0.5 %.** The distribution does not get wider or
narrower between regimes; it **changes shape**, its skewness collapsing by 68 %.

That is the mechanism. The disagreement between the median and the mode is not a defect of
either; it is what a distribution with a changing shape does to order statistics. A statistic
that tracks the bulk (the mode) and one that tracks the cumulative mass (the median) must
diverge when the skewness changes, and the skewness changes by a factor of three.

It also explains why the synthetic control passed for `lowest_mode` and failed for `lowest`:
the synthetic cloud's disparity distribution is symmetric and fixed in shape across heights,
so no statistic is biased by a shape change, and only the tail-sensitive rule fails.

## What stands

| claim | status |
|---|---|
| the median/mode disagreement is caused by a change in distribution shape | **confirmed** — median +13.0 % against mode −11.8 %, skewness −67.8 %, std −0.5 % |
| the mode is the correctly-signed estimator for level | consistent with W9's out-of-sample +0.287 |
| no *additive* correction recovers the dynamics | stands (W9) |
| the disparity field carries **no** dynamic information | **withdrawn** — a weak correctly-signed correlation of ≈ −0.37 exists at 300 s |
| that correlation is statistically established | **no** — smallest p = 0.038 against a corrected threshold of 0.00227, n_eff = 17.7 |

## Why n_eff = 17.7 is the binding constraint

The reference decorrelates over ~267 s, so 27 bins of 300 s hold only **17.7 independent
samples**. The smallest correlation detectable at that sample size, two-sided at p = 0.05, is
**≈ 0.47**. The observed −0.37 is therefore **below the detection floor of the design** — not
absent, but unmeasurable with this session length. Detecting a correlation of 0.37 at 80 %
power requires **n_eff ≈ 55**, i.e. roughly a threefold longer session or a reference that
decorrelates faster.

**That is the real boundary: not "no signal", but "a signal at or below what this session can
detect".**

## The duration a decisive session needs — corrected

The exact figures, from the Fisher-z relation for a two-sided test at p = 0.05:

| quantity | value |
|---|---|
| critical \|ρ\| for significance at n_eff = 17.7 | **0.470** |
| \|ρ\| detectable with 80 % power at n_eff = 17.7 | 0.731 |
| **n_eff needed for ρ = 0.37 at 80 % power** | **60.3** (not 55) |
| n_eff needed for ρ = 0.47 at 80 % power | 38.5 |

The binned series spans 27 × 300 s = 2.25 h and yields n_eff = 17.7, i.e. **one independent
sample per 456 s of elapsed time**. Reaching n_eff = 60 therefore needs

**2.25 h × (60.3 / 17.75) ≈ 7.6 h of binned data** — about **8 hours**, not the ">15 hours"
proposed in review. The correction is a factor of two, and it makes the decisive experiment
cheaper than it appeared.
