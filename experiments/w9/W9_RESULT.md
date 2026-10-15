# W9 — the static field diagnosed, and a cross-validated search for an operation that
# removes it

## Part 1 — what the static field depends on

The per-cell median disparity over the session, regressed on candidate predictors:

| predictor | Pearson r | R² |
|---|---|---|
| **u** (horizontal image position) | **+0.814** | **0.663** |
| cos(angle) | +0.767 | 0.588 |
| radius r | +0.499 | 0.249 |
| \|u\| | +0.373 | 0.139 |
| v | −0.239 | 0.057 |

**The static field is not primarily radial.** A linear gradient in `u` accounts for
**66 %** of its variance; radius accounts for **25 %**, and the two are correlated because
`u ≤ r`. The radial profile is monotone — 14.51 px at the centre rising to 16.57 px at
r ≈ 420, a **14 % rise** — but it is a consequence of the u-gradient as much as a radial
law in its own right.

**And the radial structure largely cancels in height.** Applying the measured corrections
(f = 654, a = −0.30, exact triangulation), the correlation between radius and height falls
to **−0.146**, with median heights of 1975 / 1955 / 1930 / 1918 / 1947 m across the radius —
a **3 % spread** against the disparity's 14 %. Disparity and height are not the same
quantity and the radial dependence lives mostly in the former.

## Part 2 — the cross-validated search

Six correction families × three estimators, **5-fold cross-validated**: every correction is
fitted on four fifths of the session and scored on the held-out fifth, so no correction can
be scored on the frames that shaped it. The statistic is height-ordered, so **positive ρ is
the correct sign and negative ρ is the inversion**.

| correction | estimator | LEVEL ρ | DYNAMICS ρ |
|---|---|---|---|
| **per-cell + linear** | **mode** | **+0.217** | −0.038 |
| linear | mode | +0.194 | −0.166 |
| per-cell | mode | +0.179 | +0.011 |
| **none** (baseline) | mode | +0.137 | −0.086 |
| per-radius | mode | +0.149 | −0.042 |
| cell-smooth | mode | +0.147 | −0.161 |
| linear | median | −0.429 | +0.122 |
| per-cell | median | −0.422 | +0.120 |
| **none** (baseline) | median | **−0.347** | −0.010 |
| per-cell + linear | median | −0.424 | +0.106 |

### Two findings, and one of them is a correction to the record

**1. The sign of the level result depends on the estimator.** The **mode** estimator is
**correctly signed** out of sample (+0.053 uncorrected, +0.217 with the best correction,
under blocked+embargo).
The **median** and 25th percentile are **inverted** (−0.41 and −0.30). The project's
"inversion" was measured on median-like statistics, and this shows the inversion is a
property of *those* statistics rather than of the disparity field as a whole. That is a
substantive correction to how the anomaly has been described and it should be carried into
`final-report.md` §2.

**2. NO correction produces out-of-sample dynamics.** The dynamics ρ across all eighteen
combinations lies between **−0.166 and +0.168**, i.e. |ρ| < 0.17 with no consistent sign.
The best level correction (`per-cell + linear`, mode) *worsens* the dynamics relative to no
correction at all (−0.034 against +0.061).

**The static field can therefore be partly removed — but removing it does not recover the
dynamics.** The correction improves the level statistic by +0.116 and does nothing for the
variation, which is the thing a measurement actually requires.

### Why the searched space is a fair test

`per-cell + linear` is the most expressive member: a free offset per 20 × 20 px cell plus a
plane. It is fitted on 80 % of the frames and evaluated on the other 20 %, five times. If
the static field were a fixed additive pattern and the dynamics were underneath it, this
family would recover the dynamics. It does not.

## Verdict

**The static field is diagnosable and partly removable; the dynamics are not recovered by
removing it.** Across six correction families and three estimators, cross-validated on
held-out folds, no operation yields a dynamics correlation above |0.17| with a consistent
sign.

The consequence for the project's central claim is that the anomaly cannot be characterised
as "a static field masking a real signal" without qualification: subtracting the static field
as well as an expressive correction permits does not reveal the signal. It remains
unattributed, and the possibility that the disparity variation carries no information about
the reference at this footprint ratio is not excluded.

## Caveats

- One session, so every fold is a time block of the same day; the folds are not independent
  in the way folds over independent sessions would be.
- The reference is a ten-metre column and the camera footprint is 1956 m at 1837 m.
- The corrections act on disparity space; no attempt was made to correct the height field
  directly, where the radial dependence is already much weaker.
