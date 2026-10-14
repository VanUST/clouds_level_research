# W4 gates 3 and 4 — and the test of the multi-layer hypothesis

Three measurements: the epipolar residual by regime, the LoFTR cross-check, and — because
the multi-layer hypothesis makes a concrete prediction — a direct test of whether the
camera is locked on a different layer than the ceilometer.

---

## Gate 3 — FAILS, and my own prediction of the gate was wrong

I predicted before running it that this gate would pass tautologically, because the shift
under study is horizontal and the epipolar lines of a rectified pair are horizontal. **That
prediction was wrong**, and the measurement is more informative for it.

| component | low regime | high regime | shift | KS p |
|---|---|---|---|---|
| **along** (horizontal `du`) | 12.273 px | 14.070 px | **+1.798 px** | ≈ 0 |
| **across** (perpendicular `dv`) | 0.866 px | **1.885 px** | **+1.019 px** | ≈ 0 |
| perpendicular p90 | 3.352 px | **6.327 px** | — | — |

**The perpendicular epipolar residual is nearly twice as large in the high regime, and its
90th percentile nearly doubles.** The shift is therefore *not* purely along the epipolar
line: the high regime's correspondences satisfy the stereo geometry *less* well, not
equally well.

**Gate 3 verdict: FAIL** by the pre-registered rule — the residual distributions differ
significantly.

### The test that carries the real weight

The reference itself makes a quantitative prediction: if the cloud genuinely moved from the
low to the high regime, the parallax must fall from `f·B/H_low` to `f·B/H_high`.

| | value |
|---|---|
| reference medians | 2250 m → 3790 m |
| corresponding parallax | 13.08 px → 7.77 px |
| **predicted shift** | **−5.315 px** |
| **observed shift** | **+1.798 px** |
| discrepancy | **+7.11 px (134 % of the predicted magnitude)** |

The observed shift corresponds to a height change of **2398 m → 2092 m (−306 m)** while the
reference reports **2250 m → 3790 m (+1540 m)**. **The sign is opposite and the magnitude
is 15 % of the prediction.** The observed displacement is not a depth change to the
reported height.

---

## Gate 4 — same direction, but 6.5× smaller

LoFTR on the same frames, same preparation. Note its population *does* vary with regime
(2,463 → 2,775 matches/frame), unlike RoMa's fixed grid.

| matcher | du shift (high − low) | 95 % CI | dv shift | matches/frame low → high |
|---|---|---|---|---|
| **RoMa** | **+1.798 px** | KS p ≈ 0 | +1.019 px | 11,655 → 11,656 (fixed) |
| **LoFTR** | **+0.275 px** | **[+0.112, +0.401]** | +0.145 px | 2,463 → 2,775 (+13 %) |

Bootstrapped over **frames**, the independent unit — not over matches, which would treat
~12,000 correlated correspondences from one sky as 12,000 observations.

**Gate 4 verdict: PASS on sign, FAIL on magnitude.** LoFTR's CI excludes zero, so the
*existence and direction* of the shift is matcher-independent. But its magnitude is 6.5×
smaller, so the *size* of the effect is representation-dependent.

This is an important qualification: the phenomenon is a property of the imagery in
direction, and a property of RoMa's regression in size.

---

## The multi-layer hypothesis, tested

The hypothesis predicts: the ceilometer reports a thin high layer while the camera matches
a thicker lower one, so the high regime should show a *larger* disparity. That sign matches
the observation. It also predicts a level relationship, which is directly testable.

| regime | n | camera estimate | ceilometer H1 | difference |
|---|---|---|---|---|
| **ceilometer LOW** (H1 < 2750) | 215 | **2361 m** | **2250 m** | **+111 m** |
| ceilometer HIGH (H1 > 3500) | 120 | 2064 m | 3790 m | −1726 m |

**In the low regime the two instruments agree to 111 m.** That is the first genuine level
agreement found anywhere in this project, and it is not trivial — the camera's estimate
comes from the imagery and the reference from a lidar.

The camera's whole distribution (median **2244 m**, IQR 522 m) sits inside the ceilometer's
**lower** population (2250–2750 m) rather than its upper one (3750–4250 m).

### But the hypothesis does not explain the inversion

| test | result |
|---|---|
| within the low regime, where both should see the same layer | rank correlation **−0.473 (p = 2.3e−13)** — still inverted |
| median absolute difference, low regime | 299 m |
| frames where the camera reports > 3000 m | 12.6 % (so it is not rigidly locked to one layer) |

**So the multi-layer mismatch explains the LEVEL and not the INVERSION.**

---

## Where this leaves the investigation

| candidate | status after gates 1–4 |
|---|---|
| certainty / hallucination | refuted (gate 1) |
| incoherent noise | refuted (gate 2) |
| calibration / geometry | refuted (W3) |
| population size | refuted — identical rate |
| purely along-epipolar depth change | **refuted (gate 3)** — the perpendicular residual degrades |
| a RoMa-specific artefact | **partly refuted (gate 4)** — LoFTR sees the same sign, but 6.5× smaller |
| **target identity / layering** | **level explained (+111 m in the low regime), inversion NOT explained (−0.473 within that regime)** |

**The honest summary.** The camera's *level* is consistent with the ceilometer's lower
cloud population to 111 m — which is either a real validation or a coincidence that must be
checked against the clear day, where no cloud exists and nothing should agree. And the
*inversion* survives every control applied so far, including the one that should have
removed it. The next test is therefore not another consistency check but a direct one:
**within the low regime, does the camera's variation track anything in the reference's own
data at the matching temporal resolution** — or is the residual variation, at ~300 m,
simply below the instruments' ability to agree?
