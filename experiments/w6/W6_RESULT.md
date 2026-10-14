# W6 — phenomenological characterisation: does the disparity field track a photometric
# property of the sky?

`experiments/w6/w6_photometric.py`. 120 frames, ground-truth-free — no reference is used
anywhere in the script.

The hypothesis: clouds are strongly non-Lambertian; the 45 m baseline makes the two
cameras view the same parcel from directions differing by B/H (1.29° at 2000 m, 0.64° at
4000 m); a matcher maximising photometric similarity tracks the apparent brightness peak
rather than the physical feature; that displacement is along the local gradient and
therefore not generally along the epipolar line, which would inflate `dv`.

---

## What was found

### (1) Gradient magnitude — a real but weak coupling to `dv`
| quantity | within-frame ρ |
|---|---|
| \|gradient\| vs **dv** | **+0.117**, positive in 82 % of frames |
| \|gradient\| vs **du** | +0.002, positive in 51 % of frames |

A weak, highly consistent coupling between photometric structure and the *perpendicular*
residual. This is the direction the hypothesis predicts for the `dv` inflation.

### (2) Gradient direction — the specific signature is ABSENT
| quantity | value |
|---|---|
| ρ(\|cos(gradient angle)\|, \|dv\|) | **−0.000 (p = 0.91)** |

A photometric pull is along the gradient, so `dv` should be **largest when the gradient is
perpendicular to the epipolar line**. It is not. **The mechanism's distinguishing prediction
fails**, and this is the strongest single result of the three probes: the coupling in (1)
is not anisotropic in the way a photometric pull requires.

### (3) Brightness, tested WITHIN frame — a real coupling with the WRONG SIGN
Comparing bright and dark regions inside one exposure holds the cloud, the regime, the sun
angle and the atmosphere fixed, so nothing in this test can be a regime-level confound.

| quantity | within-frame ρ |
|---|---|
| brightness vs **du** | **−0.176** (negative in 73 % of frames) |
| brightness vs **dv** | +0.240 (positive in 69 % of frames) |

Brightness deciles within frame:

| decile | 0 | 3 | 4 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|
| median du | 14.23 | 12.61 | 12.44 | 12.77 | 13.47 | **11.52** |

The within-frame brightness span of the disparity is **2.709 px** — larger than the
cross-regime effect under study (**1.798 px**).

**But the sign is wrong.** The coupling is *negative*: brighter regions carry a *lower*
disparity. And the high regime is *brighter* (median intensity 124 against 107). So the
photometric coupling predicts the high regime should have a **lower** disparity, whereas
the observation is **+1.798 px higher**.

---

## Verdict: the photometric hypothesis is refuted as an explanation of the inversion

Three independent reasons, in order of strength:

1. **The sign is opposite.** Bright → lower `du`; the high regime is brighter, and its `du`
   is *higher*. The mechanism is anti-correlated with the effect it was invoked to explain.
2. **The distinguishing prediction fails.** The gradient-direction test — the one signature
   that separates a photometric pull from ordinary matching noise — gives ρ = −0.000.
3. **The magnitude is in the wrong place.** The within-frame brightness contrast (2.709 px)
   exceeds the cross-regime effect (1.798 px), so a photometric mechanism would have to
   produce its largest effect *within* frames and a smaller one across regimes that runs
   the other way.

## What the hypothesis does get right

**It explains part of Gate 3.** `dv` does scale with |gradient| (+0.117) and with brightness
(+0.240). A photometric contribution to the perpendicular residual is real. Gate 3's
doubling of `dv` is therefore partly photometric scatter — but the *systematic* `dv` shift
between regimes is not, and the inversion is not.

## Supporting evidence the hypothesis had, and its status

On the cloudless day RoMa returns a **44.91 px** disparity from a smooth sky gradient —
3.4× the cloudy day's 13.13 px. That is what isophote-tracking on a featureless gradient
looks like, and it remains the hypothesis's best evidence. But a mechanism that is
*active* is not a mechanism that *causes this*, and the sign test is decisive.

## Standing after W6

Ten explanations have now been tested and refuted: certainty, incoherent noise, calibration
geometry, population size, a pure along-epipolar depth change, a RoMa-specific artefact,
target-identity layering, camera desync, a fixed pipeline artefact, and non-Lambertian
photometry.

**The estimator tracks the cloud's height regime with the wrong sign, robustly, coherently,
confidently, at every scale, in both matchers, under both layers, independently of cloud
speed, and independently of local photometry.**

## What the next instrument should be, rather than another hypothesis

The pattern across ten refutations is that every candidate mechanism is a *property of the
scene* or *property of the pipeline*, and each is measurable in isolation — and each is
absent or wrong-signed. That suggests the productive move is not an eleventh hypothesis but
a **controlled input**: a synthetic sequence with a known height, known motion and known
phase function, rendered through the same fisheye geometry, on which the estimator is
required to recover a height it cannot know. Failing that, a target or an instrumented
session. The analysis side of this question is, on present evidence, exhausted.
