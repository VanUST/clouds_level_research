# W2 result — the foreground control is not available on this data

**Outcome: `INSUFFICIENT_FOREGROUND`.** Pre-registered Gate G-A was not met, so the
diagnostic halted before any cloud was touched, exactly as the directive's fail-fast
protocol requires. The static pattern of R-15/R-16 therefore remains **not attributable**.

## What was done

The rigid foreground was located and confirmed: a lattice mast, a brick building and roof
solar panels, all at rows ≥ 1560 of the 1920×1920 frame (zenith angle ≈ 50–90°).
`experiments/w2/foreground_pose.py` estimates the residual relative rotation from the
shipped infinity rectification by building unit rays with the measured cubic distortion
and solving for the essential matrix from RoMa correspondences.

## Why it failed

Two independent matching approaches were tried and both failed on measurement.

**SIFT** (the directive's suggestion). With the foreground mask applied and 2,476/989
keypoints detected in the two frames:

| ratio test | matches |
|---|---|
| 0.75 | **11** |
| 0.85 | 109 |
| 0.90 | 327 |

Almost every correspondence is ambiguous at any defensible threshold. The cause is
geometric: a building at a few hundred metres seen from a 45 m baseline has a parallax of
**~200 px**, and no descriptor matcher bridges that.

**RoMa** (the matcher validated for this rig). On the foreground band:

| certainty threshold | matches |
|---|---|
| 0.9 / 0.7 / 0.5 | **0** |
| 0.2 | 35 (32 inside the mask) |

and on the building region alone:

| certainty threshold | stride | matches | disparity IQR |
|---|---|---|---|
| 0.5 / 0.3 | any | **0** | — |
| 0.15 | 4 | 246 | **91 px** |
| 0.15 | 6 | 112 | **93 px** |

At the only threshold that returns anything, the disparity is **not coherent** — an
interquartile range of 91 px on a ~247 px displacement. That is not a pose measurement;
it is noise. The essential matrix fitted to it produced 26 inliers and an impossible
epipolar residual, which is how the failure announced itself before being traced here.

**The measured foreground disparity is 87–247 px.** At that displacement the two cameras
see the building from viewpoints ~20–40° apart, so local appearance changes too much for
descriptor or dense matching to recover a reliable correspondence set.

## What this means

- **The one cloud-independent geometric control in this dataset is not usable.** The
  mast and building are rigid and cloud-free, exactly as the directive says, but they
  cannot be matched.
- **The H1/H2/H3 question is therefore unresolved by this route.** The static pattern
  remains a measured fact (R-15: r = 0.98 across session halves; R-16: ~16 %
  horizontal scale) whose cause is still unattributed.
- **The directive's premise that the foreground is available for calibration does not
  hold here.** The reason is specific and quantified, not a matter of effort.

## What was NOT done, and why

Phases 3–5 (sun comparison, drift, cloud collapse) were **not** run. The pre-registered
protocol halts at G-A, and running them would have required a pose that does not exist.
A cloud-collapse number computed from a 26-inlier degenerate essential matrix would have
been worse than no number.

## Pre-registered gates, as executed

| gate | condition | outcome |
|---|---|---|
| G-A foreground | ≥50 inliers, epipolar RMS < 2.0 px, ≥5 frames | **FAILED** |
| G-B sun | — | not reached |
| G-C drift | — | not reached |
| G-D collapse | — | not reached |

## What could still attribute the pattern

1. **A principal-point / focal-ratio self-calibration from the cloud field itself**,
   which is what Steps 3–4 address indirectly and what the ~0.3 % focal-ratio estimate
   (R-16) already measures. That is a parameter estimate, not a control.
2. **A second site or session with an accessible rigid target** at a stereo-appropriate
   distance. The present rig's foreground is too close and too oblique.
3. **A computed sun-based rectification** on the clear day, which is the only infinity
   reference that worked (R-12 gave a 1.34° azimuth residual there), applied as a
   rotation anchor rather than a full pose.

## Recommendation

Do not repeat the foreground route on this dataset. If the pattern must be attributed,
the honest options are a controlled calibration target or an instrumented session — not
another attempt with a different matcher, because the limit is the 45 m baseline against
a close, oblique foreground, not the matching algorithm.
