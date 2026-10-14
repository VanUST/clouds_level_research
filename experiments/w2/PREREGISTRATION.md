# W2 pre-registration — is the static spatial pattern instrumental?

Written before any w2 code ran, per guardrail G2. The directive that prompted this
experiment was reviewed first; §0 records where it does not match this project, because
those corrections change the experiment.

## 0. Corrections to the directive's premises, verified against the data

| directive says | measured here |
|---|---|
| image naming `img-{stamp}-devID{1,2}.jpg` | **the real format has no dash: `img-{date}T{HH}-{MM}-{SS}devID{n}.jpg`** |
| cameras desynced by ~1–2 s | **median offset 0 s**, p90 0 s, max 5 s (05-22) / 2 s (05-23); 8,224 of 8,633 05-23 frames share a stamp exactly |
| `dv_struct_rms_px` is "already computed by the pipeline — the `epi` block in frame logs" | **no such block and no such metric exist anywhere in `src/clouds/`**; it must be built |
| the rigid foreground (mast, buildings) is available as a control | **it is entirely OUTSIDE the pipeline's 640×640 crop** — the crop is pure zenith sky. The control can only be run on the FULL frames, and the pose is then estimated at zenith angles of 60–90° and applied near the zenith: an extrapolation (§5) |
| "the affine is the wrong MODEL (infinity-only); the essential matrix is the correct model" | **not well-posed as stated.** The affine is the *rectification*; the baseline enters the *triangulation* separately, and the pipeline already triangulates with it. Rectifying from an infinity calibration and triangulating with the baseline is standard stereo. The testable question is whether the rectification is *good enough*, not whether its model class is wrong |

## 1. Hypothesis

**H0 (null).** The static spatial pattern is a calibration error, and a relative pose
estimated independently from the rigid foreground will collapse it.

**H1.** Wrong principal point. **H2.** Relative focal length. **H3.** Cloud-climatology
confound.

The foreground is rigid and cloud-independent, so it separates H1/H2 from H3 directly.

## 2. Pre-registered metric

**`dv_struct_rms_px`** — the RMS of the smooth (position-dependent) component of the
vertical disparity residual, in pixels. Built here, not taken from the pipeline.

For each frame: fit a 2-D polynomial of order 3 to `dv(u,v)` over the cloud matches; the
fitted surface is the structured component, the residual is the random component.

- Primary: `collapse_ratio` = `dv_struct(test) / dv_struct(null)`, per frame.
- Secondary: height-space surface RMS, for reporting only.

**Depth-invariance.** `dv` is a pixel quantity and does not scale with cloud height;
a height-space RMS does, which is why it is secondary.

## 3. Pre-registered gates and outcomes

| gate | condition | failure means |
|---|---|---|
| G-A foreground | ≥50 inliers and epipolar RMS < 2.0 px, on ≥5 frames | insufficient foreground texture |
| G-B sun | `\|\|angle(R_fg, R_sun)\|\|` within the sun's own 95 % CI | intrinsics error; **halt, do not touch clouds** |
| G-C drift | `\|\|angle(R_fg(05-22), R_fg(05-23))\|\|` < threshold | rig moved; comparisons across sessions invalid |
| G-D collapse | median `collapse_ratio` < 0.5 **and** paired sign test p < 0.05 | pattern is not primarily extrinsic |

Outcomes: `COLLAPSED` / `NOT_COLLAPSED` (`ratio > 0.8`) / `INCONCLUSIVE`.

## 4. Period, config, and what would falsify

- Diagnostic session 2025-05-22 (cloudless, sun + foreground). Measurement session 2025-05-23.
- Config `experiments/w2/config.yaml`; RoMa with certainty 0.05 for clouds, SIFT for the
  foreground; f = 654 px/rad; cubic distortion a = −0.30.
- **Falsification:** if `collapse_ratio` > 0.8 the pattern is not primarily extrinsic and
  the affine-rectification hypothesis is dead. That is a result, not a failure.

## 5. Known limits, stated before the result

1. **The foreground is outside the processed crop.** The pose is estimated at zenith
   angles 60–90° and applied at the zenith. A rigid pose is global, but principal-point
   and distortion errors are not — so a positive result constrains the periphery, not
   necessarily the zenith.
2. **The baseline (45 m) is assumed and is not needed for the primary metric**, which is
   in pixels and scale-invariant (directive D7). Only the location of the epipole depends
   on the translation direction, not its magnitude.
3. **One cloudy session.** No independent-session validation exists.
4. **H3 cannot be excluded with certainty.** A clean foreground and a patterned sky is
   *consistent* with a real spatial structure in the clouds; it does not prove it.
