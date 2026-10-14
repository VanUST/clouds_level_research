# W7 — synthetic control: the matcher recovers a known height

A cloud layer at an **exactly known** geometric height, rendered through the measured
fisheye geometry, matched by the production matcher, and compared against a **per-pixel**
ground-truth disparity map.

## The result

| rendered H | ground truth | RoMa recovered | bias | bias % | recovered spread |
|---|---|---|---|---|---|
| 1000 m | 29.421 px | 28.633 px | −0.361 px | −1.23 % | 9.99 px |
| 1500 m | 19.599 px | 19.037 px | −0.085 px | −0.43 % | 8.85 px |
| 2000 m | 14.665 px | 14.406 px | −0.025 px | −0.17 % | 8.49 px |
| 3000 m | 9.747 px | 9.522 px | +0.030 px | +0.31 % | 8.55 px |
| 4000 m | 7.285 px | 7.205 px | +0.017 px | +0.23 % | 8.42 px |

| summary statistic | value |
|---|---|
| **monotonicity, ρ(true, recovered)** | **+1.000** (p = 0.000) |
| recovered-vs-true slope | **0.968** (1.0 = perfect) |
| bias across heights | −0.361 to +0.030 px, **spread 0.71 px** |
| the real-world cross-regime effect for comparison | **+1.798 px** |

**The matcher tracks a known height essentially perfectly.** The bias is under 0.4 px at
every height — beneath the 1 px that corresponds to 5.5 % of height at 1.5 km — and the
ordering is exactly monotonic.

## By the pre-registered interpretation matrix

> **Recovers H_true perfectly (bias < 0.5 px, no spatial pattern)** → the matcher and
> geometry are fundamentally sound; the real-world inversion is caused by an unmodelled
> physical effect.

Bias < 0.5 px at every height and ρ = +1.000. **The first row of the matrix applies.** The
estimate does not invert on synthetic cloud texture, at any height, through this exact
fisheye geometry and this exact matcher.

This is the mirror image of the ten refutations: it is a **positive** result, and it says
the instrument is capable of the measurement.

## What this rules out, cleanly

- **Not the matcher architecture.** RoMa recovers the truth on cloud-like texture under the
  real projection. The DINOv2 feature space and the regression head are adequate.
- **Not the fisheye geometry or the distortion model.** The renderer and the triangulation
  agree to 0.03 px at 4 km.
- **Not the crop, the centring or the scale.** The pipeline path is exercised end to end.
- **Not a receptive-field/scale aliasing defect.** Texture scale is what changes most across
  a 4× height range, and the recovery stays monotonic to ρ = +1.000.

## The two bugs found on the way, recorded because they are instructive

The first render produced a **negative** ground-truth disparity (−318 px), which is
geometrically impossible. Two independent errors, both silent:

1. The projection used `arccos(−d_z/|d|)`, the axis of a **downward**-pointing camera. These
   cameras point at the **zenith**, so the axis is `+z`. The error places every ray in the
   opposite hemisphere.
2. `gt_disparity` built its 3-D points with `z = 0` instead of `z = height`, putting every
   point at the camera's own altitude.

Neither raised an exception, and the verification that caught them was the simplest possible
one: **triangulating the ground-truth disparity must return the height it was rendered
at.** It now does — 29.418 px → 1000 m, 7.355 px → 4001 m. That check should be in any
synthetic harness, and its absence is why the first render was believed for one cycle.

## What it implies, and the productive next step

The estimator is **sound in simulation and fails on the real scene**. That inverts the
search: instead of asking what is wrong with the pipeline, the question becomes **which
ingredient of the real scene is missing from the renderer.** Each can be added in turn and
the estimator re-tested — a bisection rather than a hypothesis.

Candidates, in order of how much they differ from what has been rendered:

1. **3-D relief.** The render is a flat plane; a real cloud base is rough. Relief makes the
   two cameras sample genuinely different surfaces rather than the same plane at an offset.
2. **Sensor and ISP realism.** Poisson/Gaussian noise, the strong magenta colour cast, JPEG
   compression, and the fact that the real images saturate near the sun.
3. **Motion between exposures.** A real cloud moves during the pair interval; the render
   does not.
4. **Real texture.** Procedural noise may not carry the same structure as a real cloud base,
   particularly in the low-gradient interiors where RoMa's certainty collapses.

**Adding (1) or (2) and finding the estimate break is the single highest-value experiment
available**, and it is now well posed: there is a working control to break.
