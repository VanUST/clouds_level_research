# W8 — physical bisection, injection 1: 3-D relief

A validated synthetic wind tunnel (`experiments/w7/w7_renderer.py`, gate passing exactly)
with one real-atmosphere property added at a time. The pre-registered trigger: **the
recovered height must fail to increase with the true height** — inverting, or compressing by
more than 50 % — across at least three heights. A bias alone is not an inversion.

## What was injected

The flat plane `z = H` became a relieved surface `z = H + relief(x, y)`, with the relief
built from the same procedural generator as the texture so the two share a spectrum.
Roughness was swept from 0 % (the W7 baseline) to **20 % of the cloud height** — a real
cumulus base has roughly 10 % relief, so this overshoots the physical case.

## Result: relief does NOT trigger the failure

| relief | matcher ρ | `lowest_mode` ρ | bias range | triggered |
|---|---|---|---|---|
| 0 % (baseline) | +1.000 | +1.000 | −0.409 to +0.055 px | no |
| 5 % | +1.000 | +1.000 | −0.613 to +0.009 px | no |
| 10 % | +1.000 | +1.000 | −0.974 to −0.006 px | no |
| **20 %** | **+1.000** | **+1.000** | −0.974 to −0.069 px | no |

At 20 % relief, `lowest_mode` reports 1073 / 1487 / 1936 / 3032 / 4018 m for true heights
of 1000 / 1500 / 2000 / 3000 / 4000 m — errors within 75 m, ordering exactly monotonic.

**Three-dimensional relief is cleared.** A rough cloud base, up to twice the physically
expected roughness, does not disturb either the matcher or the estimator.

## Why this was worth running even though it is negative

It closes a mechanism that no flat-plane control can exercise. With a plane, the two cameras
see the *same* surface point and the match is geometrically well posed; with relief they
sample genuinely different points and the disparity field acquires a variance unrelated to
the mean height. That is exactly the class of effect that was suspected, and it is now
measured rather than assumed.

## Injection 1 (texture scale) — already cleared before W8 began

The renderer indexes its texture by **world coordinates** (`tx = (P[:,0] + extent/2)/extent`
with `extent = 4000 m`), so the noise is fixed in **metres**, not pixels. A 352 m world
feature therefore subtends **230 px at 1000 m and 57 px at 4000 m** — a 4× change in
apparent scale, with the matcher recovering the truth at every one. The scale cue was fully
present and the matcher is scale-robust.

## Remaining injections

| injection | status | prior |
|---|---|---|
| 1. texture scale | **cleared** — metric-space texture, 4× apparent-scale range | — |
| 2. 3-D relief | **cleared** — up to 20 % of height, ρ = +1.000 | — |
| 3. broken cloud field / cloud fraction | to run | high — untested, and it is the mechanism by which a 2 km footprint samples differently from a 10 m column |
| 4. atmospheric haze / contrast | to run | **low** — W6 measured the photometric coupling and it runs the wrong way (ρ(brightness, du) = −0.176 while the high regime is brighter) |

**Order revised from the original proposal**: haze last rather than first, on the W6
evidence. Next is the broken field, which is the only remaining mechanism that plausibly
touches what the ceilometer and the camera are actually looking at.
