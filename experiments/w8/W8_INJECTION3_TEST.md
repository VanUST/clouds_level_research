# W8 injection 3 — the broken-field / sky-hallucination hypothesis, tested on real data
# before building a renderer

The hypothesis: in a broken cloud field the camera's field of view integrates both the true
high cloud and clear sky; RoMa hallucinates a dense field on clear sky at ~44.9 px
(W5.1, the cloudless null); `lowest_mode` locks onto that low-height mode; the reported
height collapses. This predicts the sign of the inversion exactly.

**It has a zero-cost test on data already in hand**, so it was run before any renderer was
built. `experiments/w4/matches_roma_20250523.npz`: 3,904,543 real correspondences
stratified by the ceilometer's own regime.

## Result: the sky-like population is far too small

| regime | n | median du | share > 30 px | share > 38 px |
|---|---|---|---|---|
| low (H1 < 2750) | 2,505,848 | 12.27 px | 1.37 % | 0.84 % |
| high (H1 > 3500) | 1,398,695 | 14.07 px | **3.55 %** | **2.57 %** |

The high regime does carry ~2.6× more matches in the sky-like range. But:

1. **3.55 % is below `lowest_mode`'s own prominence floor of 10 %.** A mode must carry ≥10 %
   of the total weight to qualify, so the sky population cannot become the reported mode.
2. **A 2.6 % population cannot move a median by 1.80 px.** The median is the 50th
   percentile; the tail does not reach it.

**The sky-hallucination route is refuted**, and it was refuted without rendering anything.

## What the test did surface: the distribution does not translate, it reshapes

| percentile | low du | high du | shift |
|---|---|---|---|
| 1 % | −1.88 | −9.07 | **−7.19** |
| 5 % | 6.08 | 2.76 | **−3.31** |
| 10 % | 8.33 | 7.85 | −0.48 |
| 25 % | 10.22 | 11.09 | +0.87 |
| **50 %** | 12.27 | **14.07** | **+1.80** |
| 75 % | 14.33 | 15.65 | +1.32 |
| 90 % | 15.96 | 17.77 | +1.82 |
| 95 % | 17.58 | 24.34 | **+6.76** |
| 99 % | 35.16 | 72.67 | **+37.51** |

Two things are happening at once, and they have different causes:

- **The central mass genuinely shifts by about +1.5 px.** From the 25th to the 90th
  percentile the shift is consistently +0.9 to +1.8 px. This is not a tail artefact: the
  sky population cannot produce it, and the core moves.
- **The tails extend in both directions.** The span between the 1st and 99th percentile goes
  from **37.0 px to 81.7 px, a factor of 2.2**, with the *lower* tail reaching down 7.2 px
  and the upper tail up 37.5 px.

So the high regime's disparity field is **both shifted and roughly twice as dispersed**, and
the shift of the central mass is a separate effect from the tail growth.

The dispersion growth corroborates **Gate 3** independently: the perpendicular epipolar
residual also nearly doubles in the high regime (0.866 → 1.885 px). Two different statistics
on two different components agree that the high regime's matching is more dispersed.

## What this means for the bisection

**Injection 3 as specified is closed.** The mechanism it proposed — a sky mode captured by
the selector — does not occur, for the quantitative reason that the sky-like weight is
2.6–3.6 % against a 10 % prominence floor.

**But it replaces itself with a sharper target**: whatever the high regime is doing, it
(a) shifts the central disparity by ~1.5 px and (b) roughly doubles the dispersion. Any
injection that reproduces the inversion must reproduce *both*, and the dispersion is the
easier of the two to measure and to falsify.

## Revised view of the remaining injections

| injection | status |
|---|---|
| 1. texture scale | cleared |
| 2. 3-D relief | cleared |
| **3. broken field / sky hallucination** | **refuted on real data — weight is 2.6–3.6 % against a 10 % threshold** |
| 3b. **dispersion growth** (new, from this test) | to investigate — the high regime is 2.2× more dispersed |
| 4. atmospheric haze | to run, low prior |

**The next experiment should target dispersion rather than height.** What physical property
of a high cloud makes its matched disparity field twice as wide, while shifting the central
mass by a consistent +1.5 px? That is a better-posed question than the ones exhausted so far,
and it is measurable in both the real data and the wind tunnel.
