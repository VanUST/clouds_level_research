# Temporal alignment between the stereo cameras and the DOL-2 ceilometer

**Status:** complete for 2025-05-23. **Verdict:** the `windowed_correlation`
"best shift" is not a measurement of a time lag. For the roll-corrected run its
per-window maximum correlation is **below the maximum a random re-alignment of the
same two series produces**, in 70–75 % of windows. The metric is searching noise.

This document reports findings; the code and raw numbers live beside it:

| script | output |
|---|---|
| [`experiments/temporal_alignment/q1_shift_scan.py`](../../experiments/temporal_alignment/q1_shift_scan.py) | `q1_shift_scan.json` / `.png` |
| [`q2_cloud_motion.py`](../../experiments/temporal_alignment/q2_cloud_motion.py) | `q2_cloud_motion.json` / `.png` |
| [`q2b_band_sensitivity.py`](../../experiments/temporal_alignment/q2b_band_sensitivity.py) | `q2b_band_sensitivity.json` |
| [`q3_q4_lag_advection.py`](../../experiments/temporal_alignment/q3_q4_lag_advection.py) | `q3_q4_lag_advection.json` / `.png` |
| [`q5_spatial_sampling.py`](../../experiments/temporal_alignment/q5_spatial_sampling.py) | `q5_spatial_sampling.json` / `.png` |

Everything below uses only the roll-corrected run
`20261013T112649Z_evaluate_clouds_2025-05-23_3565c5` unless a comparison is
explicitly named. No measurement data was modified and the pipeline was not
touched; the analysis calls `clouds.metrics.alignment.paired_r` unchanged.

## 1. The best-shift maximum is not real

### 1.1 What the pipeline does, restated

`windowed_correlation` splits the session into 30-min windows 15 min apart (20
windows, 12:00–17:00 MSK), and inside each window scans a time shift in
`[-300, +300]` s at 5-s steps, returning the shift with the **largest** r. The
independent re-scan in this work reproduces the pipeline's own numbers exactly
(stored mean `raw_windows.mean_r` = 0.1887 vs 0.189, `filtered_windows.mean_r` =
0.2093 vs 0.209), so the diagnostics below describe the shipped metric, not a
re-implementation. The ceilometer series inside a window has a lag-1
autocorrelation of **0.906**
(median over windows), so its **152 paired samples carry only ≈3.4 independent
observations**. Taking the maximum of a correlation over 121 shifts of ≈3
effective points is a selection, not a measurement.

### 1.2 Diagnostics of the r(shift) curve

Each window was re-scanned over ±1800 s at 5-s steps **with an overlap gate**
(a shift is scored only while ≥50 % of the window still overlaps; for a 30-min
window that means the trustworthy range is ≈±900 s — beyond it a 3-pair tail can
reach r = 1.0, which was observed and rejected). Table 1, 20 windows per column:

| quantity | raw_h | filtered_h |
|---|---|---|
| mean pipeline (±300 s) peak r | **0.189** | **0.209** |
| mean r at shift 0 | −0.209 | −0.190 |
| mean peak r over the wide scan | 0.484 | 0.607 |
| windows whose global peak lies outside ±300 s | 65 % | 80 % |
| windows whose global peak beats the ±300 s peak | 65 % | 80 % |
| peak exactly at ±300 s | 15 % | 25 % |
| peak within one 5-s step of ±300 s | 25 % | 35 % |
| median half-prominence width | 168 s | 258 s |
| mean number of comparable local maxima (≥peak−0.05) | 1.45 | 1.15 |
| median fraction of the scan within 0.05 of the peak | 2.7 % | 4.3 % |
| median paired samples per window | 152 | 152 |
| **median effective independent samples** | **3.4** | **3.4** |

Two facts already condemn the procedure. (i) The more shifts are searched, the
"better" the answer looks — 0.19 → 0.48 (raw) and 0.21 → 0.61 (filtered) just by
widening the search, the signature of a maximum over noise. (ii) 65–80 % of the
true maxima inside ±1800 s lie outside the pipeline's ±300 s window, so the
pipeline is neither finding the largest correlation nor a physical lag.

**Figure:** `q1_shift_scan.png`, panel (d) shows the 20 r(shift) curves: they
cross zero repeatedly and put their maxima in unrelated places; panel (b) shows
the observed peak falling on or below the shuffled-alignment null.

### 1.3 The decisive test: a shuffled-alignment null

For each window the algorithm series was circularly rotated against the fixed
ceilometer series (120 draws), preserving both marginal distributions and both
autocorrelations, and the **same ±300 s scan** was run on each rotation. The
observed maximum is then compared with the distribution of maxima obtainable
with no real alignment at all:

| quantity | raw_h | filtered_h |
|---|---|---|
| mean null scan-max r (median) | 0.291 | 0.421 |
| mean null scan-max r (95th percentile) | 0.498 | 0.675 |
| windows where observed peak > null median | 30 % | 25 % |
| windows where observed peak > null 95th pct | **5 %** | **0 %** |
| median p-value of the observed peak | 0.70 | 0.75 |
| windows with p < 0.05 | 1/20 | 0/20 |

The filtered series produces **no window** whose peak exceeds the null 95th
percentile; the raw series does so in 1 of 20, exactly the one-in-twenty false
positive expected by chance. The observed peak is not merely insignificant — it
is **lower than the typical random re-alignment** in both series. A metric whose
answer is worse than a shuffle is not measuring agreement.

### 1.4 The boundary red flag, measured

Exactly at ±300 s: **15 % (raw) / 25 % (filtered)** of windows. Within one 5-s
step: 25 % / 35 %. Reconciling with the brief's "≈45 %": no stored run reaches
that. Scanning every `evaluate_clouds_2025-05-23` run's window detail gives
20 % (roll-corrected), 28 % (roll=0) and 30 % (two older runs) exactly at the
boundary; 45 % is not reproducible from the stored detail. The qualitative point
stands, and is stronger in §1.3: the peak is not at the boundary because the lag
is there, but because the curve is noise and the boundary is where the search is
allowed to stop.

## 2. Cloud motion measured from the imagery

Method (no ceilometer; guardrail G1): consecutive devID1 frames ~10 s apart are
high-passed (box blur, 241 px) to remove the static lens vignette, matched with a
96 × 96-pixel template over a ±60-pixel search on a grid inside the purely-sky
disc r < 450 px, and the resulting field is fitted with the exact translating-
plane model for an equidistant fisheye, r = f·θ with f = 654 px/rad. The fit's
two parameters are horizontal translation ÷ height, so the cloud speed is
`h · |(a,b)| / Δt` and the fit rms measures how rigid the layer is.

1798 pairs over 09:00–14:00 UTC:

| quantity | value |
|---|---|
| median image displacement | 10.37 px (IQR 3.46) |
| **cloud speed** | **median 3.27 m/s**, mean 3.53, p10 2.02, p90 6.03, σ 1.55 |
| window-mean speed range | 1.12 – 5.74 m/s |
| within-window pair-to-pair σ | median 1.06 m/s |
| image-frame direction | circular mean 213.3°, circular σ 9.0° |
| per-window direction σ | median 5.8°, max 13.1°, 55 % of windows < 6° |
| single-plane fit rms | median 3.88 px, p90 8.10 (26 % < 2 px, 66 % < 5 px) |
| accepted patches / match score | median 20 patches, median score 0.96 |

**Figure:** `q2_cloud_motion.png`. The speed is unimodal near 3 m/s with a tail
to 7; the direction drifts smoothly from ~225° to ~200° over five hours.

Three controls:

- **Not a camera artefact.** Patches on the static foreground (tree ring,
  r = 600–900 px) give a median displacement of **(−1, −1) px** across 271
  patches; the ~10 px sky displacement is therefore cloud motion, not a global
  image shift or exposure drift.
- **Not the filter band.** Re-measuring 149 pairs with box kernels 121, 241, 481
  and the Gaussian σ = 120 gives medians **3.23–3.26 m/s**, pairwise r ≥ 0.96 and
  ratio 1.00 — the speed is not an artefact of the chosen band.
- **Independent estimator.** Phase correlation over the same disc, restricted to
  the 410 pairs where its response exceeds 0.5, gives a block/phase speed ratio
  of **1.14** (the plane fit measures the centre scale; the disc average is
  biased low by cos²θ, so 1.14 is the expected offset) with r = 0.29. The two
  methods agree in magnitude where phase correlation is usable; the modest r
  reflects genuine per-pair estimator noise, which is why window means and not
  single pairs are quoted.

The field is not always a single rigid layer: the plane-fit rms is comparable to
the displacement at some times (p90 = 8.1 px against ~10 px), consistent with
either multiple layers or rapid cloud evolution. This is reported, not hidden;
it does not rescue the time-lag interpretation below.

## 3. The lag the motion implies

Because the instrument separation is unknown, the honest statement is a function,
not a number. At the measured speeds, `lag = separation / v`:

| separation | lag at v = 3.33 m/s (median window speed) |
|---|---|
| 50 m | 15 s |
| 250 m | 75 s |
| 1000 m | 300 s |
| 2000 m | 601 s |

The observed median |best shift| (205 s filtered, 215 s raw) would require a
separation of **≈683 m / 716 m**, and the largest observed shifts (300 s) a
separation of **≈1000 m** at that speed. Whether ~0.7–1 km is plausible cannot be decided from
this data: there are **no site coordinates anywhere in the repository** and the
ceilometer and camera positions are not recorded. A station with both instruments
on the same mast would give < 20 s, i.e. invisible inside the search; a station
1 km across would give 300 s. The measurement that would settle it is the
geodetic position of both instruments (or a single frame containing a known
common object); neither exists here.

## 4. The advection hypothesis fails directly

If the lag were advection over a fixed baseline **r**, it would obey
`lag = (r · v)/|v|²` and therefore (i) track the measured speed, (ii) reverse sign
when the wind reverses, and (iii) vary smoothly in time, because the measured wind
does. Fitting **r** to the 20 per-window best shifts (two free parameters)
explains essentially nothing:

| series | fitted \|r\| | R² | null-R² median | p |
|---|---|---|---|---|
| filtered | 1759 m | **0.099** | 0.036 | 0.18 |
| raw | 829 m | **0.018** | 0.036 | 0.71 |

The sequence is not smooth either. Adjacent windows jump by

| sequence | mean \|Δshift\| | fraction of jumps > 200 s | lag-1 autocorr |
|---|---|---|---|
| roll-corrected filtered | 230 s | 58 % | +0.10 |
| roll-corrected raw | 253 s | 47 % | −0.11 |
| roll=0 raw | 185 s | 32 % | +0.04 |

and the wind direction, which is smooth to ~9°, is incompatible with a random
±300 s jump between adjacent 15-min windows.

Two further falsifications:

- **The shift is unstable to a numerical change that cannot move a cloud.** The
  roll-corrected and roll=0 runs use identical code and differ only by a 1.5°
  rotation of camera 2 (≈2–3 px of disparity). Their per-window best shifts are
  identical in **20 %** of windows, differ by **168.5 s on average**, and
  correlate at **0.34**. A physical lag would not move by three minutes when the
  calibration is nudged by two pixels.
- **Figure `q3_q4_lag_advection.png`** shows this directly: panel (a) has the
  four best-shift sequences crossing zero and hitting both boundaries at random,
  panel (b) has the measured speed, panel (c) shows |shift| not tracking speed,
  and panel (d) shows the fitted advection model with R² = 0.10 (null p = 0.18).

## 5. Spatial sampling: a real but bounded contribution

The ceilometer samples one vertical column; the metric's 640-px crop measures a
footprint. At the median stereo height of 1837 m:

| quantity | value |
|---|---|
| crop half-angle | 28.0° |
| footprint radius / diameter | 978 m / **1956 m** |
| ceilometer column width | ~10 m at range |
| footprint : column (linear) | **≈ 200 : 1** |
| cloud-texture decorrelation length (1/e) | 82 px = **229 m** |
| footprint diameter in decorrelation lengths | **8.5** |
| advection over one 30-min window | 2.0 – 10.3 km = 9–45 decorrelation lengths |

Within a single frame, the spread of stereo disparity is `std(du) = 2.13 px`
against `std(dv) = 1.26 px` over 38 cached frames; the vertical spread is pure
noise for a rectified pair, so the horizontal excess
`√(2.13² − 1.26²) = 1.72 px` is relief — about **12 % of the 14.25 px median
disparity**, i.e. ≈ ±140 m across the matched part of the footprint. The
frame-to-frame scatter of the median height over the 76-min cached segment is an
IQR of 164 m (8 %).

So the cloud base is **not homogeneous over the footprint** (8.5 decorrelation
lengths across it, and the footprint advances 9–45 of them per window), and a
footprint average cannot be the same quantity as a point column. But this
explains a *decorrelation of fluctuations*, not the positive answer the metric
needs, and it cannot be repaired by any time shift — shifting one series in time
cannot undo a spatial average.

**What distinguishes the two causes.** A time lag is a shift: it should (a) make
r rise to a stable value, (b) reproduce across a 2-pixel calibration change, and
(c) vary smoothly with the measured wind. A sampling mismatch is not a shift: it
keeps r low for *every* shift. The measurements in §1.3 (r below the shuffled
null for every shift), §4 (R² = 0.10, p = 0.18; 20 % reproducibility across the
roll change; random jumps against a smooth wind) and §5 (footprint 8.5
decorrelation lengths, 9–45 per window) all point to the second. The decisive
experiment remains unavailable here: a stereo height computed for the ceilometer's
own azimuth/zenith sub-column, or a second ceilometer inside the camera field of
view, compared against the field mean. That needs the site geometry the
repository does not contain.

## 6. What could not be determined, and why

1. **The instrument separation.** No site coordinates exist anywhere in the repo,
   and it is not derivable from the data. Every lag statement is therefore
   conditional on a separation that must be measured, not assumed.
2. **The true lag beyond ≈±900 s.** A 30-min window cannot support a shift larger
   than about half its length; beyond that the overlap is too small to score. A
   real multi-minute lag could in principle be hidden there, but the ±300–900 s
   region already contains no significant maximum.
3. **The absolute cloud height.** All speeds here use the stored stereo heights,
   which carry the known `f ≈ 654 px/rad` bias (heights ~7 % low, W1.8). This
   scales every speed by the same factor and does not change the conclusions.
4. **The true wind at cloud level.** 2025-05-23 has no usable meteorology; the
   speed and direction are inferred from the image motion itself.
5. **Whether the 1.5× horizontal excess in the within-frame disparity is cloud
   relief or a residual calibration term.** The vertical-disparity test bounds
   match noise but cannot separate the two.
6. **A world bearing for the wind.** The image direction is measured; converting
   it to a compass bearing relies on the rig's handedness convention, which was
   not independently verified here. Bearing figures in the JSON are explicitly
   approximate.
7. **The brief's "≈45 % at the boundary".** Measured: 20 % (roll-corrected),
   up to 30 % across all stored runs. The stated figure is not reproduced.

## 7. Conclusion

The pipeline's windowed correlation is measuring the maximum of a noise curve
over ≈3.4 effective samples per window. Its per-window peak is below a shuffled
null in 70 % (raw) and 75 % (filtered) of windows, and exceeds the null 95th
percentile in 1 of 20 raw windows and 0 of 20 filtered windows. The measured
cloud motion (≈3.3 m/s, smooth, directionally stable) does not predict the shift
sequence, a fixed-baseline advection model explains R² ≤ 0.10 of it, and the
sequence changes by 168 s on average when the calibration is changed by 2 px.
The correct reading of the reported mean windowed r of 0.18–0.33 is: an artefact
of searching a shift and taking the maximum; the fixed-shift r of −0.15 is the
more honest, and it is also not significant because of the same autocorrelation.
Agreement with the ceilometer on 2025-05-23 should not be quoted as a validation
metric until the spatial sampling is addressed.

## Reproduction

```bash
source /home/iustimov/miniconda3/etc/profile.d/conda.sh && conda activate clouds
cd /home/iustimov/clouds/experiments/temporal_alignment
python q1_shift_scan.py          # ~80 s
python q2_cloud_motion.py        # ~7 min
python q2b_band_sensitivity.py   # ~4 min
python q3_q4_lag_advection.py    # ~30 s
python q5_spatial_sampling.py    # ~1 min
```
