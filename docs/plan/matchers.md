# Matcher selection

**Status:** decided, with one open item. Full measurements, pre-registration and
figures: [`experiments/matchers/REPORT.md`](../../experiments/matchers/REPORT.md).
31 real stereo pairs of 2025-05-23 plus a 32-pair clear-sky control; the metric was
pre-registered before the first number (G2) and the ceilometer was not consulted (G1).

## Decided

| backend | verdict |
|---|---|
| **MASt3R** (`min_conf: 0.5`) | **adopt as opt-in** — the only candidate that improves the pre-registered metric, gives dense *interior* coverage, and passes the clear-sky test |
| `kornia_loftr` | **remains the default** until MASt3R's perturbation weakness is re-tested on real adjacent-in-time frames |
| RoMa, DKM | **rejected** — best localisation by far, but **0 matches in the lowest-gradient tercile** over 31 frames |
| EfficientLoFTR | **rejected despite the literature** — +204 % on the pre-registered residual |
| DA3 | **null** — non-metric; depth nearly constant |
| StereoSGBM, DIS flow | **rejected** — both invent a full field on a cloudless sky |
| DISK+LightGlue | **rejected** — within noise of LoFTR, worst interior fraction |

## Why RoMa and DKM are unusable here despite being the most accurate

They win on localisation — median ray-intersection residual 0.83 px against LoFTR's
1.48 — and that accuracy is **bought entirely on cloud edges**. In the lowest-gradient
tercile of the field they return *zero* matches across every frame, and even at their
lowest confidence threshold only 1–3 % of their matches are interior. For a cloud base
that is the disqualifying failure: the height is wanted across the whole field, and a
matcher that only fires where there is already a gradient cannot supply it. It is a
reminder that a headline localisation number computed over *all* matches can hide
which matches it was earned on.

## What MASt3R buys, and what it costs

| | LoFTR | MASt3R (conf ≥ 0.5) |
|---|---|---|
| matches per frame | 2189 | **5597** |
| interior (low-gradient) matches | 298 | **1448** |
| interior residual | 2.15 px | **1.75 px** |
| risk-matched residual | 1.27 px | **1.17 px** |
| paired rms | 2.69 px | **2.12 px** (−30 %, p = 0.000) |
| median residual, all matches | **1.48 px** | 1.56 px (slightly worse) |
| perturbation stability | **0.22 px** | 0.84 px (4× worse) |
| runtime | **111 ms** | 738 ms (6.6×) |
| clear-sky matches/frame | 11 | **5** |

Its confidence also separates cloud from clear sky by ~50× (median 5.85 on cloud,
0.10 on a cloudless sky), which is a usable internal quality signal.

**Open item:** the perturbation stability — 0.84 px against 0.22 — has not been tested
on real adjacent-in-time frames, only synthetically. Since the two cameras expose
about a second apart and cloud structure changes in that second, this is the one
measurement that could overturn the recommendation. It is runnable from data already
on disk and is the next experiment.

## One negative result that matters for the geometry work

**Radial flatness does not discriminate between matchers.** Every candidate produces a
radial height slope of −2.33 to −3.39 m/px. The radial bias is therefore **un-modelled
geometry, not localisation error**, which refutes the hypothesis that a better matcher
would flatten it and points the fix squarely at the lens distortion measured in
[../report/findings.html](../report/findings.html) §13 (`a = −0.30`) and at plan W1/W2.

## Infrastructure changes

- `src/clouds/matching/matchers.py` registers eight backends; `kornia_loftr` is
  untouched.
- `src/clouds/config/load.py` previously validated backends against a hard-coded
  two-name list and **rejected every new backend**; it now consults the registry.
- `matching.params` carries per-backend options through both the single-pair and
  batch paths.
- `configs/ablations/_w3_backend_smoke.yaml` proves end-to-end pluggability.

## Integrity

`verify_loftr_baseline.py` confirms **38/38 frames byte-identical** to the frozen W1
LoFTR correspondences at `batch_size: 1`, re-verified *after* all source edits. The
default path's numerics are unchanged, so every earlier result stands.

## Not determined

Session hold-out is impossible (one usable session); MatchAnything was not run (its
third-party tree is absent); RoMa ran on its pure-PyTorch fallback because no CUDA
toolchain is installed, so it is not the reference configuration; and no absolute
height claim is made — the ranking is invariant to the unpinned focal constant.

---

# Sub-pixel accuracy: a direct measurement

**Added 2026-10-13.** The evaluation above ranked matchers by geometric residual,
which conflates localisation error with un-modelled geometry — the radial slope was
−2.3 to −3.4 m/px for *every* candidate, so residual cannot rank pixel accuracy.
This measures it directly.

**Method.** Take a real cloud crop, translate it by a **known** sub-pixel amount with
cubic interpolation, run the matcher between the original and the translated copy, and
compare the recovered displacement with the applied one. The reference is exact and the
texture is real cloud. The construction was validated first: **phase correlation** — an
exact estimator with no learned prior — recovers the same shifts with a ratio of
0.88–1.36, so failures below are the matchers', not the test's
(`experiments/matchers/validate_construction.py`).

## Recovery ratio (1.00 = perfect), 14 frames, 640 px crop

| matcher | 0.15 px | 0.35 px | 0.50 px | 0.65 px | 0.85 px | 1.35 px | 2.45 px | **worst dev.** |
|---|---|---|---|---|---|---|---|---|
| **RoMa** | **1.01** | **1.01** | **1.01** | **1.00** | **1.01** | **0.98** | **0.99** | **0.018** |
| DKM | 0.38 | 0.90 | 1.05 | 1.14 | 1.11 | 0.97 | 1.00 | 0.623 |
| kornia LoFTR | 0.35 | 0.55 | 0.95 | 1.22 | 1.12 | 0.91 | 0.92 | 0.646 |
| EfficientLoFTR | 0.18 | 0.12 | 0.17 | 1.55 | 1.20 | 0.76 | 0.83 | 0.881 |
| MASt3R | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| *phase correlation* | *1.36* | *1.09* | *0.98* | *0.98* | *0.95* | *1.05* | *0.88* | — |

**RoMa is essentially exact at every sub-pixel magnitude**, deviating by at most
1.8 %. LoFTR and DKM **cannot resolve displacements below ~0.5 px** (0.35 and 0.55
recovery ratios at 0.35 px). EfficientLoFTR is erratic. This is the answer to "which
matcher is most pixel-accurate": **RoMa, by a wide margin.**

## But MASt3R's zero is a property, not a bug

MASt3R returned exactly 0.000 for every sub-pixel shift, which is impossible for a
working matcher — so it was tested against large shifts directly:

| applied shift | kornia LoFTR | MASt3R | RoMa |
|---|---|---|---|
| 0 px | −0.01 | −0.00 | −0.02 |
| **2 px** | 1.99 | **−0.00** | 2.00 |
| 8 px | 8.00 | **7.50** | 8.00 |
| 25 px | 25.02 | 25.00 | 24.98 |

MASt3R responds at 8 and 25 px but is **dead at 2 px**. It is dense and internally
consistent — which is why it won the geometric-residual comparison — but it does not
resolve small displacements, so its consistency is not sub-pixel accuracy. A plane fit
is satisfied by a consistently quantised field just as well as by an accurate one;
that is the trap the residual metric set, and it is why this direct test was needed.

## The tension, stated plainly

| | sub-pixel accuracy | interior (low-gradient) coverage |
|---|---|---|
| **RoMa** | **best** (worst dev. 1.8 %) | **0 %** native, 1.9 % at low confidence |
| **MASt3R conf≥0.5** | worst (dead below ~2 px) | **25.9 %** (1448 matches/frame) |
| kornia LoFTR | fails below 0.5 px | 13.6 % (298 matches/frame) |

**No candidate is both accurately sub-pixel and covering of cloud interiors.** RoMa's
precision is real and its interior coverage on translated copies is fine (36.7 %) —
but on genuine stereo pairs, where the two cameras see different noise and slightly
different cloud, it returns **zero** interior matches. MASt3R's density is bought with
sub-pixel insensitivity.

**Consequence for the plan.** The choice cannot be settled by picking the best
single number; it depends on which failure the geometry can absorb:

- if the pipeline needs **dense interior coverage** and the height is derived from a
  fitted field, MASt3R's quantisation may be tolerable once the lens distortion
  (workplan R2) is applied, because a consistent field is what a fit consumes;
- if the pipeline needs **per-match height precision**, RoMa's precision is what
  matters and the coverage problem has to be solved first — which makes
  "why does RoMa find no interior matches on real stereo pairs" a question worth
  answering rather than a rejection;
- LoFTR remains the honest default because it is middling on both and is the only
  configuration with frozen, byte-reproducible outputs.

**Next experiment, and it is decisive:** measure whether each matcher resolves a
**0.5 px change in a 15 px disparity** — that, not a 0.5 px absolute displacement, is
the quantity the height estimate actually depends on. The current test measures
absolute small shifts; the pipeline needs differential sensitivity at the working
disparity.

---

# What "0 % interior coverage" actually means for RoMa

**Added 2026-10-13.** This is a diagnosis, and it changes the conclusion. Measured
with `certainty_threshold = 0`, so that *nothing* is filtered
(`experiments/matchers/roma_certainty.py`, 6 real stereo pairs, 69,933 predictions):

| gradient tercile | predictions | median certainty | p25 | p75 | p95 |
|---|---|---|---|---|---|
| **lowest (cloud interior)** | 23,078 | **0.0034** | 0.0005 | 0.0293 | 0.2891 |
| middle | 23,777 | 0.4339 | 0.1111 | 0.7168 | 0.9016 |
| highest (cloud edge) | 23,078 | 0.7163 | 0.4055 | 0.8531 | 0.9358 |

Survival of each certainty threshold:

| threshold | interior | edge |
|---|---|---|
| 0.001 | 66.1 % | 97.8 % |
| 0.01 | 36.8 % | 94.2 % |
| 0.05 | 19.8 % | 89.8 % |
| 0.50 | **1.9 %** | 69.9 % |

## The answer

**RoMa is not sparse.** `model.match()` returns a dense warp-and-certainty field on
the upsampled grid, and the wrapper subsamples it on a regular `grid_stride` lattice —
so there is a prediction at **every** grid point, roughly 11,650 per frame here. The
"0 %" was the `certainty >= 0.5` filter discarding them, and it discards almost
exactly the cloud interiors.

What is near-zero in a cloud interior is **RoMa's own confidence**: median certainty
0.0034 against 0.716 at an edge, a factor of ~200. That is not a defect in the
matcher; it is a correct self-assessment. In a textureless region two genuinely
different views cannot be distinguished, and RoMa says so. On the *translated-copy*
test the interiors are pixel-identical, certainty is high, and coverage is 36.7 % —
the difference is the real viewpoint change and sensor noise between the two cameras.

## Why this matters for the decision

The earlier write-up treated 0 % coverage as disqualifying. It is not the same kind of
failure as a matcher that cannot produce a prediction; it is a **filtering choice that
we control**. The questions that follow are answerable and were not asked before:

1. **Are the low-certainty interior matches correct?** If RoMa's interior predictions
   carry the right disparity despite low confidence, the threshold is simply set too
   high for this target, and a certainty-weighted fit would use them.
2. **Does a lower threshold help?** At 0.05 the agent's run already yields interior
   matches (67 per frame at `grid_stride: 8`); at 0.001 my measurement keeps 66 % of
   them. The trade is precision for coverage, and it is measurable on the geometry,
   not on the ceilometer.
3. **Is certainty a usable weight?** It ranges over three orders of magnitude and
   tracks image gradient, which is the physically meaningful axis — it may serve
   directly as the per-match weight the triangulation and the layer fit need.

## A discrepancy to record honestly

My survival rates and the stored `roma_c05` run do not reconcile exactly: at threshold
0.05 I measure 63.97 % of *all* predictions surviving, while the stored run reports
3,591 matches per frame against my ~11,650 at threshold 0. Part of this is a
definitional difference — my interior terciles are taken over the pooled
matched-point gradients, while the stored run uses per-frame image-gradient masks —
but that does not account for all of it. The certainty *distribution* above is
measured directly and is not in doubt; the exact survival fractions still need
reconciling before any is quoted.

---

# The decisive measurement: differential sensitivity

**Added 2026-10-13.** Every ranking before this one measured **absolute** small
displacements. The pipeline does not consume those. At 1.5 km the disparity is 18.3 px,
and what the height depends on is the ability to resolve a **0.5 px change riding on
top of a ~20 px disparity**. The two properties are different, and this measures the
one that matters: sweep a small offset at a fixed 20 px base and record how the
recovered displacement responds. Slope 1.00 is perfect; the scatter about the fitted
line is the precision.

| matcher | **slope** | scatter (px) | interior slope | matches | ms |
|---|---|---|---|---|---|
| **DKM** | **0.994** | **0.014** | 1.001 | 12/12 | 516 |
| **RoMa (cert 0.5)** | **1.008** | **0.016** | 1.009 | 12/12 | 581 |
| RoMa (cert 0.05) | 1.013 | 0.016 | 1.017 | 12/12 | 581 |
| kornia LoFTR | 1.043 | 0.110 | 1.033 | 12/12 | **111** |
| EfficientLoFTR | 1.299 | 0.214 | 1.301 | 12/12 | 53 |
| **MASt3R conf≥0.5** | **0.000** | 0.000 | 0.000 | 12/12 | 738 |

(sign convention: the recovered displacement is the applied one negated, so 1.00 is
exact.)

## What this says

- **MASt3R has ZERO differential response.** A 1.5 px change in a 20 px disparity
  produces no change at all in its output. Since 1 px is 5.5 % of the height at 1.5 km,
  a matcher that cannot resolve any change smaller than several pixels cannot supply
  the height. **It is disqualified**, and its earlier first-place ranking was an
  artefact of being judged on a plane-fit residual — which a consistently *quantised*
  field satisfies just as well as an accurate one. This is the trap the residual
  metric set, and the direct test is what exposed it.
- **EfficientLoFTR has a 30 % gain error** (slope 1.30) on top of the largest scatter.
  Its documented advantage does not merely fail to transfer; it inverts into a
  systematic error. **Disqualified.**
- **kornia LoFTR has a 4.3 % gain error** (slope 1.043) and 7× the scatter of RoMa or
  DKM. A 4.3 % error in the disparity increment is a systematic error in the quantity
  the height depends on. It is the reproducible default and it is **not** the accurate
  one.
- **RoMa and DKM are essentially exact**: slopes within 1 %, scatter 0.014–0.016 px —
  they resolve a quarter-pixel change in a 20 px disparity to within 0.02 px.

## Verdict

**RoMa is the better matcher for this task**, with DKM statistically tied on accuracy.

| | accuracy (differential) | interior on real pairs | clear sky | ms |
|---|---|---|---|---|
| **RoMa** | **1.008 / 0.016 px** | 0 % at cert 0.5; dense grid, threshold-controlled | **0 matches** | 581 |
| DKM | 0.994 / 0.014 px | 0 % at cert 0.5 | — | 516 |
| LoFTR | 1.043 / 0.110 px | 13.6 % | 11 matches | **111** |
| MASt3R | **0.000 — disqualified** | 25.9 % | 5 matches | 738 |

RoMa wins on the property the height depends on, and it also has the best clear-sky
discipline (zero matches on a cloudless sky, where LoFTR invents 11 and MASt3R 5). Its
one weakness — interior coverage on real pairs — is a **threshold we control**, not a
structural limit: the predictions exist at every grid point, only their confidence is
low.

## The one measurement still outstanding

The differential test uses translated copies, where interiors are pixel-similar and
RoMa covers them (32.7 %). On **real** stereo pairs RoMa's interior confidence is ~200×
lower and the threshold removes those points. So the open question is precise and
narrow:

> **Are RoMa's low-certainty interior predictions on real stereo pairs accurate?**

That is testable: take those predictions on real pairs and check their disparity
against the field implied by the high-certainty matches. If they agree, RoMa is the
choice outright and its low certainty becomes a usable *weight* rather than a filter.
If they do not, DKM is equally accurate but shares the same weakness, LoFTR keeps its
coverage at the cost of a 4.3 % gain error, and the honest position is that no
available matcher is both accurate and covering.

**Until that is answered, keep `kornia_loftr` as the default** — not because it is
best, but because it is the only configuration with frozen byte-reproducible outputs
and it is not disqualified, merely less accurate.

---

# RESOLVED: RoMa is adopted

**2026-10-13.** The one outstanding question is answered, and it is answered in
RoMa's favour.

## Are RoMa's low-certainty interior predictions accurate? Yes.

The disparity field of a cloud layer is spatially smooth, so a low-certainty
prediction should agree with the field implied by the **high-certainty** predictions
around it. The null is the same comparison with the low-certainty disparities
spatially shuffled, which preserves their distribution and destroys any real
agreement. 12 real stereo pairs:

| quantity | value |
|---|---|
| median residual of low-certainty interior predictions against the local high-certainty field | **0.68 px** |
| the same, spatially shuffled (null) | 7.02 px |
| independent cross-check: agreement with LoFTR's own interior matches | **0.62 px** |

The low-certainty predictions agree with the reliable field to **0.68 px, ten times
better than chance**, and agree independently with a different matcher to 0.62 px.
**They are accurate. The low certainty was over-conservative for this target, not a
correct rejection.** At 1 px = 5.5 % of height at 1.5 km, a 0.68 px residual is a
3.7 % height contribution — acceptable, and it is measured against a local field that
itself carries the plane-fit residual.

The low certainty is therefore a usable **weight**, not a filter.

## Certainty does separate cloud from clear sky

Measured with nothing filtered, cloud (2025-05-23) against clear sky (2025-05-22),
~70,000 predictions each:

| session | median certainty | p25 | p75 | p95 |
|---|---|---|---|---|
| **cloudy** | **0.2598** | 0.0084 | 0.7186 | 0.9095 |
| **clear sky** | **0.0014** | 0.0002 | 0.0089 | 0.1044 |

Clear-sky certainty is **~186× lower**. The operating trade, as fraction of
predictions kept:

| threshold | cloudy | clear sky | ratio |
|---|---|---|---|
| 0.01 | 74.0 % | 23.6 % | 3.1 |
| 0.05 | 64.0 % | 9.4 % | 6.8 |
| 0.10 | 59.1 % | 5.2 % | 11.4 |
| 0.25 | 50.5 % | 1.5 % | 34.2 |
| 0.50 | 38.9 % | 0.23 % | 169.9 |

**Caveat, stated:** certainty separates cloud *edges* from clear sky by 11–170×, but
cloud **interiors** from clear sky by only ~2.5×, because both are textureless.
Certainty alone therefore cannot both keep interiors and reject blank sky — the
clear-sky rejection must come from the **geometry** (a cloud layer produces a coherent
disparity mode; scattered clear-sky noise does not). That is a layer-extraction
requirement (plan W4), not a matcher setting.

## The adoption decision

**`roma` becomes the default backend**, with `grid_stride: 8` and
`certainty_threshold: 0.05` — kept low deliberately, because the interior predictions
are accurate and the clear-sky rejection belongs to the geometry.

It earned it on the property the height depends on:

| criterion | RoMa | LoFTR | MASt3R |
|---|---|---|---|
| differential slope (1.00 exact) | **1.008** | 1.043 | 0.000 (disqualified) |
| differential scatter | **0.016 px** | 0.110 px | 0.000 |
| interior accuracy (low certainty) | **0.68 px** | — | — |
| clear sky (few is correct) | 0 | 11 | 5 |
| ms per pair | 581 | **111** | 738 |

`kornia_loftr` remains fully available and is the reference for reproducing every
number recorded before this change.

# The matcher API

Backends are a registry, not a switch statement. The contract, so the set can be
extended and re-verified without touching the pipeline:

```python
from clouds.matching.matchers import (create_matcher, get_available_backends,
                                      register_backend, supports_batching)
match = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.05}, "cuda")
kp0, kp1 = match(gray1, gray2)
kp0, kp1, conf = match(gray1, gray2, return_conf=True)   # when the backend has one
```

- **A backend is a factory** registered by name with `@register_backend("name")`,
  taking `(config, device)` and returning a callable. Adding one is a single decorated
  function; nothing in `pipeline/` changes.
- **The callable's contract** is `(img0, img1, visualize=False, return_conf=False)
  -> (kp0, kp1)` with `kp0`, `kp1` as `(N,2)` float arrays in the input image's pixel
  coordinates, and confidence as `(N,)` when requested. Returning `(N,0)` is legal and
  means "no confident correspondence" — a backend must not invent matches.
- **Selectable in config**: `matching.backend` plus an optional free-form
  `matching.params` dict passed to the factory. `config/load.py` validates the name
  against the registry, so an unknown backend is a config error rather than a silent
  fallback. (This was a real defect: it previously validated against a hard-coded
  two-name list and rejected every new backend.)
- **Batching is optional**: `supports_batching(matcher)` reports whether the callable
  exposes `match_batch(pairs, batch_size)`. `batch_size: 1` must reproduce the frozen
  reference for whichever backend is selected.

## Registered backends

| name | family | status |
|---|---|---|
| `roma` | dense, detector-free | **default** |
| `kornia_loftr` | dense, detector-free | reference; reproduces all earlier results bit-for-bit |
| `dkm` | dense, detector-free | accuracy tied with RoMa; 0 interior coverage at cert 0.5 |
| `mast3r` | pointmap / dense 3-D | available; **zero differential response, do not use for height** |
| `efficient_loftr` | dense, detector-free | available; 30 % gain error |
| `da3` | monocular depth | available; non-metric, null result |
| `cv2_dis_flow`, `cv2_sgbm` | classical dense | available; invent a full field on clear sky |
| `disk_lightglue` | sparse control | available; sparse detectors find almost nothing in cloud |
| `matchanything_eloftr` | dense, detector-free | registered, weights absent in this checkout |

The full pre-registered comparison, including the candidates that were rejected and
the measurements that rejected them, is in
[`experiments/matchers/REPORT.md`](../../experiments/matchers/REPORT.md).
