# Selecting a matcher for a metrological task

A reusable result, not just a choice. The general principle:

> **Rank candidates on the property the product consumes, measured directly, and
> stratified by the region where the product needs them. A proxy metric can rank a
> useless matcher first.**

Everything below is one worked instance of that principle, on a stereo cloud-base-height
problem. The numbers are measurements and the scripts are in `experiments/matchers/`.

## 1. State what the product consumes

The product is a height derived from a disparity. At 1.5 km the disparity is **18.3 px**,
so **1 px of disparity error is 5.5 % of the height**. The relevant property is
therefore:

> the ability to resolve a **0.5 px change riding on a 20 px disparity** — a
> *differential* sensitivity — not the ability to recover a small absolute displacement,
> and not a goodness-of-fit residual.

Those three are different, and the difference decided this project.

## 2. Why a proxy metric failed, and how it was caught

The first evaluation ranked candidates by a **plane-fit residual** on their
correspondences. It put MASt3R first (rms 2.12 px against LoFTR's 2.69).

Then a direct test of differential sensitivity at the working disparity
(`differential_sensitivity.py`, 12 real frames, base 20 px, offsets 0–1.5 px):

| matcher | slope (1.00 exact) | scatter | verdict |
|---|---|---|---|
| RoMa | 1.008 | 0.016 px | exact |
| DKM | 0.994 | 0.014 px | exact |
| kornia LoFTR | 1.043 | 0.110 px | 4.3 % gain error |
| EfficientLoFTR | 1.299 | 0.214 px | 30 % gain error |
| **MASt3R** | **0.000** | 0.000 | **no response at all** |

**A consistently *quantised* disparity field satisfies a plane fit just as well as an
accurate one.** That is the trap: the residual measures self-consistency, and a matcher
that always returns the same answer is perfectly self-consistent. The proxy metric was
rewarding the defect it was supposed to detect.

**Rule:** state the property the product consumes; if the ranking metric is not that
property, demonstrate that it is a monotone surrogate, or do not use it.

## 3. Stratify by the region that matters

A matcher can be the most accurate on the matches it returns and return none where they
are needed. Splitting each candidate's matches by **image-gradient tercile**:

| matcher | matches/frame | interior matches/frame | interior share |
|---|---|---|---|
| MASt3R | 7589 | 2438 | 32.1 % |
| kornia LoFTR | 2189 | 298 | 13.6 % |
| DISK+LightGlue | 1909 | 126 | 6.6 % |
| **RoMa** | 1259 | **0** | **0.0 %** |
| **DKM** | 699 | **0** | **0.0 %** |

RoMa and DKM had by far the best localisation on their matches (median ray-intersection
residual 0.83 px against LoFTR's 1.48) and covered **none** of the cloud interior, which
is where the height is wanted across the field. **A headline accuracy computed over all
matches hides which matches it was earned on.**

**Rule:** stratify by the axis that matters to the product, and report coverage and
accuracy in that stratum separately from the pooled figure.

## 4. Validate the ground-truth construction before trusting the ranking

The sub-pixel and differential tests build ground truth by translating a real image by a
known amount. If that translation cannot be recovered by an exact estimator, the test is
measuring its own construction.

Using **phase correlation** — no learned prior — on the identical construction:
recovered/applied ratio 0.88–1.36 across 0.15–5.0 px. The construction is sound, so the
failures above are the matchers'.

**Rule:** every synthetic ground truth gets an independent recovery check, reported.

## 5. Distinguish a structural limit from a threshold

"0 % interior coverage" reads as a structural disqualification. It was not:

- reading the wrapper showed the matcher emits a prediction at **every** grid point
  (~11,650 per frame) and the wrapper then applies `certainty >= threshold`;
- measuring the certainty field with the threshold at zero: interior median **0.0034**
  against edge **0.716**, a factor of ~200.

So the coverage number was a property of one arbitrary threshold, not of the matcher.
That reframed the question from "can it?" to "are those discarded predictions right?" —
which was answerable, and the answer was yes:

| | value |
|---|---|
| low-certainty interior predictions vs the local reliable field | **0.68 px** |
| the same, spatially shuffled (null) | **7.02 px** |
| independent agreement with a *different* matcher on the same points | **0.62 px** |

**Rule:** before rejecting a candidate on a metric, read the code path that produced the
metric and check whether it is a limit or a setting. Then test the setting.

## 6. Use a null that can falsify

Every claim above is against a null, and the null is the same computation with the
structure destroyed — a circular roll for timing, a spatial shuffle for agreement. A
comparison without a null is a number, not an effect. In this project a metric that
searched a parameter scored *below* its own null, which is what exposed it.

## 7. Keep the runner-up path open

The selection produced a default and a registry, not a replacement:

```
tools/matchers.py list          # every backend, and whether it loads here
tools/matchers.py check roma    # re-derive the deciding quantities on real frames
```

With ten loadable backends behind one `@register_backend` factory contract, a new
matcher is one decorated function and no pipeline change, and every earlier number stays
reproducible through a frozen reference config. Selection by measurement is only credible
if re-measurement is cheap.

## Outcome for this project

**RoMa** is the default: differential slope 1.008 ± 0.016 px, accurate interior
predictions at low certainty, and zero invented matches on a cloudless sky. **MASt3R is
not used for height** — dense and consistent, with no differential response.
**kornia_loftr is retained** and reproduces every earlier result bit-for-bit.

Full evaluation, including the pre-registration and every rejected candidate with the
measurement that rejected it: [`../../experiments/matchers/REPORT.md`](../../experiments/matchers/REPORT.md).
