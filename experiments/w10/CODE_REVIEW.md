# Code review — two bugs found, each of which changed a conclusion

A systematic review of the analysis code, prompted by the three harness bugs already in the
record. Two further defects were found, both in the W9/W10 pair, and both change published
numbers. A third finding is methodological rather than a defect and is recorded below.

---

## BUG 1 — W9: random k-fold leaks on an autocorrelated series

**What the code did.** `folds = np.array_split(rng.permutation(idx), K_FOLDS)` — a random
partition of the frames into five folds.

**Why it is wrong.** The reference series has **lag-1 autocorrelation 0.934** and decorrelates
over ~267 s, i.e. ~27 frames at the 10 s cadence. Each fold holds ~77 frames. A randomly
held-out frame therefore almost always has its own near-duplicates — frames a few seconds
away, carrying essentially the same observation — in the training set. The correction is
fitted on values that are statistically the same as the values it is then scored on. This is
the textbook failure of random cross-validation on a time series.

**The fix.** Three schemes now run side by side so the leakage is **measured** rather than
asserted: random (kept deliberately, as the leaking baseline), contiguous **blocked** folds,
and **blocked + embargo** — where every training frame within 300 s of the held-out block is
dropped, so a test observation has no near-duplicate in train.

**Result.** Best level ρ by scheme, across all six corrections and three estimators:

| scheme | best LEVEL ρ | best DYNAMICS ρ |
|---|---|---|
| random (leaks) | **+0.286** | +0.052 |
| blocked | +0.214 | +0.046 |
| **blocked + embargo** | **+0.202** | **+0.022** |

**The leakage inflated the headline level result by +0.084, or 42 % relative.** The honest
out-of-sample figure is **+0.202**, not +0.286. The **conclusion is unchanged** — no
correction produces dynamics under any scheme (best +0.022) — but the number reported in the
results index, the workplan and the two HTML reports was too high.

---

## BUG 2 — W10: the reference was binned at the bin *start*, not over the bin

**What the code did.** Each 300 s bin's disparity statistics were paired with
`ref_at(t_bin_start)` — the single ceilometer sample nearest the **start** of the bin.

**Why it is wrong.** A 300 s bin contains ~20 ceilometer samples (15 s cadence). Pairing the
bin's pooled disparity with a single reading from its leading edge mismatches the two series
by up to 300 s, and 150 s on average — comparable to the reference's own 267 s decorrelation
time. The specified design was the **median over the bin**; the implementation took the
nearest sample to its start.

**The fix.** `ref_bin(t0, t1)` returns the median of the reference over the bin's span, and
returns `None` for a bin with fewer than three samples rather than resting a value on one or
two readings.

**Result — and this one reverses a published finding:**

| statistic | before (bin start) | after (bin median) |
|---|---|---|
| median, LEVEL ρ | +0.159 (p = 0.43) | **+0.542 (p = 0.0063)** |
| median, DYNAMICS ρ | **−0.409 (p = 0.038)** | **+0.095 (p = 0.667)** |
| p25, DYNAMICS ρ | −0.400 (p = 0.043) | −0.055 (p = 0.802) |
| skew, LEVEL ρ | −0.230 (p = 0.25) | −0.518 (p = 0.0096) |

**The "weak, correctly-signed dynamics correlation of ≈ −0.37" does not exist.** It was an
artefact of the reference being read from the wrong place in the bin. After the fix the
dynamics correlations are all non-significant, with the largest |ρ| = 0.385 (kurtosis,
p = 0.069) and no consistent sign.

**The shape finding is altered too:**

| quantity | before | after |
|---|---|---|
| median | +13.0 % | **+17.5 %** |
| **mode** | **−11.8 %** | **+1.5 %** |
| **skewness** | −67.8 % | **−83.7 %** |
| standard deviation | −0.5 % | −8.7 % |

The skewness collapse is **larger** than reported (84 % against 68 %), but the headline
"median and mode move in **opposite** directions" **does not survive**: after the fix both
move positively, the median by +17.5 % and the mode by +1.5 %. The shape effect is still
visible in that disparity — an 11× difference in response between two order statistics of the
same distribution — but it is a difference of **magnitude**, not of sign.

---

## Finding 3 — the W10 "correction" of W9 is itself withdrawn

The chain was: W9 found no dynamics → W10 reported a weak correctly-signed −0.37 and stated
that W9 had been "too strong" → **the −0.37 was the bug**. With the reference binned correctly:

**W9's original conclusion was right.** There is no recoverable dynamics signal in the
per-match disparity field at this footprint ratio under any statistic tested, with or without
a correction, at either the per-frame or the binned resolution.

The amendment block in `final-findings.html`, the paragraph in `W10_RESULT.md`, and the
corresponding entries in the workplan and the results index are all reversed below.

---

## What the review did **not** find

Scanned and clean: mutable default arguments (none); bare `except: pass` (none); `astype(int)`
applied to possibly-negative coordinates (all twelve sites are clipped); the closest-approach
triangulation algebra (standard and correct, including the degenerate-denominator guard); the
`pixels_to_rays` fisheye law and its distortion term; the field-mask coordinate conversion
(crop → original → centre, verified numerically); and the background-baseline convention in
`triangulate`, which is consistent with the observed disparity direction in the real data.

## The pattern, again

Both bugs are the same failure as the four already in the record: **a quantity read from
somewhere other than where it was claimed to come from.** A fold that is not independent, a
reference read from the edge of a bin rather than its middle. Neither raised an error, and
neither was visible in the output — the W10 result looked *better* after the bug than before
it, which is exactly why it survived a round.

---

## Finding 4 — a third bug in W9, caught by the guard it motivated

While wiring `assert_disjoint_with_embargo` into the fold construction it fired immediately:

> training index 75 at t=1747978021 lies 20.0 s from held-out index 74, within the 300 s
> embargo

**The cause.** The "blocked + embargo" folds were built as

```
keep = [i for i in idx if (i in set(f_test)) or
        (|t_i - t_start| > EMBARGO and |t_i - t_end| > EMBARGO)]
```

which is the set of frames **allowed into the analysis** — the held-out block *plus* the
distant training frames — and that set was then stored as the **fold**. So `f_test` contained
both the block and its own training data, and `train = idx not in test_i` silently dropped
the distant frames. **The embargo was applied to the test side rather than the training
side**, and the "blocked + embargo" figures reported in the previous round were computed with
a fold definition that neither blocked nor embargoed correctly.

**The fix.** `schemes` carries `(test_folds, embargo_s)`; the purge is applied where the
training set is assembled, dropping every training frame within the embargo of any held-out
frame.

**Corrected figures** (best level ρ across six corrections and three estimators):

| scheme | best LEVEL ρ | best DYNAMICS ρ |
|---|---|---|
| random (leaks) | +0.287 | +0.052 |
| blocked | +0.216 | +0.046 |
| **blocked + embargo** | **+0.217** | **+0.047** |

The leakage inflates the headline by **+0.070** (not +0.084 as first reported), and the
honest figure is **+0.217**. The estimator-dependence survives: the mode is correct-signed
(+0.053 uncorrected, +0.217 corrected) and the **median is inverted at −0.347 to −0.246**
across corrections. The conclusion is unchanged — no correction produces dynamics.

**Worth noting:** once the embargo is applied to the correct side it barely changes the
blocked result (+0.216 → +0.217). The inflation therefore comes almost entirely from
**random** folds on an autocorrelated series, not from the absence of an embargo. That does
not make the embargo pointless — it is what proves the folds are now honest — but the
comparison that matters is random against blocked.

## Performance (requested separately)

Three fixes, all measured:

| fix | before | after |
|---|---|---|
| hoist the `.npz` object arrays out of the 386-frame loop (`d["uv"][i]` re-decompresses the whole 61 MB member on every access) | **258 s** to load | **1.4 s** |
| move the KDE in `lowest_prominent_mode` to the GPU (it is one dense 512 × N computation per call) | 97 ms/call | **1.5 ms/call** (CPU fallback kept, output bit-identical to 0.000000 m) |
| W9 end to end | >50 min | **57 s** |
| W10 end to end | minutes | **7 s** |

The GPU path is exercised only for frames with ≥2000 matches and falls back to the identical
NumPy code otherwise, so a machine without CUDA produces the same answers.
