# W8 injection 7 — phase aliasing: REFUTED on real data

The last untested physical hypothesis: the matcher locks onto a different period of a
repetitive cloud billow, producing a discrete disparity jump.

**Refuted before any renderer work, on 3.9 M real correspondences**, in the same way as the
sky-hallucination route.

## Two grounds

**1. The stated magnitude is two orders of magnitude off.** A discrete **+1.8 px** jump needs
a dominant texture wavelength of ~1.8 px. A 100–1000 m cloud feature at 2–4 km subtends
**100–1000 px** at f = 654 px/rad. There is no repetitive structure at the pixel scale in
these images.

**2. There is no periodic structure in the disparity histogram at all.**

| regime | period peaks above the 0.05 autocorrelation floor | share beyond median+50 px |
|---|---|---|
| low | **none** | 0.208 % |
| high | **none** | **1.245 %** |

A period-jump mechanism would show equally spaced peaks, and the excess tail would sit near
an integer multiple of the period. Neither regime shows any periodicity whatsoever.

## What survives

The tail is **6× heavier** in the high regime (1.245 % against 0.208 % beyond median+50 px).
Something does generate a heavy tail — but it is not a periodic aliasing mechanism, and no
tested mechanism reproduces it.

## Standing after eleven refutations

| # | hypothesis | verdict |
|---|---|---|
| 1 | certainty / low-confidence hallucination | refuted |
| 2 | incoherent matching noise | refuted |
| 3 | calibration / geometry | refuted |
| 4 | population size | refuted |
| 5 | pure along-epipolar depth change | refuted |
| 6 | a RoMa-specific artefact | partly refuted (LoFTR agrees in sign) |
| 7 | target-identity layering | refuted as an explanation of the sign |
| 8 | camera desync / motion | refuted |
| 9 | a fixed pipeline artefact | refuted |
| 10 | non-Lambertian photometry | refuted |
| 11 | photometric superposition | refuted |
| 12 | phase aliasing | **refuted** |

**The cause of the inversion remains unattributed.** It is measured, robust, coherent and
confident — and no mechanism tested accounts for it.
