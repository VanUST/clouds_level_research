# W8 injections 4/5 — photometric superposition: REFUTED on its pre-registered trigger

## The 551 m discrepancy: found, and it was mine

None of the four proposed suspects was correct. The cause was a **call-site default
mismatch between two scripts**:

| script | call | effective roll |
|---|---|---|
| `w7_control.py` | `render_pair(H, tex, tex_extent_m=4000.0)` | **0.0** (the parameter's default) |
| `w8_superposition.py` via `cloud_layers` | `render_pair(..., roll_deg=roll)` with `roll=1.5` | **1.5** |

Rendering the same cloud 1.5° apart gives, on identical everything else:

| roll_deg | median du | `lowest_mode` |
|---|---|---|
| 0.0 | 14.8451 px | **1984 m** |
| 1.5 | 14.2948 px | **1433 m** |

That is the entire 551 m. Verified by direct comparison, not inference.

**The other suspects were each checked and cleared:**
- **alpha-blend LSB noise** — inputs are byte-identical; the renderer's output lies in
  [0.113, 0.957], entirely inside [0,1], so the clip is a no-op and 0 pixels differ.
- **`lowest_mode` bandwidth trap** — RoMa is deterministic (5 calls, median du identical to
  4 decimal places), and the selector is a pure function of its input.
- **JPEG quantisation** — no images are written to disk anywhere in either script.
- **double-warp / frame convention** — both operate on the same pipeline-convention crops.

The control did exactly what a control is for: it refused to reproduce, and the
discrepancy was in the harness rather than the physics.

## The corrected sweep

| H | α | median du | IQR | 1–99 span | `lowest_mode` |
|---|---|---|---|---|---|
| 2000 | **1.00** | 14.85 | 0.73 | 18.96 | **1984** ← reproduces the control |
| 2000 | 0.80 | 14.79 | 0.95 | 20.88 | 1997 |
| 2000 | 0.60 | 14.75 | 1.69 | 20.38 | 1957 |
| 2000 | 0.40 | 14.98 | 2.62 | 20.85 | 1943 |
| 2000 | 0.20 | 13.71 | **6.73** | 32.44 | 1321 |
| 4000 | **1.00** | 7.48 | 0.43 | 6.63 | **3936** |
| 4000 | 0.80 | 7.51 | 0.43 | 8.57 | 3916 |
| 4000 | 0.60 | 7.51 | 0.51 | 9.24 | 3917 |
| 4000 | 0.40 | 7.45 | 0.59 | 9.96 | 3934 |
| 4000 | 0.20 | 7.24 | 1.22 | 12.36 | 4064 |

## Pre-registered trigger: span ×≥1.8 at IQR ×<1.4 — NOT MET anywhere

| H | α | span × | IQR × | triggered |
|---|---|---|---|---|
| 2000 | 0.80 | 1.10 | 1.30 | no |
| 2000 | 0.60 | 1.07 | **2.30** | no |
| 2000 | 0.40 | 1.10 | **3.58** | no |
| 2000 | 0.20 | 1.71 | **9.19** | no |
| 4000 | 0.80 | 1.29 | 1.00 | no |
| 4000 | 0.60 | 1.39 | 1.20 | no |
| 4000 | 0.40 | 1.50 | 1.38 | no |
| 4000 | 0.20 | **1.86** | **2.87** | no |

**Superposition widens the CORE, where the real atmosphere widens the TAILS.** The real
signature is a 1–99 span of ×2.2 with an IQR of only ×1.11. At H=4000, α=0.20 the span
reaches ×1.86 — in the right range — but the IQR is ×2.87 against the real ×1.11. At
H=2000, α=0.20 the IQR grows by **×9.19**. The mechanism reproduces the wrong phenomenon.

## Two independent reasons this is refuted, not merely unconfirmed

1. **Wrong dispersion signature** — core widening instead of tail growth (above).
2. **Wrong cloud fraction.** The effect only appears at α = 0.20, i.e. **80 % clear sky.**
   The real high regime carries a **2.6–3.6 %** sky-like population. At the mixing fractions
   the real data supports (α ≥ 0.8) the spans are ×1.10 and ×1.29 — nowhere near the real
   ×2.2.

## What it does get right

At H=2000 mixing does pull `lowest_mode` down: 1984 → 1321 m, a **−663 m** pull against a
real-world −297 m. So photometric mixing *can* move the estimate in the observed direction —
it simply does not do so with the observed distributional signature, and only at a cloud
fraction the real data excludes.

## Standing

| injection | status | reason |
|---|---|---|
| 1. texture scale | cleared | metric-space texture, 4× apparent range |
| 2. 3-D relief | cleared | up to 20 % of height, ρ = +1.000 |
| 3. broken field / sky mode | refuted | 2.6–3.6 % against a 10 % prominence floor |
| **4/5. photometric superposition** | **refuted** | widens the core (IQR ×9.19) where reality widens the tails (×1.11); needs 80 % sky where reality shows 2.6–3.6 % |
| 6. atmospheric haze | **not run — now low value** | it is a contrast-reduction variant of the same photometric mechanism just refuted |

## Where this leaves W8

**The dispersion signature is the discriminating measurement, and it has now refuted the
photometric route.** The remaining fact is that the real high regime's disparity field is
2.2× wider in the tails with an essentially unchanged core, and shifted +1.5 px centrally.
No injection tested reproduces that combination — relief leaves it untouched, and
superposition widens the wrong part of the distribution.
