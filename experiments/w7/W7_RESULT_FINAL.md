# W7 — the validated synthetic control: the matcher is sound, the SHIPPED SELECTION RULE is not

Supersedes both the withdrawn first attempt (`W7_RETRACTION.md`) and its claimed pass.

## The renderer, and its gate

`experiments/w7/w7_renderer.py`. The rig was derived from the calibration rather than
assumed, and it was the derivation that took the work:

- the pipeline warps **camera 1** by the affine, so the pair must be built so the warp
  aligns rather than misaligns them;
- the baseline lies along world **−Y**, camera 2 at (0, −45, 0), so the parallax lands in
  `u` and is **positive in the pipeline's own convention** (`kp1 − kp2`), as the real data
  is (du ≈ 13 px against dv ≈ 2 px);
- camera 2's sensor is rotated **−90°** about the optical axis;
- **camera 2's principal point is (1014.10, 965.03)**, not the image centre.

That last term is the one whose absence produced a non-physical parallax. The affine maps
camera 1's principal point onto camera 2's, so `c2 = R·c1 + t`; with both at the image
centre the rendered parallax came out **negative and growing with height**.

**The gate** — `du(zenith)` must equal `f·atan(B/H)` and fall as 1/H — **passes exactly**:

| H | du(zenith) | f·atan(B/H) |
|---|---|---|
| 1000 m | 29.428 px | 29.410 px |
| 2000 m | 14.715 px | 14.713 px |
| 4000 m | 7.357 px | 7.357 px |

## 1. The matcher recovers the known height

| H_true | ground truth | RoMa | bias | recovered IQR |
|---|---|---|---|---|
| 1000 m | 29.56 px | 28.72 px | −0.914 px | 2.88 px |
| 1500 m | 19.73 px | 19.58 px | −0.239 px | 1.41 px |
| 2000 m | 14.80 px | 14.85 px | +0.060 px | 0.73 px |
| 3000 m | 9.87 px | 9.80 px | −0.071 px | 0.53 px |
| 4000 m | 7.40 px | 7.48 px | +0.072 px | 0.43 px |

**ρ(ground truth, RoMa) = +1.000, median |bias| = 0.072 px.**

**The matcher and the triangulation recover a known geometric height essentially exactly**
through the real fisheye geometry and the real distortion model. This is now established on
a renderer that has passed its own gate, which is the difference between this result and the
withdrawn one.

## 2. The shipped selection rule fails badly; the new one does not

| H_true | truth | `median` | **`lowest` (shipped)** | **`lowest_mode` (new default)** |
|---|---|---|---|---|
| 1000 m | 1000 | 1024 | **675** | 1025 |
| 1500 m | 1500 | 1507 | **722** | 1501 |
| 2000 m | 2000 | 1988 | **529** | 1984 |
| 3000 m | 3000 | 3007 | 2423 | 3003 |
| 4000 m | 4000 | 3944 | 3598 | 3936 |

| selector | ρ(true, reported) | p | error range |
|---|---|---|---|
| `median` | +1.000 | 0.000 | −56 to +24 m |
| **`lowest` (shipped)** | **+0.700** | **0.188 (not significant)** | **−1471 to −325 m** |
| **`lowest_mode` (new default)** | **+1.000** | **0.000** | **−64 to +25 m** |

**The shipped `lowest` rule reports 675 m for a cloud at 1000 m, 722 m for 1500 m and**
**529 m for 2000 m** — errors of 325 to 1471 m on clean synthetic data with a known answer,
and a rank correlation that is not even statistically significant.

## What this settles

1. **The direct answer to the Gate 0 question: yes, the selection rule is a major defect**,
   and it is now demonstrated on ground truth rather than inferred.
2. **The matcher, the fisheye geometry and the distortion model are cleared** — on a
   validated renderer this time.
3. **`lowest_mode`, which is already the project default, removes the defect** — errors
   within 64 m where the shipped rule errs by up to 1471 m.

## What it does not settle

The synthetic fields are flat planes with procedural texture and no atmosphere. The
real-world inversion therefore has a cause outside both the matcher and the selection rule —
consistent with the earlier finding that replacing `lowest` with `lowest_mode` on real data
moved the regime shift only from ≈0 to −30 m (p = 0.058), i.e. it did not remove it.

**So the instrument's front end is sound and its back end is now fixed, and the real-world
inversion survives both.** That is a sharper position than the project has had at any point:
two stages are cleared on ground truth, and the remaining failure is demonstrably not in
either of them.
