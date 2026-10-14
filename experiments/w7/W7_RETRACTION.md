# W7 — the synthetic control is WITHDRAWN pending a validated renderer

I am retracting the W7 result. The renderer does not reproduce the rig it claims to, and
the error is mine.

## What W7 claimed

That on a cloud layer at a known height, rendered through the measured fisheye, RoMa
recovers the truth with bias < 0.4 px and ρ = +1.000 — clearing the matcher, the geometry
and the distortion model.

## What is wrong with it

**The renderer never modelled the 90° relative sensor rotation.** The real cameras' sensor
frames differ by 89.919° (the affine's own rotation), which is why the pipeline exists in
the form it does: `prepare_pair` warps camera 1 by that affine to bring it into camera 2's
frame. My renderer produced both images in a **shared orientation**, so:

- the pair it matched was never in the pipeline's convention;
- the affine — a 90° rotation — was applied to a camera-1 image that did not need it;
- the two frames were then **misaligned by ~90°**, which is why the corrected run fails
  catastrophically (recovered disparity IQR **688 px**, ρ = −0.600).

In other words, W7 tested a rig in which both cameras point the same way. **That rig does
not exist.** The "bias < 0.4 px" result describes the matcher's behaviour on an
un-rectified, identically-oriented pair, not on the production configuration.

## How the flaw survived

The check I ran was: *triangulate the ground-truth disparity and confirm it returns the
rendered height.* That check passed — in the **native** frame. The warped-frame geometry
was never validated, and it is the warped frame the matcher actually consumes. A ground
truth is only as good as the frame it is expressed in, and I verified the wrong one.

## What happened when it was corrected

Adding the 90° relative rotation makes the pair matchable again (raw pixel correlation
0.575 at roll₂ = +88.4°), but the ground-truth parallax then **does not behave like a
physical baseline**:

| origin₂ | H=1000 | H=2000 | H=4000 | expected |
|---|---|---|---|---|
| (45, 0, 0) | 54.15 px | 54.13 px | 54.12 px | **constant** — no parallax at all |
| (0, 45, 0) | 24.63 px | 39.35 px | 46.72 px | **increasing** with height — inverted |

A correct stereo parallax must fall as 1/H. Neither does. So the renderer's geometry is
still not validated, and one more thing is wrong with it that I have not located.

## What stands and what falls

| claim | status |
|---|---|
| "RoMa recovers a known height on synthetic cloud texture" | **WITHDRAWN** |
| "the matcher architecture is cleared" | **not established** |
| "the fisheye geometry and distortion model are cleared" | **not established** — W7 did not test them in the production frame |
| the analytic round-trip check (`frame_bug_test.py`: an ideal plane at 1500 m triangulates to 1500.00 m) | **stands** — that test is analytic and uses the production triangulation directly, with no renderer |
| W7 Gate 0's observation that the shipped selection rule reports 37–278 m for clouds at 1000–2000 m | **stands as a measurement**, but is taken from the same flawed renderer and must be re-run |

**The only thing W7 established is that the shipped `lowest` selection rule is fragile on
synthetic fields** — and even that needs re-confirmation on a correct renderer.

## What this changes

The project's position reverts to: **ten refuted explanations, and no positive control.**
The matcher has *not* been cleared. The instrument *may* still be capable of the
measurement, but that is now unproven.

## The fix, and the check that must gate it

1. Render camera 1 and camera 2 in their true relative orientation (89.919° apart), so the
   affine warp aligns them rather than misaligning them.
2. Place the baseline so that the parallax appears along `u` in the warped frame, as it
   does in the real data (du ≈ 13 px, dv ≈ 2 px).
3. **Gate the renderer before using it**: the ground-truth disparity, expressed in the
   warped frame, must (a) fall as 1/H across at least three heights, and (b) triangulate
   back to the rendered height through `heights_from_matches`. Until both hold, no result
   from the renderer may be reported.

I would rather withdraw this than let a control that does not control for anything sit in
the record as the project's one positive result.
