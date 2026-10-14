# Plan — a physically-validated CBH estimator for real inference

**Status:** proposed, awaiting sign-off on the decisions listed at the end.
**Readable report:** [`../report/findings.html`](../report/findings.html) — the
same material as a self-contained document with figures.

**Companion documents:** [stage-audit.md](stage-audit.md) (what is wrong now),
[guardrails.md](guardrails.md) (how we are allowed to work),
[w1-calibration.md](w1-calibration.md) (current W1 results),
[literature.md](literature.md) (prior art and reusable tools),
[performance.md](performance.md) (where inference time goes),
[pipeline-steps.md](pipeline-steps.md) (what was removed from the chain and why),
[evaluation.md](evaluation.md) (how a result is allowed to be claimed),
[temporal-alignment.md](temporal-alignment.md) (why the old correlation metric was withdrawn),
[research-workplan.md](research-workplan.md) (the prioritised research plan),
[matchers.md](matchers.md) (which matcher the evidence selects).

---

## 1. Objective

Produce a cloud-base-height estimator that works at runtime with **calibration
and photographs only**, is physically justified at every step, and whose
behaviour is falsifiable against the historical data. Accuracy is the priority;
runtime cost is not a constraint.

Concretely: the estimator must track the *dynamics* of the ceilometer, and where
it does not, the divergence must be explained by a physical mechanism (different
layer sampled, optical-vs-geometric difference, wind advection) rather than
absorbed into a fudge factor.

## 2. Runtime contract

Available at runtime:

- two synchronised fisheye photographs per acquisition;
- the camera calibration (intrinsics, projection model, relative orientation);
- a timestamp per photograph;
- camera location and pose (assumed known and stable).

**Not** available: ceilometer, meteorology, any historical statistic, any
parameter fitted against ground truth.

**Additional constraint (2025-10-13):** the calibration results that exist are
**fixed and cannot be changed** — no calibration target, no re-calibration, no
new measurements. The calibration is therefore an *input with an unknown error*,
not something this project may correct. What this project must do about it is
measure its error and state the margin within which the product is still usable;
if the fixed calibration lies outside that margin, the honest deliverable is
dynamics with an explicitly uncalibrated scale.

Note the one distinction that matters here: the projection model constant `f`
(and the lens's projection law) is part of the **geometry model**, not of the
calibration matrix. The fixed affine maps one image into the other's frame; it
does not fix how angles map to pixels. `f` may and must be determined from the
data, because an error in it scales every height.

Therefore every deliverable component must be a function of
`(image pair, calibration, time, location)` alone. Anything that needs a
ceilometer is a research instrument and will be labelled as such.

## 3. What "works" means

Three separate, separately-reported claims. Nothing is called a success because
another claim passed.

**C1 — Physical validity.** Every reported estimate is a possible cloud base:
positive disparity, height in a stated range, derived from a stated number of
independent matches, using the exact projection model. Rejection reasons are
counted.

**C2 — Internal consistency.** Independent evidence that does not use the
ceilometer:
- **Infinity reference:** the Sun (or another object at infinity) must rectify to
  zero disparity within a stated tolerance. This is the only absolute geometric
  check available and it validates the calibration, not the cloud estimate.
- **Cross-region agreement:** disjoint image regions of the same layer must give
  heights agreeing within their stated uncertainty.
- **Reprojection residual:** the observed 2-D shift must match the shift the
  exact model predicts at the estimated height, with the residual reported.
- **Temporal coherence:** the estimated layer motion must stay within a stated
  physical bound.

**C3 — Agreement with the ceilometer.** Reported in two parts and never merged:
- **Dynamics:** correlation of change, in windows, with a stated expected level;
- **Level:** the offset and its stability over time, *characterised* rather than
  minimised. The written work's mechanism (ceilometer detects a backscatter
  threshold inside the cloud, stereo the geometric boundary) predicts a
  non-negative, cloud-type-dependent offset. That prediction is testable: the
  offset should be systematically positive and should co-vary with cloud type.

A method that improves C3-level by distorting C3-dynamics, or that improves
either by weakening C1/C2, is rejected.

## 4. Strategy

Rebuild the chain in the order in which errors dominate, validating each stage
internally before using any ground truth. The audit's ranked causes
([§8](stage-audit.md#8-ranked-candidate-causes-of-the-absolute-disagreement-with-the-ceilometer))
set the order: the focal constant and calibration first, geometry second,
matching third, layer extraction fourth, temporal last.

A component is only accepted if it beats the simplest baseline it replaces on a
pre-registered metric on held-out data (guardrail G4).

## 5. Work packages

### W0 — Measurement harness and frozen baseline

*Purpose:* make every later claim comparable and prevent silent drift.

- Freeze the current pipeline's outputs on the historical data as the **baseline**
  (already reproducible: jump 78.48/28.54 m, autocorr 0.6945/0.9779, windowed
  r 0.2015/0.3454, rejection 0.1545 — the manifest exists).
- Define dataset splits **by time**, not by frame: a tuning period, a validation
  period, and (if enough data) a final test period. Write them down.
  **Revised 2026-10-13 (W1.14):** splits must be temporal *within 2025-05-23*,
  because 2025-05-22 is cloud-free and yields no stereo signal. There is one usable
  session, so session-level hold-out is impossible and any held-out block is
  correlated with the tuning block. This is a limitation to state, not to hide.
- Implement the metric set once, with the pre-registration recorded
  (guardrail G2): dynamics correlation, level bias and its stability, internal
  consistency residuals, rejection counts, coverage.
- Extend the existing `provenance` manifest to record every W-package parameter.

*Falsification:* a re-run of the frozen configuration must reproduce the baseline
numbers exactly; if it does not, the harness is the suspect, not the method.

### W1 — Calibration verification and the workable margin *(highest priority)*

*Purpose:* the calibration is **fixed** (see §2), so this package does not correct
it. It answers three questions: how big is its error, how does that error
propagate into height, and what margin of error can the product survive.

- **Determine the projection model and `f` from the data.** `f` is a geometry
  constant, not a fixed calibration result. The audit finds the image-circle
  radius at ≈1080–1156 px against the 960 px that `f = ξ₀/π` assumes, implying
  `f` may be 10–20 % too small, which scales every height (audit §1.3). Decide
  between the candidate projection laws (Kannala–Brandt equidistant, equisolid,
  stereographic) using the image circle, the horizon and the Sun, and report `f`
  with an uncertainty. **This is the single largest identified error source.**
- **Verify the fixed calibration with an infinity reference.** The Sun is the only
  object at infinity in these images. Rather than hand-rolling the detector (the
  audit's threshold attempt measured saturated cloud — a negative result), adopt
  an existing tool: **SuMo** (DLR) calibrates intrinsics and orientation from
  ordinary day/night Sun and Moon images and explicitly supports **continuous
  drift monitoring**, reporting RMSE ≤ 0.14° and 0.14–0.38° across camera types;
  **ORION** is the open-source star-based method it cross-checks against. Apply
  one of them to measure the *residual* of the fixed affine — how far the Sun's
  images are from coinciding — across both sessions.
- **Check the horizon** as a second angle reference: in a level zenith-pointing
  fisheye the horizon is a circle at the 90° radius, giving `f` and camera tilt
  independently of the Sun.
- **Drift between sessions — CLOSED as not measurable.** 2025-05-22 is cloud-free
  and produces no correspondences, so there is only one usable session (W1.14). The
  answer is "not measurable from this data", with the reason demonstrated.
- **State the error budget and the margin.** Convert calibration error into
  height error over a range of heights and cloud types (audit §1.1 gives the
  conversion), and answer the question that motivates this work: *for a stated
  height range, what calibration error is tolerable, and is the fixed
  calibration inside it?*

*Deliverables:* a measured `f` with uncertainty and the chosen projection law; a
repeatable infinity-reference residual for the fixed calibration; a measured
session-to-session drift; a calibration-error → height-error propagation; and an
explicit statement of the workable margin.

*Falsification:* if the fixed calibration cannot be shown to place the Sun's
images within a stated tolerance, or `f` cannot be pinned to better than a few
percent, then absolute heights from this rig are not defensible and the product
reports dynamics with an uncalibrated scale. Stated in advance: the expected
tolerance for a useful product is **≲0.5 px** equivalent disparity, i.e. ~3 %
height error at 1.5 km (audit §1.1). This is a go/no-go on *absolute height*,
not on the project: dynamics remain the necessary condition either way.

### W2 — Exact geometry and height estimation

- Replace the paraxial formula with **exact triangulation on the camera sphere**:
  each match yields a height from the intersection of two rays, using the
  measured projection model from W1.
- Use the **full 2-D shift**, not the x-component (audit §2.2).
- Propagate calibration, localisation and model uncertainty into a **per-match
  height uncertainty**, and report the height distribution per frame rather than
  a single number.
- Add the **reprojection residual** as a first-class output (C2).

*Falsification:* exact triangulation must reproduce the paraxial result in the
image centre (a correctness check, not an improvement claim) and must remove the
position-dependent bias at the field edge (audit §2.1).

### W3 — Matching quality

> **RESOLVED 2026-10-13:** see [matchers.md](matchers.md). **RoMa is the default.**
> MASt3R is disqualified (zero differential response); EfficientLoFTR has a 30 % gain
> error; LoFTR has a 4.3 % gain error. RoMa's earlier "0 % interior coverage" was a
> confidence threshold, and the discarded interior predictions were then measured
> accurate (0.68 px against the reliable field). All backends remain selectable —
> `tools/matchers.py list` / `check`.

- Measure LoFTR's **sub-pixel localisation accuracy and its distribution** on
  these images — not assume it. Establish the actual per-match disparity error,
  which is the dominant term in the budget.
- Evaluate alternatives (EfficientLoFTR, RoMa, DKM, OmniGlue, and classical
  methods) against a pre-registered criterion, not against a metric tuned here.
- Apply the **epipolar/rectification constraint** and left-right consistency, and
  measure how much they remove.
- **Mask the near-field occluders** (trees, building, mast, wire — audit §5) using
  a circular field mask plus an explicit occluder mask, so that non-sky objects
  cannot be reported as very low cloud.
- Produce match density and uncertainty maps as diagnostics.

*Falsification:* the measured localisation error must be small enough to meet the
W1 error budget; if it is not, no downstream component can fix it, and the
limitation must be stated.

### W4 — Layer extraction

> **Blocked on W1 until 2026-10-13's finding:** the calibration carries a ~1.6°
> residual roll that inflates the within-frame disparity spread ~4× (IQR 8.14 px →
> 2.11 px once corrected). Clustering the raw disparity field is therefore
> clustering calibration error. See
> [w1-calibration.md](w1-calibration.md#w112--the-calibration-residual-is-measured-a-16-roll).
> W4 must consume the roll-corrected disparity.

- Replace HDBSCAN-with-a-3-px-threshold plus a 300 m merge hack (audit §4) with a
  method that is **stable under its own parameters** and whose output is a
  physically-meaningful set of layers.
- Candidate direction: model the per-match height distribution as a
  small number of modes with uncertainty, rather than clustering raw disparities;
  select the number of layers by a stated criterion (information criterion or
  bootstrap stability), not by a magic minimum size.
- Report layer persistence and support, so a layer is reported only if it is
  established by enough independent matches over enough frames.

*Falsification:* the layer count and heights must not change materially when the
free parameters are varied within a stated range; if they do, the method is not
reporting structure.

### W5 — Temporal processing

- Track **layers**, not "the lowest cluster": association by height, image
  position and size, with a physical motion bound.
- Report **all** layers with uncertainty, and a designated cloud base with an
  explicit rule (the meteorological definition is the lowest layer, but the
  estimator should not hide the others).
- Model wind advection explicitly, since it is the reason the two instruments
  decouple in time (and it is also the reason the comparison needs a per-window
  shift).
- Keep the filter's job to smoothing and outlier rejection; it must not be
  performing selection (audit §6).

*Falsification:* the filter must improve held-out agreement, not merely
smoothness; a smoother series that agrees no better is not an improvement.

### W6 — Runtime inference and self-reporting

- Assemble the runtime pipeline: `(pair, calibration) → layers with uncertainty`.
- Emit a **per-frame self-assessment**: number of supporting matches, internal
  consistency residuals, whether the infinity check is available, and a
  confidence derived from those — never from a ground-truth-fitted model.
- Define failure modes explicitly (featureless overcast, sun not visible, layer
  unestablished) and what the system reports in each case.

*Falsification:* the whole pipeline must run on a session it has never seen,
using no information unavailable at runtime (guardrail G13).

## 6. Milestones and decision points

| # | Milestone | Decision it gates |
|---|---|---|
| M0 | Baseline frozen, splits written, metrics pre-registered | work may begin |
| M1 | `f` measured; infinity residual of the *fixed* calibration measured; margin stated | **go/no-go for absolute height** |
| M2 | Exact geometry validated; bias removed | geometry frozen |
| M3 | Matcher localisation characterised; occluders masked | error budget updated |
| M4 | Layer extraction stable under parameters | layers trustworthy |
| M5 | Temporal tracking on layers | dynamics claim testable |
| M6 | End-to-end runtime run with self-assessment | deliverable |

M1 is a genuine gate: if the focal constant and orientation cannot be measured
well enough to place absolute heights inside a useful margin, the honest outcome
is to report dynamics with an uncalibrated scale and say so.

## 7. Open questions for the record

- Which camera and lens exactly (`§1.3`)? The datasheet Fixes the nominal FOV and
  projection model, which bounds the f question immediately.
- Is a physical calibration target or a surveyed reference point ever available?
  That would settle absolute scale far better than the Sun alone.
- Are the two sessions the only data, or is more available (audit and W1 would
  both be strengthened by more sessions, especially with calibration events)?
- What cloud-base range must the product cover? The achievable accuracy collapses
  with height (§1.1), so the target range decides the required calibration
  precision.
- Is the camera rig still on the same mount as the 2025 acquisition, and has it
  been re-calibrated since? This determines whether the historical calibration is
  a starting point or a historical artefact.

## 8. Prior art and what is reused

Full reconnaissance with URLs: [literature.md](literature.md). The load-bearing
findings:

- **A directly comparable study exists.** Crispel & Roberts (AMT 2018) do
  180° fisheye stereo over a **150 m** baseline (VIVOTEK, 2944 px), calibrate
  intrinsics with OcamCalib, derive relative orientation *without a target*
  (GPS + level, then SIFT → essential matrix, rejecting solutions inconsistent
  with known rig geometry), rectify, match densely, and validate against a
  Vaisala CL31 — cumulus at 1500 m and two layers at 1000 + 2300 m. They also
  state the same warning the audit found numerically: the pinhole disparity
  relation "becomes less relevant for larger incident angles".
- **Beekmans et al. (ACP 2016)** do hemispheric fisheye stereo on a **300 m**
  baseline, orientate from **stars**, triangulate in the angle domain, and report
  **<5 %** relative CBH error. This is the method-level template.
- **The 45 m baseline is the project's structural weakness.** Disparity scales as
  `f·B/D`: at 1.5 km this rig gives **18 px**, Crispel & Roberts' 150 m rig ≈**94 px**,
  Beekmans' 300 m rig ≈**187 px**. With 5–10× less disparity, the same angular
  error costs 5–10× more height error. No downstream method removes this; it
  bounds what is achievable and must be stated.
- **Herrero del Barrio et al. (AMT 2026)** validate against a collocated
  ceilometer over 2.5 years and reach **r² = 0.93** for median CBH within 150 m.
  Their validation recipe (per-pixel ±1 px shift → height uncertainty, plausible
  height band, layer histogram, median across pairs) is the modern practice to
  follow, and it is ground-truth-free in construction.
- **The systematic offset has a physical explanation in the literature**: an AMT
  discussion attributes it to *unmodelled 3-D cloud-side geometry* rather than to
  instrument error — consistent with the reasoning that the ceilometer detects a
  backscatter threshold inside the cloud while stereo sees the geometric lower
  boundary.
- **Sub-pixel matching is a known weak point, with a fix.** EfficientLoFTR
  (CVPR 2024) documents that LoFTR's fine refinement has *spatial variance*
  caused by soft-argmax over the whole correlation patch, and fixes it with a
  two-stage correlation. It is a drop-in replacement. For per-match uncertainty,
  SURE (ICRA 2026) is the only matcher found that emits aleatoric + epistemic
  per-match variance — code announced but not yet released, so treated as a
  candidate, not a dependency.
- **Exact triangulation is a solved problem.** Eising (2022) gives closed-form
  unit-sphere triangulation for omnidirectional cameras, valid at large zenith
  angles and yielding the two-ray residual for free. That directly replaces the
  paraxial estimator of audit §2.1.
- **Target-free calibration and drift monitoring are solved tools**, not research:
  SuMo (DLR, Sun/Moon, drift monitoring) and ORION (stars), both open source —
  but neither is a drop-in dependency here (SuMo is not on PyPI; ORION is a
  bundled app), so the infinity check is implemented in-project with their
  published accuracy as the target. See
  [availability](literature.md#availability-checked-2025-10-13).
- **No prior art found** for: pixel-level LoFTR localisation error on cloud-like
  imagery; calibration-drift budgets for cloud-camera rigs; IMM/MHT applied to
  ceilometer layer series; or an explicit wind-advection-lag correction between
  separated cameras and a zenith ceilometer. Those are open gaps this project
  would be filling rather than reusing.

## 9. Relationship to the existing code

Kept: the provenance/config layer, the source readers, the exact height↔disparity
relation (used exactly rather than paraxially), the matcher backend abstraction,
the temporal filter as a smoothing component, and the frozen baseline for
regression.

Replaced: the focal constant assumption, the paraxial estimator, the
HDBSCAN+merge layer extraction, and cluster selection as a temporal mechanism.

Deleted already (`config.py`, `unified_time_shift.py`) or to be deleted if
superseded: anything whose only justification is a number it produced under the
old, contaminated comparison.
