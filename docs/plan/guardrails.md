# Guardrails

Rules this work follows. They exist because the previous iteration of this
project produced correlation figures that a later audit showed were inflated by
letting the ground truth select the measurement (mean windowed r 0.24 → 0.76 on
the same data). Nothing here is enforced by a tool; they are stated so a reader
can check whether the work honoured them, and so a violation is visible rather
than arguable.

## G1 — Ground truth never participates in producing an estimate

The ceilometer may be used **only after** a full estimate exists, to compare
against it. It must not influence any of: cluster selection, layer choice,
calibration, matcher parameters, filtering, or which frame is reported.

- Every function that produces an estimate is written so it cannot see the
  ceilometer; if it needs it, that is a design defect.
- A run that does not need the ceilometer at all must still work. If it does not,
  the estimate is contaminated.

## G2 — Pre-registration

Before a comparison is run, state in writing: the metric, the period, the
config, and what result would count as failure. Add it to
[`experiments/`](../../experiments/) or the relevant document **first**.

A metric chosen after seeing the result is not evidence.

## G3 — No tuning on the data used to report

- Tune on one period, report on another. The validation set is touched once.
- Never tune a parameter per frame or per cluster on the evaluation period.
- If a parameter is selected by looking at agreement with the ceilometer, that is
  a fit; it must be declared, and the resulting number must be reported as a
  fitted result, not a validated one.

## G4 — Baselines first, and they must be beaten

Every added component is compared against the simplest thing it replaces:
the paraxial formula against exact triangulation; `largest`/`lowest` selection
against any learned rule; no filter against the Kalman filter. A component that
does not beat its baseline on a pre-registered metric on held-out data is removed,
not kept.

## G5 — Physical plausibility is a requirement, not a tie-breaker

An estimate must be a possible cloud base:

- height in a stated physical range, and no negative or zero disparity;
- the estimated motion of a tracked layer must not exceed physically possible
  vertical wind (state the bound);
- a layer must persist for a minimum number of frames to be reported;
- the estimate must come from a stated number of independent matches.

Rejections are counted and reported, not silently dropped.

## G6 — Report uncertainty, and make it honest

- Every estimate carries an uncertainty derived from the error sources that
  actually dominate (calibration, disparity localisation, geometry) — not from
  the within-cluster spread of a handful of correlated matches, which understates
  the error by an order of magnitude (measured: median within-cluster σ is 0.6 %
  of the disparity, while a 1 px calibration error is 5.5 % at 1.5 km).
- Calibration error is reported as part of the budget, not folded into a fudge
  factor.
- If uncertainty cannot be justified, say so rather than printing a number.

## G7 — Falsification is designed in, not added later

For each claim, state the observation that would refute it. Examples that are
required to exist for this project:

- *"The rectification is correct."* → the Sun's images must coincide; measured
  residual must be reported.
- *"Heights are physically real."* → independent image regions of the same cloud
  layer must agree within their stated uncertainty.
- *"The temporal filter improves the signal."* → it must improve agreement on
  held-out data, not only smoothness.
- *"The estimator tracks dynamics."* → correlation of *changes*, and a statement
  of which divergences are expected.

A claim with no possible refutation is not a result.

## G8 — Dynamics and level are separate claims

Agreement in **dynamics** (does the estimate move when the cloud moves) and
agreement in **level** (does it give the same height) are different claims and
are reported separately. The ceilometer measures an *optical* base and the stereo
a *geometric* boundary, so a level offset is expected and must be characterised
rather than minimised. A method that improves level by distorting dynamics is a
failure, and vice versa.

## G9 — Cross-validation of instruments is not validation of the algorithm

Agreement with the ceilometer is the only external check, and it is imperfect by
physical construction. Every result must also be supported by at least one
**internal** consistency check that does not involve the ceilometer (G7).

## G10 — Negative and null results are reported

A failed test is a result. Record what was tried, what happened, and what it
rules out — including in the audit trail. Deleting a failed experiment is how the
same mistake gets made twice.

## G11 — Nothing is claimed without a run manifest

Every number in a report is traceable to a run directory containing the code
revision, config hash, calibration identity, dataset hash and results. A number
without a manifest is a recollection.

## G12 — Simplest explanation first

When a result looks good, first ask whether it is an artefact of selection,
leakage, an off-by-one in time alignment, or a systematic scale error, before
attributing it to the method. The two largest errors found in this project so
far were a selection leak and an assumed focal constant — not modelling failures.

## G13 — The method must run at runtime with what runtime has

Runtime has **calibration and photographs only**. Any step that needs the
ceilometer, historical data, or a fitted parameter is a research tool, not part
of the deliverable, and must be labelled as such. Every deliverable step must be
runnable end-to-end on a single unseen session.
