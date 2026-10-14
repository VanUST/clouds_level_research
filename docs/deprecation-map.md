# Deprecation map

Every module in the tree, its status, and the one-line reason. Statuses:

- **active** — on the validated path; a change to it can change a reported result.
- **active (presentation)** — reachable, but cannot affect a number.
- **active (mode-gated)** — reachable only under a non-default configuration mode.
- **deprecated** — superseded; kept as evidence or for reference, imported by nothing.
- **retired** — replaced by an explicit failure that names its successor.

## Configuration and provenance

| Module | Status | Notes |
|---|---|---|
| `config/schema.py` | active | The frozen dataclass tree and `ConfigError`. |
| `config/load.py` | active | YAML loading, variable substitution, whole-schema validation, summary rendering. |
| `config/calibration.py` | active | The calibration registry: matrix plus when it was calibrated, how, and for which sessions. |
| `provenance.py` | active | Run manifests and the JSONL run log. |

## Domain

| Module | Status | Notes |
|---|---|---|
| `geometry/camera.py` | active | `CameraModel`, `StereoCameraSystem`, height↔disparity. |
| `geometry/imaging.py` | active | Centre crop used to prepare a pair. |
| `matching/matchers.py` | active | Kornia LoFTR and MatchAnything EloFTR backends; backend registry. |
| `matching/filters.py` | active | Correspondence extraction and the match filters (epipolar, confidence, MAD, cycle-consistency). |
| `clustering/disparity.py` | active | HDBSCAN over disparity vectors; separates cloud layers and rejects outliers. |
| `clustering/spatial.py` | active | Three-stage spatial coherence filter. |
| `temporal/measurement.py` | active | `ClusterMeasurement` — the tracker's input record. |
| `temporal/track.py` | active | `TrackStatus`, `CloudTrack` — Kalman state, lifecycle, quality. |
| `temporal/merge.py` | active | Height-based merging of fragmented clusters. |
| `temporal/kalman.py` | active | `TemporalFilter` — lifecycle, association, predict/update, outlier rejection, CBH selection. |
| `temporal/association.py` | active (mode-gated) | Hungarian association; used only when `temporal.mode: multi_track`. The operational mode is `simple`. |
| `temporal/reliability.py` | active | Unsupervised Reliability Index and its trust label. A reporting signal only — never used to select a cluster. |
| `temporal/report.py` | active (presentation) | CBH series, track summaries, formatted report. |
| `temporal/settings.py` | active | Config → tracker settings, reporting deviations from library defaults. |

## Application

| Module | Status | Notes |
|---|---|---|
| `sources/images.py` | active | Image pairing: directory sweep, and calendar-day nearest-timestamp window. |
| `sources/ceilometer.py` | active | DOL-2 reader; handles both on-disk dialects. |
| `sources/algorithm_log.py` | active | Writes the per-pair `.txt` summary and `.npz` match archive. |
| `metrics/alignment.py` | active | Clock conversion, nearest-time pairing, windowed correlation, wide shift scan. |
| `metrics/series.py` | active | Frame-to-frame jump, lag-1 autocorrelation, improvement factors. |
| `pipeline/analyze.py` | active | `StereoAnalyzer` — one pair in, candidate cloud layers out. |
| `pipeline/evaluate.py` | active | The honest end-to-end path and all comparison metrics. |
| `pipeline/tasks.py` | active | Task-level entry points: directory sweep, windowed time series, trail, calibration ablation. |
| `viz/plots.py` | active (presentation) | Plotting only; never on the numeric path. |

## Entry points

| File | Status | Notes |
|---|---|---|
| `main.py` | active | Dispatcher; the task is a config value (`processing.task`). |
| `tools/run_info.py` | active | Reports config, calibration and dataset identity without running anything. |
| `tools/evaluate.py` | active | Honest evaluation; writes the manifest. |
| `tools/compare_calibrations.py` | active | Calibration A/B through the honest path. |
| `experiments/reproduce_paper.py` | active | Scores the evaluation against the DLCP2026 claims. |

## Deleted

Removed from the tree. Both are recoverable from commit `96c5f75`:

```bash
git show 96c5f75:tools/unified_time_shift.py   # the old analysis pipeline
git show 96c5f75:config.py                     # the old mutable root config
```

| Path | What it was | Why it went |
|---|---|---|
| `tools/unified_time_shift.py` | The earlier end-to-end comparison pipeline, including `parse_algorithm_logs`. | Its DOL-2 reader is superseded by `sources/ceilometer.py` and its metrics by `metrics/`. Its cluster selection consulted the ceilometer, which inflates agreement (mean windowed r 0.24 → 0.76); the finding is measured and recorded in [`repro/dlcp2026.md`](repro/dlcp2026.md), so the code is no longer needed as evidence. |
| `config.py` (repository root) | Configuration as mutable module-level globals. | Replaced by the versioned YAML configs and `clouds.config`. |
| `pipeline/runner.py` | The god class mixing pairing, stereo analysis, four tasks and log writing. | Replaced by `sources/images.py`, `sources/algorithm_log.py`, `pipeline/analyze.py` and `pipeline/tasks.py`. |

## Known gaps and defects

These are open, recorded so they are not rediscovered:

- **No calibration for the 2025-05-23 session.** The 2025-05-22 matrix is applied
  to it; drift of up to 30 px (≈900 m) is documented but unquantified.
- **`run_iterative_refinement` is not implemented** — it prints a message. It is
  exposed by the dispatcher and will silently do nothing.
- **The published correlation figures are not reproducible** by any
  ceilometer-free selection rule on this data. Measured effect and method:
  [`repro/dlcp2026.md`](repro/dlcp2026.md).
- **The DOL-2 range stated in the written work (10–3000 m) is contradicted by the
  data**, which reaches 7010 m.
- **`MatchAnything` backend is unverified.** `matching/matchers.py` implements it
  and `matching.backend` can select it, but the third-party tree and weights are
  absent and no result in this project was produced with it.
- **Eleven driver scripts that produced the published figures were deleted** in
  commit `96c5f75` and are deliberately **not** restored. They remain readable in
  git history if a specific published number must be re-derived. Among them:
  `evaluate_temporal.py`, `diagnose_windows.py`, `plot_correlation_profile.py`,
  `manual_calibrator.py`, `bruteforce_homography.py`, `track_clouds.py`,
  `compare_backends.py`, `visualize_eloftr.py`, `trail_matcher.py`,
  `setup_matchanything.py`, `raw_converter.py`.
- **`tools/__init__.py`** exists only so `main.py` can import `tools.evaluate`.
