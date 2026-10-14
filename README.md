# clouds — cloud base height from stereo fisheye imagery

Ground-based stereo vision estimates the **geometric** cloud base height (CBH)
from two fisheye cameras on a 45 m baseline. Results are validated against a
co-located **DOL-2 laser ceilometer**, which measures an **optical** cloud base.
The two quantities differ systematically; that difference is a result, not a bug.

## Layout

```
configs/            versioned configuration + calibration registry
  default.yaml        every tunable, documented
  paper_dlcp2026.yaml the parameters the written work states
  calib/*.json        each affine matrix with its provenance
src/clouds/         the importable package
  config/             schema, loader and calibration registry
  provenance.py       run manifests and JSONL run logs
  geometry/           camera model, height<->disparity, image preparation
  matching/           matcher backends; correspondence extraction and filtering
  clustering/         disparity-space clustering; image-space spatial filter
  temporal/           measurement, track, merge, Kalman filter, association,
                      reliability index, reporting
  sources/            readers: image pairing, DOL-2 ceilometer, algorithm logs
  metrics/            time base, windowed alignment, series statistics
  pipeline/           per-pair analysis, the honest evaluation, task entry points
  viz/                plotting (presentation only)
tools/              command-line entry points
experiments/        claim scoring and one-off analysis
docs/               plan, runbooks, data notes, reproduction records, deprecation map
papers/             authored work: coursework, DLCP submission, LaTeX sources
references/         third-party instrument manuals and certificates
```

Measurement data does **not** live in this repository. It lives under
`~/clouds_data/` (images, DOL-2 ceilometer logs, auxiliary meteorology, archives,
generated logs and report assets). Code and research items live here.

Start with [`docs/plan/`](docs/plan/) for what the project is building and the
rules it works under, then [`docs/`](docs/): [runbooks](docs/runbooks/) for how to run it,
[data notes](docs/data/) for what the instruments actually produce,
[reproduction records](docs/repro/) for which published numbers hold, and the
[deprecation map](docs/deprecation-map.md) for which modules to trust.

## Setup

```bash
conda env create -f environment.yml        # or use the existing `clouds` env
conda activate clouds
pip install -e .
```

## Running

```bash
# Show the fully resolved configuration and its provenance, without running anything
python tools/run_info.py --config configs/paper_dlcp2026.yaml

# Honest evaluation of one session against the ceilometer
# (writes a run manifest, run log, series and metrics)
python tools/evaluate.py --config configs/paper_dlcp2026.yaml

# Same thing through the task dispatcher
python main.py --config configs/paper_dlcp2026.yaml

# Score the result against the published claims
python experiments/reproduce_paper.py --config configs/paper_dlcp2026.yaml

# Compare two calibrations on the same data
python tools/compare_calibrations.py --config configs/paper_dlcp2026.yaml
```

Always smoke-test with `--limit 30` before committing to a full run, and read
[`docs/runbooks/evaluate.md`](docs/runbooks/evaluate.md) first — it lists the
pitfalls that have already cost time here (a 30-frame correlation is noise, and
`--stride` invalidates correlation numbers entirely).

## Provenance contract

Every run writes, into its own output directory:

- `run_manifest.json` — git commit and dirty state, interpreter and package
  versions, the **resolved config plus its sha256**, the **calibration identity
  including when it was calibrated and for which sessions**, the **dataset
  identity including the ceilometer file's sha256**, argv, counts and timings.
- `run.log` — structured JSON-lines events, flushed per event so a crashed run
  still leaves its history.

A result that cannot be traced to a config hash, a calibration id and a dataset
hash is not a result.

## Design notes

Three choices explain most of the structure, and each was made because the
opposite caused a concrete problem here:

- **Configuration is versioned data, not module globals.** It used to be a
  repository-root `config.py` whose constants were mutated at run time, so an
  output could not be attributed to the parameters that produced it. Values now
  live in `configs/*.yaml`, are validated on load and are hashed into the run
  manifest; per-run deviations are recorded as explicit overrides.
- **Every run writes a manifest.** Figures were previously produced by drivers
  that left no record of their inputs, so reconstructing them meant guessing at
  window lengths, selection rules and parameters. Numbers now travel with the
  code revision, config hash, calibration identity and dataset hash that made
  them.
- **Superseded code is deleted, and the recovery commit is recorded.** Keeping
  dead modules "just in case" is how this tree became hard to read. Anything
  whose only evidence is "you can run the old code" is turned into a recorded
  measurement first; see [`docs/deprecation-map.md`](docs/deprecation-map.md).

## Known correctness traps

These were each verified against the data and are recorded so they are not
rediscovered:

- **Two clocks.** Image filenames are UTC; DOL-2 logs are MSK. The offset is
  exactly 3 h, verified physically from the diurnal brightness envelope.
- **Two cameras, two timestamps.** devID1 and devID2 filenames differ by ~1 s;
  pairing must be nearest-timestamp, not exact-name.
- **One directory, several days.** The image window must filter by date.
- **Ground-truth leakage.** The removed `parse_algorithm_logs` selected the
  cluster *closest to the ceilometer reading*, inflating mean windowed agreement
  from 0.24 to 0.76. No selection rule may consult the ceilometer; the finding is
  recorded in [`docs/repro/dlcp2026.md`](docs/repro/dlcp2026.md).
- **Resolution.** The written work states a 640×640 centre crop; `config.py`
  shipped 320×320.
