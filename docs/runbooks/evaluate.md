# Runbook — evaluating cloud base height

Runs the honest path: stereo pairs → LoFTR → HDBSCAN → cluster selection →
spatial filter → Kalman filter → comparison against the DOL-2 ceilometer. No
cluster is ever chosen by looking at the ceilometer. Doing so inflates the
agreement substantially; the measured effect is in
[`repro/dlcp2026.md`](../repro/dlcp2026.md).

## Before you start

```bash
conda activate clouds
```

Check what a run would do — configuration, calibration identity, dataset
identity — without loading a model or touching an image:

```bash
python tools/run_info.py --config configs/paper_dlcp2026.yaml
```

Read the output. Two fields decide whether a result means anything:

- `calibration` — which matrix, **when it was calibrated**, and whether that date
  is verified. If it says `verified=False`, the calibration's own date is
  inferred, and applying it to another session is an unquantified error source.
- `applies to this session` — whether the registry claims that calibration covers
  the session you are about to process.

## Run

```bash
# Full evaluation
python tools/evaluate.py --config configs/paper_dlcp2026.yaml

# Cheap smoke test first (a few frames; enough to prove the path works,
# far too few for any correlation to mean anything)
python tools/evaluate.py --config configs/paper_dlcp2026.yaml --limit 30
```

Useful options:

| Option | Effect |
|---|---|
| `--stride N` | every Nth pair; spans the whole window, so it screens parameters cheaply |
| `--strategy {lowest,lowest_reliable,largest}` | cluster selection rule |
| `--no-spatial` | disable the spatial coherence filter (ablation) |
| `--shift-scan` | also locate the best global time offset between the instruments |
| `--dump-clusters` | write every per-frame cluster to `clusters.jsonl` for offline analysis |
| `--out DIR` | write to a chosen directory instead of a timestamped run directory |

The dispatcher in `main.py` reaches the same code:

```bash
python main.py --config configs/paper_dlcp2026.yaml            # task from the config
python main.py --config configs/paper_dlcp2026.yaml --limit 30
```

## Reading the result

Everything lands in one run directory:

```
run_manifest.json   code revision, config hash, calibration identity, dataset hash, counts, results
run.log             JSON-lines events
series.json         raw and filtered series, and the ceilometer series, in the ceilometer clock
metrics.json        every metric
clusters.jsonl      only with --dump-clusters
```

## Pitfalls that have already cost time

- **A high correlation on few frames is noise.** The per-window shift search will
  happily find a shift that fits a handful of points. Check `n_scored` in
  `metrics.json`; the shift scan exists precisely to expose this.
- **`--stride` invalidates correlation numbers.** It is a screening tool. On this
  dataset a stride-4 run reported mean r ≈ 0.60 where the full run gave 0.20.
- **The ceilometer reports an optical cloud base; the stereo measures the
  geometric one.** A constant vertical offset is expected and is not an error.
- **`lowest` is not the pure minimum.** It prefers clusters at or above
  `min_cluster_size_for_tracking` and falls back to the full set, preserving the
  historical behaviour. `lowest_reliable` is the strict form.
