# PURPOSE: A controlled A/B for the spatial-filter cost fix, without editing
#          production code between runs. The production change lets the tracker
#          restrict spatial validation to the labels it actually consumes; in
#          "base" mode this driver monkeypatches that restriction away, restoring
#          the original all-labels behaviour byte-for-byte in the same process
#          path. Both modes therefore see identical data, calibration and config,
#          and only the spatial-filter work differs — which is what makes the
#          wall-clock difference attributable.
# INPUTS: --config, --limit, --mode {base,opt}, --out.
# OUTPUTS: wall-clock seconds, frames/s and the run's own in-loop fps on stdout.
# KEYWORDS: a_b, throughput, spatial_filter, benchmark, evaluate
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clouds.config import load_config                       # noqa: E402
from clouds.pipeline.evaluate import evaluate, write_artefacts  # noqa: E402
from clouds.provenance import Run                           # noqa: E402
from clouds.temporal import kalman as kalman_mod            # noqa: E402

_ORIG = kalman_mod.validate_clusters_spatial


# PURPOSE: Restore the pre-fix behaviour by dropping the cluster_ids restriction.
# INPUTS: none.
# OUTPUTS: None (mutates the kalman module's bound name).
# KEYWORDS: monkeypatch, baseline, spatial_filter
def force_all_labels():
    def all_labels(kp1, labels, shifts, centers, config, cluster_ids=None):
        return _ORIG(kp1, labels, shifts, centers, config)
    kalman_mod.validate_clusters_spatial = all_labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--mode", choices=("base", "opt"), default="opt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", action="store_true",
                    help="write series.json/metrics.json/clusters.jsonl for a diff")
    args = ap.parse_args()

    if args.mode == "base":
        force_all_labels()

    cfg = load_config(args.config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(run_dir=out, entry_point="experiments/w1/opt/ab_evaluate.py",
              argv=sys.argv)
    run.open(repo_root=REPO_ROOT, config=cfg)

    t0 = time.perf_counter()
    result = evaluate(cfg, run, limit=args.limit, dump_clusters=args.dump)
    wall = time.perf_counter() - t0
    if args.dump:
        write_artefacts(run, result, write_clusters=True)
    run.close(status="ok")

    n = result["stats"]["n_frames_processed"]
    print(f"\nMODE={args.mode} limit={args.limit} frames={n} "
          f"wall={wall:.1f}s  fps={n/wall:.3f}  "
          f"clusters={result['stats']['n_frames_with_clusters']} "
          f"spatial_reject={result['stats']['spatial_rejections']}"
          f"/{result['stats']['spatial_clusters_seen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
