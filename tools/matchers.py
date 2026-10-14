# PURPOSE: Give the matcher registry a supported command-line surface, so that any
#          backend can be listed, smoke-tested, and verified without editing the
#          pipeline. The matchers are an API, not a fixed choice: RoMa is the default
#          on measured grounds, and the alternatives must stay reachable for future
#          verification and for re-testing on new data.
#
#          `list` reports every registered backend and whether its dependencies are
#          importable in this environment; `check` runs one backend over real frames
#          and reports the quantities that decided the selection, so a claim about a
#          matcher can be re-derived rather than taken on trust.
# INPUTS: a subcommand; for `check`, a backend name, optional params JSON, and a
#         frame count.
# OUTPUTS: stdout; exit code 1 if a requested backend cannot be loaded.
# KEYWORDS: matchers, api, cli, verification, registry, reproducibility
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from clouds.matching.matchers import (create_matcher, get_available_backends,   # noqa: E402
                                      supports_batching)

IMAGES = "/home/iustimov/clouds_data/images"

# Which third-party package each backend needs, so `list` can say why one is absent.
DEPENDENCY = {
    "kornia_loftr": "kornia",
    "efficient_loftr": "efficient_loftr (cloned source)",
    "roma": "romatch",
    "dkm": "dkm",
    "mast3r": "mast3r (cloned source)",
    "da3": "depth_anything_3",
    "disk_lightglue": "kornia",
    "cv2_dis_flow": "opencv",
    "cv2_sgbm": "opencv",
    "matchanything_eloftr": "MatchAnything (third_party)",
}


def frames(n):
    """Real rectified cloud crops, the same preprocessing the pipeline applies."""
    from clouds.config import load_config
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    crop = cfg.camera.target_size[0]
    names = sorted(x for x in os.listdir(IMAGES)
                   if x.endswith("devID1.jpg") and x.startswith("img-2025-05-23")
                   and "10-" <= x[15:18] <= "14-")[:n]
    out = []
    for nm in names:
        base = nm.replace("devID1.jpg", "")
        i1 = cv2.imread(os.path.join(IMAGES, nm))
        i2 = cv2.imread(os.path.join(IMAGES, base + "devID2.jpg"))
        if i1 is None or i2 is None:
            continue
        w = cv2.warpAffine(i1, M, dsize=cfg.camera.original_size)
        y = (w.shape[0] - crop) // 2
        out.append((cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)[y:y + crop, y:y + crop],
                    cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)[y:y + crop, y:y + crop]))
    return out


def cmd_list(_args):
    print(f"{'backend':>22} {'dependency':>30} {'batching':>9} {'loadable':>9}")
    for name in sorted(get_available_backends()):
        dep = DEPENDENCY.get(name, "?")
        ok, why = "yes", ""
        try:
            m = create_matcher(name, {}, "cpu")
            batch = "yes" if supports_batching(m) else "no"
        except Exception as exc:
            ok, why, batch = "NO", f" ({type(exc).__name__})", "-"
        print(f"{name:>22} {dep:>30} {batch:>9} {ok:>9}{why}")
    print("\nThe default is `roma`; see docs/plan/matchers.md for why, and for the")
    print("measurements that rejected the alternatives for THIS task.")
    return 0


def cmd_check(args):
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as exc:
        print(f"--params must be JSON: {exc}")
        return 1
    try:
        matcher = create_matcher(args.backend, params, args.device)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"backend {args.backend}  params {params}  device {args.device}")
    fs = frames(args.n)
    print(f"frames {len(fs)}\n")
    print(f"{'frame':>26} {'matches':>9} {'median disp px':>15} {'median |dv| px':>15}")
    ns, ds, dvs = [], [], []
    for g1, g2 in fs:
        try:
            k0, k1 = matcher(g1, g2)
        except Exception:
            traceback.print_exc()
            return 1
        k0 = np.asarray(k0, float).reshape(-1, 2)
        k1 = np.asarray(k1, float).reshape(-1, 2)
        if len(k0) == 0:
            print(f"{'':>26} {0:>9}")
            continue
        du = k0[:, 0] - k1[:, 0]
        dv = np.abs(k0[:, 1] - k1[:, 1])
        ns.append(len(k0)); ds.append(float(np.median(du))); dvs.append(float(np.median(dv)))
        print(f"{'':>26} {len(k0):>9} {np.median(du):>15.2f} {np.median(dv):>15.2f}")
    if ns:
        print(f"\nmedian over frames: matches {np.median(ns):.0f}, "
              f"disparity {np.median(ds):.2f} px, |dv| {np.median(dvs):.2f} px")
        print("A working cloud matcher should report a positive disparity around 9-27 px")
        print("(2-3 km down to ~1 km) and a small vertical disparity. A backend that")
        print("reports a full field on a clear-sky pair is inventing matches; use")
        print("2025-05-22 to check that.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="List and verify the registered stereo matchers.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list registered backends and whether they load")
    c = sub.add_parser("check", help="run one backend over real frames")
    c.add_argument("backend")
    c.add_argument("--params", default="", help="JSON passed to the backend factory")
    c.add_argument("--n", type=int, default=4)
    c.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    return cmd_list(args) if args.cmd == "list" else cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
