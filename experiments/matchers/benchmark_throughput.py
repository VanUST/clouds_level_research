# PURPOSE: Measure backend throughput on its own, with the matcher call isolated
#          from the CPU-side metric computation that the evaluation loop also
#          performs. The evaluation JSON's ms field is not a clean matcher timing
#          because the plane fit runs in the same timed region; this script exists
#          so the reported ms/frame is a measurement of the matcher and nothing
#          else. Speed is reported but never used to rank (pre-registration §4).
# INPUTS: a list of backend names and the two sessions; N frames per backend.
# OUTPUTS: experiments/matchers/results/throughput.json and a stdout table.
# KEYWORDS: throughput, timing, benchmark, matcher_only
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

import harness  # noqa: E402
from clouds.matching.matchers import create_matcher  # noqa: E402

BACKENDS = [
    ("kornia_loftr", {}),
    ("efficient_loftr", {}),
    ("roma", {"grid_stride": 8, "certainty_threshold": 0.5}),
    ("dkm", {"grid_stride": 8, "certainty_threshold": 0.5}),
    ("mast3r", {}),
    ("da3", {}),
    ("cv2_dis_flow", {"grid_stride": 8}),
    ("cv2_sgbm", {"grid_stride": 8}),
    ("disk_lightglue", {}),
]


def main():
    import torch
    cfg = harness.load_cfg(None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = harness.select_frames(cfg, "2025-05-23", t0="12:00:00", t1="13:00:00", step_s=600)
    pairs = pairs[:6]
    prepped = []
    for p1, p2, st in pairs:
        prep = harness.prepare_crop(cfg, p1, p2)
        if prep is not None:
            prepped.append((prep[0], prep[1]))
    out = {}
    for name, ccfg in BACKENDS:
        try:
            m = create_matcher(name, ccfg, dev)
            m(*prepped[0])
            if dev == "cuda":
                torch.cuda.synchronize()
            ts = []
            for g1, g2 in prepped:
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                m(g1, g2)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000.0)
            out[name] = {"ms_median": float(np.median(ts)), "ms_all": [float(x) for x in ts],
                         "n_frames": len(ts), "device": dev}
            print(f"{name:<17} {np.median(ts):8.1f} ms   (min {min(ts):.1f}, max {max(ts):.1f})")
        except Exception as e:  # a backend that cannot be built is reported, not hidden
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{name:<17} ERROR {type(e).__name__}: {e}")
    with open(os.path.join(HERE, "results", "throughput.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/throughput.json")


if __name__ == "__main__":
    main()
