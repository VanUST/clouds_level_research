# PURPOSE: Produce the raw per-match evidence W4 needs, with RoMa's own certainty
#          attached to every correspondence.
#
#          WHY A NEW DUMP. W3.4 showed the pipeline's clustering and selection act as a
#          structural low-pass filter: the median disparity tracks the cloud regime at
#          rho = +0.55, while every statistic the pipeline reports is flat (|shift| <= 30 m,
#          p > 0.05). The matched population must therefore be studied BEFORE clustering.
#
#          No existing artefact can serve. heights.npz carries stamps, heights, uv, kp2
#          and the geometry constants but NO certainty; matches_cache_2025-05-23.npz
#          carries kp1, kp2, stamps and frames, also without certainty. Certainty is the
#          first gate of the W4 triad, so it has to be captured at the source.
#
#          The preparation is byte-identical to the pipeline's - the same affine warp,
#          the same roll correction, the same centre crop - so the population recorded
#          here is the population the pipeline actually sees, not a variant of it.
# INPUTS: the image archive, configs/paper_dlcp2026_roll.yaml.
# OUTPUTS: experiments/w4/matches_roma.npz with, per frame, the matched coordinates in
#          the cropped frame, RoMa's certainty, and the frame's filename stamp.
# KEYWORDS: W4, dump, raw matches, certainty, population, bypass clustering
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

IMAGES = "/home/iustimov/clouds_data/images"
OUT = None  # set per-date in main()
NAME = re.compile(r"^img-(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})devID(\d)\.jpg$")


def main():
    from clouds.config import load_config
    from clouds.geometry.imaging import center_crop, rotation_about
    from clouds.matching.matchers import create_matcher

    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    M = np.array(cfg.calibration_record.matrix_2x3)
    roll = float(cfg.camera.roll_correction_deg or 0.0)
    crop = cfg.camera.target_size[0]
    orig = cfg.camera.original_size
    cxy = (orig[0] / 2.0, orig[1] / 2.0)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    date = sys.argv[3] if len(sys.argv) > 3 else "2025-05-23"

    names = sorted(f for f in os.listdir(IMAGES)
                   if f.startswith(f"img-{date}") and f.endswith("devID1.jpg")
                   and "08-" <= f[15:18] <= "17-")
    names = names[::stride][:limit]
    global OUT
    OUT = os.path.join(HERE, f"matches_roma_{date.replace('-', '')}.npz")
    print(f"{date}: {len(names)} pairs (stride {stride}), crop {crop}, roll {roll}")

    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")

    stamps, uv, kp2, conf = [], [], [], []
    for k, nm in enumerate(names):
        base = nm.replace("devID1.jpg", "")
        i1 = cv2.imread(os.path.join(IMAGES, nm))
        i2 = cv2.imread(os.path.join(IMAGES, base + "devID2.jpg"))
        if i1 is None or i2 is None:
            continue
        if roll:
            i2 = cv2.warpAffine(i2, rotation_about(cxy, roll), dsize=orig)
        w1 = cv2.warpAffine(i1, M, dsize=orig)
        g1 = cv2.cvtColor(center_crop(w1, crop, crop), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(center_crop(i2, crop, crop), cv2.COLOR_BGR2GRAY)
        a, b, c = matcher(g1, g2, return_conf=True)
        a = np.asarray(a, float).reshape(-1, 2)
        b = np.asarray(b, float).reshape(-1, 2)
        c = np.asarray(c, float).reshape(-1)
        stamps.append(base)
        uv.append(a.astype(np.float32))
        kp2.append(b.astype(np.float32))
        conf.append(c.astype(np.float32))
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(names)}  last frame {len(a)} matches")
    np.savez_compressed(OUT, stamps=np.array(stamps, dtype="U32"),
                        uv=np.array(uv, dtype=object),
                        kp2=np.array(kp2, dtype=object),
                        conf=np.array(conf, dtype=object))
    tot = sum(len(x) for x in uv)
    print(f"\nwrote {OUT}: {len(stamps)} frames, {tot} matches "
          f"({tot/max(len(stamps),1):.0f} per frame)")


if __name__ == "__main__":
    main()
