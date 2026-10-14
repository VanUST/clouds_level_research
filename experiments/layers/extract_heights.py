# PURPOSE: Extract per-match cloud heights for a sample of frames, so that layer
#          structure can be studied directly from the height distribution instead of
#          from the clustered output.
#
#          Why this exists: the pipeline reduces each frame to one number via the
#          "lowest cluster" rule, which cannot express the fact that the sky often
#          contains several distinct cloud layers at once. The ceilometer reports two
#          (H1, H2) and its lowest layer switches between them, so a single-number
#          comparison is comparing a mixture against a mixture. To test the layer
#          hypothesis we need the full per-match height distribution per frame.
#
#          Heights use the exact unit-sphere triangulation, not the paraxial formula.
#          The roll correction is applied (it is a measured calibration residual).
#          The ceilometer is not read here at all.
# INPUTS: the 2025-05-23 image session and the project config.
# OUTPUTS: experiments/layers/heights.npz — per frame: timestamp, per-match heights,
#          per-match image coordinates, and the match count.
# KEYWORDS: layer_extraction, per_match_heights, dataset_build, triangulation
from __future__ import annotations

import datetime as dt
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from clouds.config import load_config                                        # noqa: E402
from clouds.geometry.camera import CameraModel, StereoCameraSystem           # noqa: E402
from clouds.geometry.triangulation import heights_from_matches               # noqa: E402
from clouds.matching.filters import get_corresponding_feature_points         # noqa: E402
from clouds.pipeline.analyze import StereoAnalyzer                           # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
OUT = os.path.join(HERE, "heights.npz")


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    stereo = StereoCameraSystem(base=cfg.camera.stereo_base_m,
                                angle_of_view=cfg.camera.angle_of_view_deg,
                                image_width=cfg.camera.image_width_px,
                                model=CameraModel(cfg.camera.model))
    analyzer = StereoAnalyzer(cfg, stereo)
    f = cfg.camera.image_width_px / np.pi
    B = cfg.camera.stereo_base_m
    crop = cfg.camera.target_size[0]
    centre = (crop / 2.0, crop / 2.0)          # principal point in the cropped frame

    shift_h = cfg.dataset.clock_offset_h
    t0 = (dt.datetime.strptime(cfg.dataset.window_start, "%H:%M:%S")
          - dt.timedelta(hours=shift_h)).strftime("%H-%M-%S")
    t1 = (dt.datetime.strptime(cfg.dataset.window_end, "%H:%M:%S")
          - dt.timedelta(hours=shift_h)).strftime("%H-%M-%S")
    names = sorted(n for n in os.listdir(IMAGES)
                   if n.endswith("devID1.jpg") and n.startswith("img-2025-05-23")
                   and t0 <= n[15:23] <= t1)
    names = names[::stride]
    print(f"config    : {os.path.basename(cfg.source_path) if hasattr(cfg,'source_path') else 'paper_dlcp2026_roll'}")
    print(f"roll      : {cfg.camera.roll_correction_deg} deg   f = {f:.1f} px/rad   B = {B} m")
    print(f"frames    : {len(names)} (stride {stride} of the window)\n")

    stamps, heights, kp1s, kp2s = [], [], [], []
    for i, n in enumerate(names):
        base = n.replace("devID1.jpg", "")
        p1 = os.path.join(IMAGES, n)
        p2 = os.path.join(IMAGES, base + "devID2.jpg")
        prep = analyzer.prepare_pair(p1, p2)
        if prep is None:
            continue
        kp1, kp2 = get_corresponding_feature_points(
            prep[2], prep[3], visualize=False,
            match_thresh=cfg.matching.match_threshold,
            backend=cfg.matching.backend, filter_config=cfg.matching.filter)
        if len(kp1) < 50:
            continue
        r = heights_from_matches(kp1, kp2, f, B, centre=centre)
        h = r["heights"]
        keep = np.isfinite(h) & (h > 50) & (h < 20000)
        stamps.append(base.replace("img-", ""))
        heights.append(h[keep].astype(np.float32))
        kp1s.append(kp1[keep].astype(np.float32))
        kp2s.append(kp2[keep].astype(np.float32))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(names)}] {base}  n={int(keep.sum())}  "
                  f"median={np.median(h[keep]):.0f} m", flush=True)

    np.savez_compressed(OUT, stamps=np.array(stamps),
                        heights=np.array(heights, dtype=object),
                        uv=np.array(kp1s, dtype=object),
                        kp2=np.array(kp2s, dtype=object),
                        f=f, baseline=B, centre=np.array(centre))
    tot = sum(len(h) for h in heights)
    print(f"\nwrote {OUT}: {len(stamps)} frames, {tot} matches, "
          f"median {np.median(np.concatenate(heights)):.0f} m")
    print(f"frame match counts: min {min(len(h) for h in heights)}, "
          f"median {int(np.median([len(h) for h in heights]))}, "
          f"max {max(len(h) for h in heights)}")


if __name__ == "__main__":
    main()
