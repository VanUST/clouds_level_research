# PURPOSE: Run one matcher backend over a fixed set of real cloud stereo pairs and
#          measure the pre-registered ground-truth-free criteria, writing one JSON
#          per (backend, session) so the report can aggregate without re-running the
#          GPU. The criteria live in harness.py; this file only drives them, so the
#          definition of a metric cannot drift between candidates.
# INPUTS: a backend name, a session date, an optional backend-config JSON, and the
#         frame grid (defaults from the pre-registration).
# OUTPUTS: experiments/matchers/results/<backend>_<date>.json with per-frame
#          metrics, timing, and the perturbation-stability summary; stdout progress.
# KEYWORDS: matcher_evaluation, runner, criteria, throughput, reproducibility
from __future__ import annotations

import argparse
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

RESULTS = os.path.join(HERE, "results")

# Coverage-matched comparison size, fixed in the pre-registration amendment: every
# backend that exposes a per-match confidence is reduced to its K most confident
# matches per frame before the plane fit, so a backend is not judged on how many
# low-confidence outliers its native threshold happens to emit.
TOPK = 2500


# PURPOSE: Ask the matcher for its native output together with a per-match
#          confidence when the backend exposes one, and return the two metric
#          bundles the pre-registration defines: the native one, and the
#          coverage-matched one over the TOPK most confident matches.
# INPUTS: matcher (callable); g1, g2 (2-D uint8); f, baseline (floats).
# OUTPUTS: (kp1, kp2, conf, native_metrics, topk_metrics_or_None).
# KEYWORDS: confidence, topk, coverage_matched, criteria
def measure(matcher, g1, g2, f, baseline):
    res = matcher(g1, g2, return_conf=True)
    if isinstance(res, tuple) and len(res) == 3:
        kp1, kp2, conf = res
    else:
        kp1, kp2 = res[0], res[1]
        conf = None
    kp1 = np.asarray(kp1, float).reshape(-1, 2)
    kp2 = np.asarray(kp2, float).reshape(-1, 2)
    native = {"plane": harness.plane_fit(kp1, kp2, f, baseline),
              "epi": harness.epipolar_split(kp1, kp2),
              "radial": harness.radial_profile(kp1, kp2, f, baseline)}
    # Accuracy inside cloud interiors specifically: the same plane fit restricted to
    # matches in the lowest-gradient tercile of image 1, and to the highest.
    lo_mask, hi_mask, _ = harness.gradient_masks(g1)
    if len(kp1):
        ui = np.clip(kp1[:, 0].astype(int), 0, g1.shape[1] - 1)
        vi = np.clip(kp1[:, 1].astype(int), 0, g1.shape[0] - 1)
        in_lo, in_hi = lo_mask[vi, ui], hi_mask[vi, ui]
    else:
        in_lo = in_hi = np.empty(0, bool)
    native["plane_lowgrad"] = harness.plane_fit(kp1[in_lo], kp2[in_lo], f, baseline)
    native["plane_highgrad"] = harness.plane_fit(kp1[in_hi], kp2[in_hi], f, baseline)
    topk = None
    if conf is not None:
        conf = np.asarray(conf, float).reshape(-1)
        if len(conf) == len(kp1) and len(conf) > TOPK:
            idx = np.argsort(-conf)[:TOPK]
            topk = {"plane": harness.plane_fit(kp1[idx], kp2[idx], f, baseline),
                    "epi": harness.epipolar_split(kp1[idx], kp2[idx]),
                    "radial": harness.radial_profile(kp1[idx], kp2[idx], f, baseline)}
    return kp1, kp2, conf, native, topk


# PURPOSE: Build the matcher once, run it over every selected frame, and return the
#          per-frame metrics plus the perturbation summary.
# INPUTS: args namespace (see main).
# OUTPUTS: a JSON-serialisable dict; writes it to RESULTS unless --dry-run.
# KEYWORDS: run, criteria, frame_loop
def run(backend, date, cfg_extra, t0, t1, step_s, limit, perturb_n, dry_run, tag=""):
    import torch

    cfg = harness.load_cfg(None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f = cfg.camera.image_width_px / np.pi
    baseline = cfg.camera.stereo_base_m
    matcher = create_matcher(backend, cfg_extra, device)
    pairs = harness.select_frames(cfg, date=date, t0=t0, t1=t1, step_s=step_s, limit=limit)
    print(f"backend={backend} date={date} device={device} frames={len(pairs)} f={f:.2f} B={baseline}")

    # Warm-up: the first forward pass includes lazy kernel/weight preparation and
    # would otherwise be charged to frame 1 of the timing.
    if pairs:
        g1, g2, _, _ = harness.prepare_crop(cfg, pairs[0][0], pairs[0][1])
        matcher(g1, g2)
        if device == "cuda":
            torch.cuda.synchronize()

    frames = []
    for i, (p1, p2, stamp) in enumerate(pairs):
        prep = harness.prepare_crop(cfg, p1, p2)
        if prep is None:
            continue
        g1, g2, _, _ = prep
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        kp1, kp2, conf, native, topk = measure(matcher, g1, g2, f, baseline)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000.0
        rec = {"stamp": stamp, "n_matches": int(len(kp1)), "ms": float(ms),
               "plane": native["plane"], "epi": native["epi"], "radial": native["radial"],
               "plane_lowgrad": native["plane_lowgrad"],
               "plane_highgrad": native["plane_highgrad"],
               "grad": harness.gradient_strata(g1, kp1)}
        if topk is not None:
            rec["topk"] = topk
        frames.append(rec)
        print(f"  [{i + 1:3d}/{len(pairs)}] {stamp}  n={len(kp1):5d}  {ms:7.1f} ms  "
              f"rms={native['plane']['rms_full_px']:6.2f} px  "
              f"medabs={native['plane']['med_abs_res_px']:5.2f} px  "
              f"dv_rand={native['epi']['dv_rand_rms_px']:5.2f} "
              f"h_med={native['radial']['height_median_m']:7.0f} m  "
              f"lowgrad={rec['grad']['low_grad_frac']:.2f}")

    # Perturbation stability on a subset: shift image 2 by a known sub-pixel amount
    # and measure the spread of the induced disparity change over matches whose
    # image-1 point is the same in both runs.
    perturb = {"dx": 0.5, "dy": 0.25, "frames": []}
    for p1, p2, stamp in pairs[:perturb_n]:
        prep = harness.prepare_crop(cfg, p1, p2)
        if prep is None:
            continue
        g1, g2, _, _ = prep
        kp1, kp2 = matcher(g1, g2)
        kp1 = np.asarray(kp1, float).reshape(-1, 2)
        kp2 = np.asarray(kp2, float).reshape(-1, 2)
        g2p = harness.subpixel_shift(g2, perturb["dx"], perturb["dy"])
        kp1p, kp2p = matcher(g1, g2p)
        kp1p = np.asarray(kp1p, float).reshape(-1, 2)
        kp2p = np.asarray(kp2p, float).reshape(-1, 2)
        ia, ib = harness.pair_by_image1(kp1, kp1p, tol=1.5)
        rec = {"stamp": stamp, "n_ref": int(len(kp1)), "n_pert": int(len(kp1p)),
               "n_paired": int(len(ia))}
        if len(ia) >= 10:
            ddu = (kp1[ia, 0] - kp2[ia, 0]) - (kp1p[ib, 0] - kp2p[ib, 0])
            ddv = (kp1[ia, 1] - kp2[ia, 1]) - (kp1p[ib, 1] - kp2p[ib, 1])
            rec["ddu_median_px"] = float(np.median(ddu))
            rec["ddu_rms_px"] = float(np.sqrt(np.mean((ddu - np.median(ddu)) ** 2)))
            rec["ddv_rms_px"] = float(np.sqrt(np.mean((ddv - np.median(ddv)) ** 2)))
        perturb["frames"].append(rec)
    print(f"perturbation: {len(perturb['frames'])} frames")

    out = {"backend": backend, "date": date, "device": device, "f_px_per_rad": f,
           "baseline_m": baseline, "config_extra": cfg_extra,
           "frame_grid": {"t0": t0, "t1": t1, "step_s": step_s, "limit": limit},
           "frames": frames, "perturb": perturb,
           "versions": _versions()}
    if not dry_run:
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, f"{backend}{tag}_{date}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {path}")
    return out


# PURPOSE: Capture the exact package versions behind a result, so a number can be
#          traced to the software that produced it (guardrail G11).
# INPUTS: none.
# OUTPUTS: dict of package name -> version for the ones that matter here.
# KEYWORDS: provenance, versions, reproducibility
def _versions():
    import importlib.metadata as md
    out = {}
    for name in ("torch", "kornia", "romatch", "einops", "yacs", "timm",
                 "opencv-python", "numpy", "scipy"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--date", default="2025-05-23")
    ap.add_argument("--t0", default="08:30:00")
    ap.add_argument("--t1", default="16:30:00")
    ap.add_argument("--step-s", type=int, default=900)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--perturb-n", type=int, default=8)
    ap.add_argument("--cfg-json", default="{}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    a = ap.parse_args()
    run(a.backend, a.date, json.loads(a.cfg_json), a.t0, a.t1, a.step_s,
        a.limit, a.perturb_n, a.dry_run, a.tag)


if __name__ == "__main__":
    main()
