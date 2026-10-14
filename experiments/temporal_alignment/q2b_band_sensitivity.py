# PURPOSE: Quantify how much the measured cloud speed depends on the spatial
#          band the block matcher is given, because that sensitivity bounds how
#          firmly any single speed can be claimed. The same frame pairs are
#          re-measured with four high-pass filters of different scale (three box
#          kernels and the Gaussian used first).
# INPUTS: the image archive and the stored stereo height series.
# OUTPUTS: experiments/temporal_alignment/q2b_band_sensitivity.json and a
#          printed table.
# KEYWORDS: sensitivity, bandpass, cloud_speed, systematic_uncertainty
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import common  # noqa: E402
import q2_cloud_motion as q2  # noqa: E402

STRIDE = 12   # process every 12th consecutive pair (~2 min apart) for speed


# PURPOSE: Temporarily replace q2's band-pass with a Gaussian of given sigma.
# INPUTS: sigma (float).
# OUTPUTS: a context manager-free helper is not needed; this returns the
#          replacement function so the caller can restore the original.
# KEYWORDS: monkeypatch, gaussian, comparison
def gaussian_band(sigma):
    def fn(gray):
        x = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
        return x - x.mean()
    return fn


# PURPOSE: Measure the per-pair speed under one band-pass configuration.
# INPUTS: g0, g1 (central crops); centres; when; height_m; dt_s; band_fn.
# OUTPUTS: float speed in m/s, or None when the field cannot be fitted.
# KEYWORDS: measurement, block_matching, plane_fit
def speed_under(g0, g1, centres, when, height_m, dt_s, band_fn):
    d = q2.frame_displacement(g0, g1, centres, when, band_fn=band_fn)
    if d is None:
        return None
    fit = q2.fit_plane_translation(d["patches"])
    if fit is None:
        return None
    return float(height_m * fit["speed_factor"] / dt_s)


# PURPOSE: Entry point: sample pairs, measure under each band, correlate, report.
# INPUTS: none.
# OUTPUTS: q2b_band_sensitivity.json.
# KEYWORDS: main, entry_point
def main():
    series = common.load_series(common.RUN_ROLL_CORRECTED)
    a_ts, h_all = series["a_ts"], series["filt_h"]
    names = sorted(glob.glob(os.path.join(common.IMAGES_DIR,
                                          "img-2025-05-23T*devID1.jpg")))
    stamps = [(q2.parse_name(n), n) for n in names]
    stamps = [(t, n) for t, n in stamps
              if t is not None and q2.T_START_UTC <= t <= q2.T_END_UTC]
    stamps.sort(key=lambda x: x[0])
    # Build explicit callables: box kernels read q2.BLUR_KSIZE at call time.
    def box_band(k):
        def fn(gray):
            old = q2.BLUR_KSIZE
            q2.BLUR_KSIZE = k
            try:
                return q2.bandpass(gray)
            finally:
                q2.BLUR_KSIZE = old
        return fn

    bands = {"box121": box_band(121), "box241": box_band(241),
             "box481": box_band(481), "gauss120": gaussian_band(120)}
    results = {k: [] for k in bands}
    times = []
    n_done = 0
    for (t0, p0), (t1, p1) in zip(stamps[:-1], stamps[1:]):
        dt_s = (t1 - t0).total_seconds()
        if not (0 < dt_s <= 20):
            continue
        n_done += 1
        if n_done % STRIDE:
            continue
        centres = q2.patch_centres(t0)
        g0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        if g0 is None or g1 is None:
            continue
        g0 = g0[q2.CROP_HALF: q2.CROP_HALF + q2.CROP_N,
                q2.CROP_HALF: q2.CROP_HALF + q2.CROP_N]
        g1 = g1[q2.CROP_HALF: q2.CROP_HALF + q2.CROP_N,
                q2.CROP_HALF: q2.CROP_HALF + q2.CROP_N]
        j = int(np.argmin(np.abs(a_ts - (t0.timestamp() + 3 * 3600.0))))
        h = float(h_all[j])
        times.append(t0.isoformat())
        for name, fn in bands.items():
            results[name].append(speed_under(g0, g1, centres, t0, h, dt_s, fn))
        print(f"  {len(times)} pairs  {t0.isoformat()}", flush=True)
    out = {"times": times, "stride": STRIDE}
    speeds = {}
    for name, vals in results.items():
        arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        speeds[name] = arr
        out[name] = arr.tolist()
    print(f"\nsampled pairs: {len(times)}")
    for name, arr in speeds.items():
        good = arr[np.isfinite(arr)]
        print(f"  {name:>9}: median {np.median(good):5.2f} m/s  "
              f"mean {good.mean():5.2f}  p10 {np.percentile(good,10):5.2f}  "
              f"p90 {np.percentile(good,90):5.2f}  n={good.size}")
    print("\ncorrelation between band choices:")
    keys = list(speeds)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            m = np.isfinite(speeds[a]) & np.isfinite(speeds[b])
            if m.sum() > 3:
                print(f"  {a:>9} vs {b:<9} r={np.corrcoef(speeds[a][m], speeds[b][m])[0,1]:.2f}  "
                      f"ratio med={np.median(speeds[a][m]/speeds[b][m]):.2f}")
    with open(os.path.join(HERE, "q2b_band_sensitivity.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
