# PURPOSE: Answer question 5 — how much of the stereo/ceilometer disagreement is
#          explainable by the spatial sampling mismatch (one vertical column
#          versus a footprint kilometres wide) rather than by a time lag. Three
#          independent quantities are measured: the camera footprint size from
#          the crop geometry; the spatial decorrelation length of the cloud
#          texture from the images themselves; and the within-frame spread of
#          stereo disparity from the cached correspondences, with a vertical-
#          versus-horizontal test that separates real cloud relief from match
#          noise. No height is produced here and no measurement data is changed.
# INPUTS: the image archive and experiments/w1/matches_cache_2025-05-23.npz.
# OUTPUTS: experiments/temporal_alignment/q5_spatial_sampling.json and
#          q5_spatial_sampling.png, plus a printed report.
# KEYWORDS: spatial_sampling, footprint, decorrelation, disparity_spread,
#           noise_vs_relief
from __future__ import annotations

import glob
import json
import math
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

CACHE = os.path.join(REPO, "experiments", "w1", "matches_cache_2025-05-23.npz")
F = 654.0
BASELINE = 45.0
CROP = 640


# PURPOSE: Compute the camera footprint size on the cloud plane and the ratio to
#          the ceilometer's column, which is the sampling-mismatch scale.
# INPUTS: height_m (float), crop_px (int), f_px_per_rad (float).
# OUTPUTS: dict {half_angle_rad, footprint_radius_m, footprint_diameter_m,
#          column_to_footprint_ratio} (ratio uses a 10 m representative
#          ceilometer beam width at range).
# KEYWORDS: footprint, geometry, sampling
def footprint(height_m, crop_px=CROP, f_px_per_rad=F):
    half_px = crop_px / 2.0
    half_angle = half_px / f_px_per_rad
    radius = height_m * math.tan(half_angle)
    return {"half_angle_rad": half_angle,
            "half_angle_deg": math.degrees(half_angle),
            "footprint_radius_m": radius,
            "footprint_diameter_m": 2 * radius,
            "column_to_footprint_ratio": (2 * radius) / 10.0}


# PURPOSE: Estimate the spatial decorrelation length of the cloud texture from a
#          single band-passed frame, as the lag at which the radial
#          autocorrelation falls to 1/e.
# INPUTS: gray (2-D frame); when (datetime) for sun exclusion.
# OUTPUTS: dict {lag_px, lag_m, height_m} where lag_m uses the local angular
#          scale height/f; lag_px is NaN when the curve never reaches 1/e in the
#          measured range (over 200 px).
# KEYWORDS: autocorrelation, decorrelation_length, homogeneity
def texture_decorrelation(gray, when, height_m):
    # A coarse high-pass (sigma 250 px) removes only the lens vignette, so the
    # autocorrelation reflects cloud structure rather than pixel-scale texture.
    b = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 250)
    n = 1024
    c = 960 - n // 2
    win = b[c:c + n, c:c + n] * cv2.createHanningWindow((n, n), cv2.CV_32F)
    # Mask out the sun and anything outside the inner sky disc.
    zen, az = q2.sun_position(when)
    imaz = math.radians((az - q2.AZIMUTH_OFFSET) % 360.0)
    sr = F * math.radians(zen)
    sx, sy = 960 + sr * math.cos(imaz) - c, 960 + sr * math.sin(imaz) - c
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - n / 2, yy - n / 2)
    win[rr > 480] = 0
    if np.hypot(sx - n / 2, sy - n / 2) < 600:
        win[np.hypot(xx - sx, yy - sy) < 180] = 0
    win = win - win.mean()
    spec = np.fft.rfft2(win)
    ac = np.fft.irfft2(spec * np.conj(spec), s=win.shape)
    ac = np.fft.fftshift(ac)
    ac /= ac[n // 2, n // 2] + 1e-12
    # Radial average.
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - n / 2, yy - n / 2).astype(int)
    prof = np.bincount(rr.ravel(), weights=ac.ravel()) / np.maximum(
        np.bincount(rr.ravel()), 1)
    lags = np.arange(len(prof))
    below = np.where(prof < math.exp(-1))[0]
    lag_px = float(below[0]) if below.size and prof[0] > math.exp(-1) else float("nan")
    return {"lag_px": lag_px,
            "lag_m": float(height_m * lag_px / F) if np.isfinite(lag_px) else float("nan"),
            "height_m": float(height_m),
            "profile": [float(x) for x in prof[:201]]}


# PURPOSE: Measure the within-frame spread of stereo disparity from cached
#          correspondences, split into horizontal (relief plus noise) and
#          vertical (noise only) components after the measured roll correction.
#          If the two spreads are comparable the horizontal spread is match
#          noise, not cloud relief.
# INPUTS: cache_path (str), roll_deg (float, applied to camera 2's observations),
#         f (float), baseline (float).
# OUTPUTS: list of per-frame dicts {stamp, n, med_du, iqr_du, std_du, iqr_dv,
#          std_dv, med_height, height_iqr}; writes nothing.
# KEYWORDS: disparity, relief, match_noise, roll_correction
def disparity_spread(cache_path, roll_deg=1.5, f=F, baseline=BASELINE):
    data = np.load(cache_path, allow_pickle=True)
    kp1s, kp2s, stamps = data["kp1"], data["kp2"], data["stamps"]
    a = math.radians(roll_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = []
    for k1, k2, st in zip(kp1s, kp2s, stamps):
        d2 = k2 - 960.0
        k2r = d2 @ R.T + 960.0
        d = k2r - k1
        du, dv = d[:, 0], d[:, 1]
        med_du = float(np.median(du))
        # A translation is common to both axes; remove it so only spread remains.
        du = du - np.median(du)
        dv = dv - np.median(dv)
        iqr_du = float(np.percentile(du, 75) - np.percentile(du, 25))
        iqr_dv = float(np.percentile(dv, 75) - np.percentile(dv, 25))
        med_h = f * baseline / med_du if med_du != 0 else float("nan")
        # Height from the IQR edges: scatter of D implied by the du quartiles.
        hi = f * baseline / (med_du + iqr_du / 2)
        lo = (f * baseline / (med_du - iqr_du / 2)
              if med_du - iqr_du / 2 > 0 else float("nan"))
        out.append({
            "stamp": str(st), "n": int(len(k1)),
            "med_du": med_du, "iqr_du": float(iqr_du), "std_du": float(np.std(du)),
            "iqr_dv": float(iqr_dv), "std_dv": float(np.std(dv)),
            "med_height": float(med_h),
            "height_at_du_plus": float(hi), "height_at_du_minus": float(lo),
        })
    return out


# PURPOSE: Plot the decorrelation profile and the within-frame disparity spreads.
# INPUTS: decor (dict), disp (list), out_png (str).
# OUTPUTS: writes the PNG; returns None.
# KEYWORDS: figure, decorrelation, disparity
def make_figure(decor, disp, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(decor["profile"], lw=1)
    ax[0].axhline(math.exp(-1), color="r", ls="--", label="1/e")
    ax[0].set_xlabel("lag (px)")
    ax[0].set_ylabel("radial autocorrelation")
    ax[0].set_title("cloud texture decorrelation")
    ax[0].legend()
    ax[1].scatter([d["med_du"] for d in disp], [d["iqr_du"] for d in disp],
                  label="horizontal (relief+noise)")
    ax[1].scatter([d["med_du"] for d in disp], [d["iqr_dv"] for d in disp],
                  marker="^", label="vertical (noise only)")
    ax[1].set_xlabel("median horizontal disparity (px)")
    ax[1].set_ylabel("within-frame IQR (px)")
    ax[1].legend()
    ax[1].set_title("within-frame disparity spread")
    hh = [d["med_height"] for d in disp]
    ax[2].hist(hh, bins=15)
    ax[2].set_xlabel("per-frame median stereo height (m)")
    ax[2].set_ylabel("frames")
    ax[2].set_title("frame-to-frame level scatter")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"figure written: {out_png}")


# PURPOSE: Entry point: compute all three quantities, print, save JSON, figure.
# INPUTS: none.
# OUTPUTS: q5_spatial_sampling.json and q5_spatial_sampling.png.
# KEYWORDS: main, entry_point
def main():
    series = common.load_series(common.RUN_ROLL_CORRECTED)
    h_med = float(np.median(series["filt_h"]))
    out = {"height_used_m": h_med, "footprint": footprint(h_med)}
    print("footprint at median stereo height "
          f"{h_med:.0f} m: radius {out['footprint']['footprint_radius_m']:.0f} m "
          f"(diameter {out['footprint']['footprint_diameter_m']:.0f} m, "
          f"half-angle {out['footprint']['half_angle_deg']:.1f} deg)")
    # One representative frame for the texture decorrelation (noon, textured sky).
    names = sorted(glob.glob(os.path.join(common.IMAGES_DIR,
                                          "img-2025-05-23T12-*devID1.jpg")))
    decors = []
    for p in names[::30][:8]:
        t = q2.parse_name(p)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        decors.append(texture_decorrelation(g, t, h_med))
    out["texture_decorrelation"] = decors
    lags = np.array([d["lag_px"] for d in decors], dtype=float)
    lags_m = np.array([d["lag_m"] for d in decors], dtype=float)
    print(f"texture 1/e decorrelation over {len(decors)} noon frames: "
          f"median {np.nanmedian(lags):.0f} px = {np.nanmedian(lags_m):.0f} m "
          f"(n={int(np.sum(np.isfinite(lags)))})")
    disp = disparity_spread(CACHE)
    out["disparity_spread"] = disp
    du = np.array([d["std_du"] for d in disp])
    dv = np.array([d["std_dv"] for d in disp])
    hh = np.array([d["med_height"] for d in disp])
    print(f"within-frame disparity spread over {len(disp)} cached frames (roll +1.5 deg):")
    print(f"  horizontal std: median {np.median(du):.2f} px   median IQR "
          f"{np.median([d['iqr_du'] for d in disp]):.2f} px")
    print(f"  vertical   std: median {np.median(dv):.2f} px   median IQR "
          f"{np.median([d['iqr_dv'] for d in disp]):.2f} px")
    print(f"  ratio median std(du)/std(dv) = {np.median(du/dv):.2f}")
    print(f"  per-frame median height: median {np.median(hh):.0f} m, IQR "
          f"{np.percentile(hh,75)-np.percentile(hh,25):.0f} m")
    med_du = float(np.median([d["med_du"] for d in disp]))
    iqr_du = float(np.median([d["iqr_du"] for d in disp]))
    out["summary"] = {
        "texture_decorr_px_median": float(np.nanmedian(lags)),
        "texture_decorr_m_median": float(np.nanmedian(lags_m)),
        "median_within_frame_std_du_px": float(np.median(du)),
        "median_within_frame_std_dv_px": float(np.median(dv)),
        "median_std_ratio_du_over_dv": float(np.median(du / dv)),
        "median_within_frame_iqr_du_px": iqr_du,
        "median_med_du_px": med_du,
        "within_frame_height_fractional_spread": float(iqr_du / abs(med_du)),
        "per_frame_height_iqr_m": float(np.percentile(hh, 75) - np.percentile(hh, 25)),
        "relief_induced_du_std_px": float(math.sqrt(max(
            0.0, np.median(du) ** 2 - np.median(dv) ** 2))),
        "relief_height_fractional_std": float(math.sqrt(max(
            0.0, np.median(du) ** 2 - np.median(dv) ** 2)) / abs(med_du)),
    }
    print("summary:", json.dumps(out["summary"], indent=1))
    make_figure(decors[0], disp, os.path.join(HERE, "q5_spatial_sampling.png"))
    with open(os.path.join(HERE, "q5_spatial_sampling.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
