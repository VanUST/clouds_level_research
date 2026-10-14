# PURPOSE: Answer question 2 — measure cloud motion from the imagery alone and
#          convert it to a linear cloud speed and direction, using the
#          instrument's own pixels (never the ceilometer) so guardrail G1 holds.
#          Method: block matching on consecutive frames of the same camera
#          (10 s apart), restricted to a purely-sky central disc so that trees,
#          buildings and the mast cannot anchor the match, with the sun disc
#          painted out. A translating cloud layer at height h projects to an
#          almost uniform image translation, so the median patch displacement is
#          the layer displacement and the patch spread measures how rigid the
#          field is.
# INPUTS: nothing (reads the image archive and the stored stereo height series).
# OUTPUTS: experiments/temporal_alignment/q2_cloud_motion.json and
#          q2_cloud_motion.png, plus a printed summary.
# KEYWORDS: optical_flow, block_matching, cloud_advection, angular_rate, speed
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "w1"))

import cv2  # noqa: E402
import common  # noqa: E402
from sun_reference import AZIMUTH_OFFSET, sun_position  # noqa: E402

F_PX_PER_RAD = 654.0        # W1.8 through-origin estimate (spread 642-665)
SKY_R = 450                 # purely-sky disc: high-pass median std < 4 px here
PATCH = 96
SEARCH = 60
BANDPASS_SIGMA = 120        # removes the static vignette / sky gradient
BLUR_KSIZE = 241            # odd box kernel wider than the vignette scale
SCORE_MIN = 0.7
# Every operation is confined to this central crop, which contains the whole
# sky disc (r < 480 px) and nothing else; the Blur/match cost falls ~4x versus
# filtering the full 1920x1920 frame for the same result.
CROP_HALF = 480
CROP_N = 2 * CROP_HALF
CENTRE = float(CROP_HALF)   # principal point in crop coordinates
# Analysis window in image (UTC) clock, matching the metric's ceilometer window
# 12:00-17:00 MSK.
T_START_UTC = dt.datetime(2025, 5, 23, 9, 0, 0)
T_END_UTC = dt.datetime(2025, 5, 23, 14, 0, 0)
# Optional override for a fast smoke run (Q2_SMOKE_MIN minutes from the start).
if os.environ.get("Q2_SMOKE_MIN"):
    T_END_UTC = T_START_UTC + dt.timedelta(
        minutes=float(os.environ["Q2_SMOKE_MIN"]))


# PURPOSE: Parse the acquisition time from an image filename.
# INPUTS: name (str) like "img-2025-05-23T09-00-03devID1.jpg".
# OUTPUTS: datetime, or None when the name does not match.
# KEYWORDS: timestamp, filename, parsing
def parse_name(name: str):
    base = os.path.basename(name).replace("img-", "")
    if not base.endswith("devID1.jpg"):
        return None
    stamp = base[: -len("devID1.jpg")]
    try:
        return dt.datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


# PURPOSE: High-pass a frame to suppress the static optical vignette so phase /
#          block matching locks onto cloud texture rather than the lens.
# INPUTS: gray (2-D float32 image).
# OUTPUTS: float32 image with zero mean and the smooth component removed.
#          A box blur is used (rather than a Gaussian with the same scale)
#          because its cost is independent of kernel width; the vignette is a
#          scale far above the cloud texture, so the box approximation is safe.
# KEYWORDS: bandpass, vignette, preprocessing
def bandpass(gray: np.ndarray) -> np.ndarray:
    x = gray - cv2.blur(gray, (BLUR_KSIZE, BLUR_KSIZE))
    return x - x.mean()


# PURPOSE: Build the list of patch centres inside the sky disc and away from the
#          sun's predicted image position.
# INPUTS: when (datetime UTC) — used for the solar ephemeris.
# OUTPUTS: list of (cx, cy) int pixel centres; empty when the Sun is high enough
#          that its disc would intrude on the sky disc.
# KEYWORDS: mask, sky_region, sun_exclusion
def patch_centres(when: dt.datetime) -> list:
    zen, az = sun_position(when)
    imaz = np.radians((az - AZIMUTH_OFFSET) % 360.0)
    sun_r = F_PX_PER_RAD * np.radians(zen)
    sx, sy = CENTRE + sun_r * np.cos(imaz), CENTRE + sun_r * np.sin(imaz)
    out = []
    for cy in range(int(CENTRE - SKY_R), int(CENTRE + SKY_R) + 1, 150):
        for cx in range(int(CENTRE - SKY_R), int(CENTRE + SKY_R) + 1, 150):
            if np.hypot(cx - CENTRE, cy - CENTRE) > SKY_R:
                continue
            if np.hypot(cx - sx, cy - sy) < 250.0:
                continue
            if (cx - PATCH // 2 - SEARCH < 0 or cy - PATCH // 2 - SEARCH < 0
                    or cx + PATCH // 2 + SEARCH > CROP_N - 1
                    or cy + PATCH // 2 + SEARCH > CROP_N - 1):
                continue
            out.append((cx, cy))
    return out


# PURPOSE: Estimate the sub-pixel integer-peak refinement of one match.
# INPUTS: res (2-D correlation map), peak (col,row) integer location.
# OUTPUTS: (dx, dy) float displacement relative to the search-window centre.
# KEYWORDS: subpixel, parabolic_fit, block_matching
def subpixel_peak(res: np.ndarray, peak) -> tuple:
    col, row = int(peak[0]), int(peak[1])

    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d

    def get(r, c):
        if 0 <= r < res.shape[0] and 0 <= c < res.shape[1]:
            return float(res[r, c])
        return 0.0

    fx = np.clip(para(get(row, max(col - 1, 0)), get(row, col),
                      get(row, min(col + 1, res.shape[1] - 1))), -1, 1)
    fy = np.clip(para(get(max(row - 1, 0), col), get(row, col),
                      get(min(row + 1, res.shape[0] - 1), col)), -1, 1)
    return col - SEARCH + fx, row - SEARCH + fy


# PURPOSE: Estimate the rigid cloud displacement between two frames from a grid
#          of block matches inside the sky disc.
# INPUTS: g0, g1 (2-D uint8 or float32 frames, same shape); centres (list);
#         when (datetime) for sun exclusion; band_fn (optional callable) to
#         override the high-pass filter, used by the band-sensitivity check.
# OUTPUTS: dict {patches: [(cx,cy,dx,dy,score), ...], dx, dy, mag, n_used,
#          spread_px, iqr_px, score_median} where dx/dy are the median
#          displacement of accepted patches; None when fewer than 3 patches are
#          accepted.
# KEYWORDS: block_matching, motion_field, robust_median
def frame_displacement(g0, g1, centres, when: dt.datetime, band_fn=None):
    b0 = (band_fn or bandpass)(g0.astype(np.float32))
    b1 = (band_fn or bandpass)(g1.astype(np.float32))
    zen, az = sun_position(when)
    imaz = np.radians((az - AZIMUTH_OFFSET) % 360.0)
    sun_r = F_PX_PER_RAD * np.radians(zen)
    sx, sy = CENTRE + sun_r * np.cos(imaz), CENTRE + sun_r * np.sin(imaz)
    patches, dxs, dys, scores = [], [], [], []
    for cx, cy in centres:
        if np.hypot(cx - sx, cy - sy) < 250.0:
            continue
        tmpl = b0[cy - PATCH // 2: cy + PATCH // 2, cx - PATCH // 2: cx + PATCH // 2]
        y0, x0 = cy - PATCH // 2 - SEARCH, cx - PATCH // 2 - SEARCH
        search = b1[y0: y0 + PATCH + 2 * SEARCH, x0: x0 + PATCH + 2 * SEARCH]
        if search.shape[0] != PATCH + 2 * SEARCH or search.shape[1] != PATCH + 2 * SEARCH:
            continue
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx < SCORE_MIN:
            continue
        dx, dy = subpixel_peak(res, ml)
        patches.append((float(cx), float(cy), dx, dy, float(mx)))
        dxs.append(dx)
        dys.append(dy)
        scores.append(float(mx))
    if len(dxs) < 3:
        return None
    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    mags = np.hypot(dxs, dys)
    return {
        "patches": patches,
        "dx": dx, "dy": dy, "mag": float(np.hypot(dx, dy)),
        "n_patches": len(centres), "n_used": len(dxs),
        "spread_px": float(np.std(mags)),
        "iqr_px": float(np.percentile(mags, 75) - np.percentile(mags, 25)),
        "score_median": float(np.median(scores)),
    }


# PURPOSE: Model the image displacement of a horizontally translating plane at
#          one height, projected through an equidistant fisheye r = f*theta.
#          The height cancels when positions are expressed in units of it, so
#          the two free parameters (a, b) are horizontal translations divided by
#          height: a = u*dt/h, b = w*dt/h.
# INPUTS: cx, cy (np.ndarray patch pixel centres); a, b (floats).
# OUTPUTS: (dx, dy) np.ndarray image displacements in pixels at those centres.
# KEYWORDS: fisheye, projection_model, translation, plane
def plane_displacement(cx, cy, a: float, b: float):
    xr, yr = cx - CENTRE, cy - CENTRE
    r = np.hypot(xr, yr)
    theta = r / F_PX_PER_RAD
    phi = np.arctan2(yr, xr)
    # Horizontal offset in units of the plane height (h = 1).
    xcap = np.tan(theta) * np.cos(phi)
    ycap = np.tan(theta) * np.sin(phi)
    xcap2, ycap2 = xcap + a, ycap + b
    rho2 = np.hypot(xcap2, ycap2)
    theta2 = np.arctan(rho2)
    phi2 = np.arctan2(ycap2, xcap2)
    return (F_PX_PER_RAD * (theta2 * np.cos(phi2) - theta * np.cos(phi)),
            F_PX_PER_RAD * (theta2 * np.sin(phi2) - theta * np.sin(phi)))


# PURPOSE: Fit the translating-plane model to a patch displacement field and
#          report the fit quality, which measures how rigid (single-layer) the
#          field actually is.
# INPUTS: patches (list of (cx, cy, dx, dy, score)); f (unused, kept for clarity).
# OUTPUTS: dict {a, b, speed_factor, rms_px, n}; speed_factor is the
#          dimensionless |translation|/height so that speed = height/dt * factor;
#          None when scipy is unavailable or the fit fails.
# KEYWORDS: least_squares, plane_fit, rigidity, residual
def fit_plane_translation(patches: list):
    from scipy.optimize import least_squares
    cx = np.array([p[0] for p in patches], dtype=float)
    cy = np.array([p[1] for p in patches], dtype=float)
    mx = np.array([p[2] for p in patches], dtype=float)
    my = np.array([p[3] for p in patches], dtype=float)

    def residual(p):
        dx, dy = plane_displacement(cx, cy, p[0], p[1])
        return np.concatenate([dx - mx, dy - my])

    try:
        # Huber loss down-weights the patches that match a different layer or a
        # poorly-textured region; the reported rms still uses every patch, so it
        # stays an honest measure of field rigidity.
        sol = least_squares(residual, x0=[0.0, 0.0], method="trf", loss="huber",
                            f_scale=2.0, max_nfev=400)
    except Exception:  # noqa: BLE001 - any solver failure falls back to median
        return None
    r = residual(sol.x)
    return {"a": float(sol.x[0]), "b": float(sol.x[1]),
            "speed_factor": float(np.hypot(sol.x[0], sol.x[1])),
            "rms_px": float(np.sqrt(np.mean(r ** 2))), "n": int(cx.size)}


# PURPOSE: Independent displacement estimate by phase correlation over the sky
#          disc, used only as a cross-check on the block-matching field.
# INPUTS: g0, g1 (2-D frames); when (datetime) for sun exclusion.
# OUTPUTS: dict {dx, dy, mag, response} or None when the crop is unusable; the
#          sun and everything outside the sky disc are replaced by the local
#          median before correlating.
# KEYWORDS: phase_correlation, cross_check, independent_estimator
def phase_displacement(g0, g1, when: dt.datetime):
    n = CROP_N
    half = CROP_HALF
    if g0.shape[0] != n or g0.shape[1] != n:
        return None
    a = bandpass(g0.astype(np.float32))
    b = bandpass(g1.astype(np.float32))
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - half, yy - half)
    m = (rr < SKY_R - 8).astype(np.float32)
    zen, az = sun_position(when)
    imaz = np.radians((az - AZIMUTH_OFFSET) % 360.0)
    sr = F_PX_PER_RAD * np.radians(zen)
    sx, sy = half + sr * np.cos(imaz), half + sr * np.sin(imaz)
    if np.hypot(sx - half, sy - half) < SKY_R + 150:
        m[np.hypot(xx - sx, yy - sy) < 250.0] = 0
    if m.sum() < 1000:
        return None
    # Smooth the mask into a window so its edge does not create correlation
    # structure, and mean-subtract so brightness drift does not dominate.
    win = cv2.GaussianBlur(m, (0, 0), 12)
    win = win / (win.max() + 1e-9)
    a = (a - float(np.median(a[m > 0.5]))) * win
    b = (b - float(np.median(b[m > 0.5]))) * win
    (dx, dy), resp = cv2.phaseCorrelate(a, b)
    return {"dx": float(dx), "dy": float(dy), "mag": float(np.hypot(dx, dy)),
            "response": float(resp)}


# PURPOSE: Compute the analysis: per-pair displacement, per-pair speed (using the
#          stereo height closest in time), and per-window aggregation on the
#          metric's window grid.
# INPUTS: none.
# OUTPUTS: dict {pairs, windows, config}; pair times are ISO UTC.
# KEYWORDS: pipeline, motion, aggregation
def compute():
    series = common.load_series(common.RUN_ROLL_CORRECTED)
    a_ts = series["a_ts"]  # ceilometer clock (MSK); +3 h from UTC
    h_all = series["filt_h"]
    names = sorted(glob.glob(os.path.join(common.IMAGES_DIR,
                                          "img-2025-05-23T*devID1.jpg")))
    stamps = [(parse_name(n), n) for n in names]
    stamps = [(t, n) for t, n in stamps
              if t is not None and T_START_UTC <= t <= T_END_UTC]
    stamps.sort(key=lambda x: x[0])
    pairs = []
    for (t0, p0), (t1, p1) in zip(stamps[:-1], stamps[1:]):
        dt_s = (t1 - t0).total_seconds()
        if not (0 < dt_s <= 20):
            continue
        centres = patch_centres(t0)
        if not centres:
            continue
        g0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        if g0 is None or g1 is None:
            continue
        g0 = g0[CROP_HALF: CROP_HALF + CROP_N, CROP_HALF: CROP_HALF + CROP_N]
        g1 = g1[CROP_HALF: CROP_HALF + CROP_N, CROP_HALF: CROP_HALF + CROP_N]
        d = frame_displacement(g0, g1, centres, t0)
        if d is None:
            continue
        # Height at the pair, on the ceilometer clock.
        h_ts = t0.timestamp() + 3 * 3600.0
        j = int(np.argmin(np.abs(a_ts - h_ts)))
        h = float(h_all[j])
        # Rigid single-plane fit: speed = h/dt * |(a,b)|; the fit rms measures
        # how well one translating layer explains the whole field.
        fit = fit_plane_translation(d["patches"])
        if fit is None:
            continue
        pc = phase_displacement(g0, g1, t0)
        speed = float(h * fit["speed_factor"] / dt_s)
        angle_img = float(np.degrees(np.arctan2(fit["b"], fit["a"])) % 360.0)
        bearing = float((angle_img - AZIMUTH_OFFSET) % 360.0)
        pairs.append({
            "t": t0.isoformat(), "dt_s": dt_s,
            "dx": d["dx"], "dy": d["dy"], "mag_px": d["mag"],
            "fit_a": fit["a"], "fit_b": fit["b"],
            "fit_rms_px": fit["rms_px"],
            "n_used": d["n_used"], "n_patches": d["n_patches"],
            "spread_px": d["spread_px"], "iqr_px": d["iqr_px"],
            "score_median": d["score_median"],
            "height_m": h,
            "speed_mps": speed,
            "pc_mag_px": None if pc is None else pc["mag"],
            "pc_speed_mps": (None if pc is None else
                             float(h * pc["mag"] / (F_PX_PER_RAD * dt_s))),
            "pc_response": None if pc is None else pc["response"],
            "img_angle_deg": angle_img, "bearing_deg_approx": bearing,
        })
        if len(pairs) % 200 == 0:
            print(f"  {len(pairs)} pairs through {t0.isoformat()}", flush=True)
    # Per-window aggregation on the metric's grid (ceilometer clock).
    windows = []
    for w in common.window_grid(a_ts, series["c_ts"], 30, 15):
        lo = w["start_s"] - 3 * 3600.0
        hi = lo + 30 * 60.0
        sel = [p for p in pairs if lo <= dt.datetime.fromisoformat(p["t"]).timestamp() < hi]
        if len(sel) < 10:
            continue
        sp = np.array([p["speed_mps"] for p in sel], dtype=float)
        ang = np.array([p["img_angle_deg"] for p in sel], dtype=float)
        windows.append({
            "start_utc": dt.datetime.fromtimestamp(lo).isoformat(),
            "start_ceil": w["start_dt"].isoformat(),
            "n_pairs": len(sel),
            "speed_mean": float(np.mean(sp)),
            "speed_median": float(np.median(sp)),
            "speed_std": float(np.std(sp)),
            "speed_iqr": float(np.percentile(sp, 75) - np.percentile(sp, 25)),
            "angle_mean_deg": float(np.degrees(np.angle(
                np.mean(np.exp(1j * np.radians(ang))))) % 360.0),
            "angle_circstd_deg": float(np.degrees(np.sqrt(
                -2 * np.log(np.abs(np.mean(np.exp(1j * np.radians(ang)))))))),
            "spread_px_mean": float(np.mean([p["spread_px"] for p in sel])),
            "height_mean": float(np.mean([p["height_m"] for p in sel])),
        })
    return {"pairs": pairs, "windows": windows,
            "config": {"f_px_per_rad": F_PX_PER_RAD, "sky_r_px": SKY_R,
                       "patch_px": PATCH, "search_px": SEARCH,
                       "bandpass_sigma": BANDPASS_SIGMA, "score_min": SCORE_MIN,
                       "t_start_utc": T_START_UTC.isoformat(),
                       "t_end_utc": T_END_UTC.isoformat()}}


# PURPOSE: Plot the measured cloud speed and direction over the session, and the
#          speed distribution, with the metric's window means overlaid.
# INPUTS: out (dict from compute); out_png (str).
# OUTPUTS: writes the PNG; returns None.
# KEYWORDS: figure, speed, direction, matplotlib
def make_figure(out: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs, windows = out["pairs"], out["windows"]
    t = np.array([dt.datetime.fromisoformat(p["t"]).timestamp() for p in pairs])
    sp = np.array([p["speed_mps"] for p in pairs])
    ang = np.array([p["img_angle_deg"] for p in pairs])
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    ax[0].plot(t - t[0], sp, ".", ms=3, alpha=0.6)
    wc = np.array([dt.datetime.fromisoformat(w["start_ceil"]).timestamp()
                   for w in windows])
    wm = np.array([w["speed_mean"] for w in windows])
    ax[0].plot(wc - t[0], wm, "r-o", ms=4, label="30-min window mean")
    ax[0].set_ylabel("cloud speed (m/s)")
    ax[0].set_xlabel("seconds since 09:00 UTC")
    ax[0].legend()
    ax[0].set_title("cloud speed from image block matching (height from stereo series)")
    ax[1].plot(t - t[0], ang, ".", ms=3, alpha=0.6)
    wa = np.array([w["angle_mean_deg"] for w in windows])
    ax[1].plot(wc - t[0], wa, "r-o", ms=4)
    ax[1].set_ylabel("image motion angle (deg, y-down)")
    ax[1].set_xlabel("seconds since 09:00 UTC")
    ax[1].set_title("image-plane direction of cloud motion")
    ax[2].hist(sp, bins=60)
    ax[2].set_xlabel("cloud speed (m/s)")
    ax[2].set_ylabel("frame pairs")
    ax[2].set_title(f"speed distribution (n={len(sp)} pairs)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"figure written: {out_png}")


# PURPOSE: Entry point: compute, save JSON, print summary, draw figure.
# INPUTS: none.
# OUTPUTS: q2_cloud_motion.json and q2_cloud_motion.png.
# KEYWORDS: main, entry_point
def main():
    out = compute()
    with open(os.path.join(HERE, "q2_cloud_motion.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    sp = np.array([p["speed_mps"] for p in out["pairs"]], dtype=float)
    ang = np.array([p["img_angle_deg"] for p in out["pairs"]], dtype=float)
    print(f"\npairs: {len(sp)}")
    print(f"speed m/s: median {np.median(sp):.2f} mean {sp.mean():.2f} "
          f"p10 {np.percentile(sp,10):.2f} p90 {np.percentile(sp,90):.2f} "
          f"std {sp.std():.2f}")
    print(f"image angle deg: circular mean "
          f"{np.degrees(np.angle(np.mean(np.exp(1j*np.radians(ang)))))%360:.1f}  "
          f"circular std "
          f"{np.degrees(np.sqrt(-2*np.log(np.abs(np.mean(np.exp(1j*np.radians(ang))))))):.1f}")
    print(f"per-pair patch scatter: median spread {np.median([p['spread_px'] for p in out['pairs']]):.2f} px, "
          f"median IQR {np.median([p['iqr_px'] for p in out['pairs']]):.2f} px")
    rms = np.array([p["fit_rms_px"] for p in out["pairs"]], dtype=float)
    print(f"single-plane fit rms px: median {np.median(rms):.2f}  "
          f"p90 {np.percentile(rms,90):.2f}  (field rigidity)")
    both = [p for p in out["pairs"]
            if p.get("pc_speed_mps") is not None and np.isfinite(p["speed_mps"])]
    if both:
        s1 = np.array([p["speed_mps"] for p in both])
        s2 = np.array([p["pc_speed_mps"] for p in both])
        resp = np.array([p["pc_response"] for p in both])
        hi = resp > 0.5
        print(f"cross-check vs phase correlation (all n={len(both)}): "
              f"Pearson r {np.corrcoef(s1, s2)[0,1]:.3f}")
        if hi.sum() > 3:
            print(f"cross-check, high phase response (>0.5, n={int(hi.sum())}): "
                  f"ratio block/phase median {np.median(s1[hi]/s2[hi]):.3f}  "
                  f"Pearson r {np.corrcoef(s1[hi], s2[hi])[0,1]:.3f}")
    for w in out["windows"]:
        print(f"  {w['start_ceil'][11:16]} MSK  n={w['n_pairs']:3d}  "
              f"speed {w['speed_mean']:5.2f}+-{w['speed_std']:4.2f} m/s  "
              f"angle {w['angle_mean_deg']:6.1f} (circstd {w['angle_circstd_deg']:4.1f})  "
              f"height {w['height_mean']:.0f} m")
    make_figure(out, os.path.join(HERE, "q2_cloud_motion.png"))


if __name__ == "__main__":
    main()
