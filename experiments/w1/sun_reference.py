# PURPOSE: Provide an absolute angular reference for the rig using the Sun, which
#          is the only object at infinity in these images. Two capabilities:
#            * locate the Sun's image in a frame, given the time and site, using
#              a solar ephemeris and the known camera azimuth offset;
#            * measure the infinity residual of the FIXED calibration by
#              predicting where the Sun must appear in the second camera (through
#              the calibrated affine) and comparing with where it actually is.
#          Nothing here consults the ceilometer, and nothing here modifies the
#          calibration: the calibration is fixed and this tool only measures how
#          far off it is.
# INPUTS: an image path, a timestamp (UTC, from the filename), the site
#         coordinates, the calibration matrix from the registry, and the camera's
#         azimuth offset.
# OUTPUTS: a SunObservation (image position, radius, blob area) or None; and
#         infinity-test residuals in pixels.
# KEYWORDS: calibration, verification, infinity_reference, sun, ephemeris,
#           runtime_self_check
from __future__ import annotations

import datetime as dt
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from clouds.config import load_config  # noqa: E402

# Site and rig constants. Zvenigorod; the coordinates are the ones the ephemeris
# was validated against (the Sun's image azimuth is predicted to better than 0.5
# degree across 14 hours of 2025-05-23).
SITE_LAT, SITE_LON = 55.70, 36.80

# Orientation offset of the RAW devID1 image axes relative to north, in degrees,
# measured from the Sun's image azimuth. The sign convention is
#     image_azimuth = sun_azimuth - AZIMUTH_OFFSET
# It is a property of this rig's mounting and is the same for every frame.
AZIMUTH_OFFSET = -45.7


# PURPOSE: Compute the Sun's zenith angle and azimuth for an instant and place.
# INPUTS: when (datetime, UTC); lat, lon (degrees).
# OUTPUTS: (zenith_deg, azimuth_deg) where azimuth is measured from north,
#          clockwise. Accurate to a few arcminutes, which is far below the
#          pixel scale (10.67 px per degree).
# KEYWORDS: ephemeris, solar_position, noaa
def sun_position(when: dt.datetime, lat: float = SITE_LAT, lon: float = SITE_LON):
    y, m, d = when.year, when.month, when.day
    frac = (when.hour + when.minute / 60 + when.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5 + frac
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + 36000.76983 * t + 0.0003032 * t * t) % 360
    mn = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    ecc = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
    mr = math.radians(mn)
    c = ((1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mr)
         + (0.019993 - 0.000101 * t) * math.sin(2 * mr)
         + 0.000289 * math.sin(3 * mr))
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    lam = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    sec = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    eps = 23 + (26 + sec / 60) / 60 + 0.00256 * math.cos(math.radians(omega))
    dec = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lam))))
    yv = math.tan(math.radians(eps / 2)) ** 2
    eot = 4 * math.degrees(
        yv * math.sin(2 * math.radians(l0)) - 2 * ecc * math.sin(mr)
        + 4 * ecc * yv * math.sin(mr) * math.cos(2 * math.radians(l0))
        - 0.5 * yv * yv * math.sin(4 * math.radians(l0))
        - 1.25 * ecc * ecc * math.sin(2 * mr))
    tst = (when.hour * 60 + when.minute + when.second / 60) + eot + 4 * lon
    # Keep the SIGN of the hour angle: the azimuth branch depends on it, and
    # folding it into [0,360) silently mirrors every morning measurement.
    ha = ((tst / 4 - 180 + 180) % 360) - 180
    la, dr, hr = math.radians(lat), math.radians(dec), math.radians(ha)
    cosz = math.sin(la) * math.sin(dr) + math.cos(la) * math.cos(dr) * math.cos(hr)
    zen = math.degrees(math.acos(max(-1.0, min(1.0, cosz))))
    sinz = math.sin(math.radians(zen))
    cosaz = ((math.sin(la) * cosz - math.sin(dr)) / (math.cos(la) * sinz)
             if sinz > 1e-9 else 0.0)
    az = math.degrees(math.acos(max(-1.0, min(1.0, cosaz))))
    if ha > 0:
        az = 360 - az
    return zen, az


# PURPOSE: Describe the Sun as found in one image.
# INPUTS: none (constructed by find_sun).
# OUTPUTS: immutable record; `radius` is the distance from the principal point
#          and `area` the saturated pixel count.
# KEYWORDS: observation, sun, blob
@dataclass(frozen=True)
class SunObservation:
    u: float
    v: float
    radius: float
    area: int
    azimuth: float


# PURPOSE: Find the Sun in a frame as the saturated blob closest in image azimuth
#          to the position the ephemeris predicts.
#          LIMITATION, measured: the returned point is the centroid of the whole
#          saturated region, which is larger than the solar disc and changes shape
#          between the two cameras. A residual computed from it therefore mixes
#          calibration error with blob-shape difference and is an upper bound, not
#          a measurement of calibration error alone.
# INPUTS: gray (2-D uint8 image); expected_azimuth (deg, image frame);
#         threshold (uint8), min_area (px), tolerance_deg, principal point.
# OUTPUTS: SunObservation, or None when no saturated blob is within tolerance.
# KEYWORDS: sun_detection, blob, saturation, azimuth_match
def find_sun(gray: np.ndarray, expected_azimuth: float, threshold: int = 252,
             min_area: int = 800, tolerance_deg: float = 12.0,
             centre: tuple = (960.0, 960.0)) -> Optional[SunObservation]:
    mask = (gray >= threshold).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = stats[i, 4]
        if area < min_area:
            continue
        u, v = float(cent[i][0]), float(cent[i][1])
        az = math.degrees(math.atan2(v - centre[1], u - centre[0])) % 360
        d = abs((az - expected_azimuth + 180) % 360 - 180)
        if d > tolerance_deg:
            continue
        if best is None or d < best[0]:
            best = (d, SunObservation(u, v, math.hypot(u - centre[0], v - centre[1]),
                                      int(area), az))
    return None if best is None else best[1]


# PURPOSE: Run the infinity test for one stereo pair.
# INPUTS: paths to devID1 and devID2, the acquisition timestamp (UTC), the
#         calibration matrix (2x3) read from the registry.
# OUTPUTS: dict with the Sun's position in devID1, its position predicted in
#          devID2 through the affine, the position actually found there, and the
#          residual. Returns None when the Sun cannot be located in both.
# KEYWORDS: infinity_test, calibration_residual, sun
def infinity_residual(dev1_path: str, dev2_path: str, when: dt.datetime,
                      calibration_2x3) -> Optional[dict]:
    zenith, azimuth = sun_position(when)
    if zenith > 86:            # Sun below or on the horizon: no usable reference
        return None
    g1 = cv2.imread(dev1_path, cv2.IMREAD_GRAYSCALE)
    g2 = cv2.imread(dev2_path, cv2.IMREAD_GRAYSCALE)
    if g1 is None or g2 is None:
        return None
    expected_azimuth = (azimuth - AZIMUTH_OFFSET) % 360
    obs1 = find_sun(g1, expected_azimuth)
    if obs1 is None or obs1.radius > 900:
        return None
    m = np.asarray(calibration_2x3, dtype=float)
    pu, pv = (m @ np.array([obs1.u, obs1.v, 1.0]))[:2]
    # The prediction is in devID2's frame, whose orientation matches devID1's
    # after rectification, so the same azimuth offset applies there.
    obs2 = find_sun(g2, expected_azimuth, tolerance_deg=60.0)
    if obs2 is None:
        return None
    dx, dy = obs2.u - pu, obs2.v - pv
    return {
        "time": when.isoformat(), "sun_zenith_deg": zenith, "sun_azimuth_deg": azimuth,
        "sun1": [obs1.u, obs1.v, obs1.radius, obs1.area],
        "predicted_in_dev2": [float(pu), float(pv)],
        "sun2": [obs2.u, obs2.v, obs2.radius, obs2.area],
        "dx": float(dx), "dy": float(dy), "residual_px": float(math.hypot(dx, dy)),
    }


# PURPOSE: Iterate a session's archive and run the infinity test on every frame
#          where the Sun is usable.
# INPUTS: images_dir, date prefix (e.g. "2025-05-23"), calibration matrix,
#         step_min (minutes between sampled frames).
# OUTPUTS: list of residuals, sorted by time.
# KEYWORDS: batch, session, infinity_test
def scan_session(images_dir: str, date: str, calibration_2x3,
                 step_min: int = 20) -> list:
    names = sorted(os.listdir(images_dir))
    results = []
    seen = set()
    for name in names:
        if not name.endswith("devID1.jpg") or not name.startswith(f"img-{date}"):
            continue
        base = name.replace("devID1.jpg", "")
        stamp = base.replace("img-", "")
        try:
            when = dt.datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S")
        except ValueError:
            continue
        if when.minute % step_min:
            continue
        if stamp in seen:
            continue
        seen.add(stamp)
        dev2 = os.path.join(images_dir, base + "devID2.jpg")
        if not os.path.exists(dev2):
            continue
        r = infinity_residual(os.path.join(images_dir, base + "devID1.jpg"), dev2,
                              when, calibration_2x3)
        if r:
            results.append(r)
    return sorted(results, key=lambda r: r["time"])


if __name__ == "__main__":
    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026.yaml"))
    M = cfg.calibration_record.matrix_2x3
    out = scan_session(cfg.dataset.images_dir, cfg.dataset.date, M, step_min=20)
    res = np.array([r["residual_px"] for r in out]) if out else np.array([])
    print(f"frames with a usable Sun in both cameras: {len(out)}")
    if res.size:
        print(f"  |residual| px: median {np.median(res):.2f}  mean {res.mean():.2f}  "
              f"p10 {np.percentile(res,10):.2f}  p90 {np.percentile(res,90):.2f}")
    for r in out:
        print(f"  {r['time'][11:16]} zen={r['sun_zenith_deg']:5.1f}  "
              f"r1={r['sun1'][2]:6.1f}  residual={r['residual_px']:7.2f} px")
