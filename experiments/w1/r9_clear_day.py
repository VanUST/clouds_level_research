# PURPOSE: Test the owner's R9 hypothesis — that the cloudless day gives a usable Sun
#          reference even though the cloudy day did not.
#
#          The earlier attempt to use the Sun as an infinity reference failed on
#          2025-05-23 because the detected "Sun" is the whole saturated region around
#          it, whose centroid is dragged by whatever bright cloud sits alongside. The
#          measured bias was large enough to leave the method at ~4 deg resolution
#          against a 1 px target, so it was abandoned.
#
#          2025-05-22 is cloudless. No cloud glare means the saturated region should be
#          the Sun plus its atmospheric glow alone, and therefore far more symmetric.
#          That session was dismissed for stereo ("no cloud signal") — correct for
#          stereo, and wrong for calibration.
#
#          The test is comparative and needs no ground truth: the Sun's direction is
#          known from an ephemeris to arcseconds, so the scatter of the inferred image
#          position around the ephemeris-predicted track measures the method's
#          precision directly. If the clear day is materially tighter, R9 succeeds.
# INPUTS: both sessions' images, the solar ephemeris in sun_reference.py.
# OUTPUTS: experiments/w1/r9_clear_day.json and a printed comparison.
# KEYWORDS: R9, calibration, infinity_reference, sun, clear_day, precision
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from sun_reference import sun_position                                      # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
OUT = os.path.join(HERE, "r9_clear_day.json")
CENTRE = 960.0
F0 = 1920.0 / math.pi


# PURPOSE: Locate the Sun's saturated region in one frame.
# INPUTS: grayscale image, minimum blob area, threshold.
# OUTPUTS: list of (area, u, v) sorted by area descending.
# KEYWORDS: sun_detection, blob
def blobs(gray, min_area=200, threshold=252):
    mask = (gray >= threshold).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, 4])
        if a >= min_area:
            out.append((a, float(cent[i][0]), float(cent[i][1])))
    return sorted(out, reverse=True)


# PURPOSE: For one session, find the Sun in the frames where it is visible and measure
#          how well its image position follows the ephemeris.
# INPUTS: date string, hour range (UTC), frame stride.
# OUTPUTS: dict with the azimuth-offset estimate, the azimuth residual scatter, and the
#          radial fit of f, plus the blob-area statistics that motivate the comparison.
# KEYWORDS: session, ephemeris, residual, precision
def measure(date, hours=range(4, 18), step_min=20):
    # first pass: vote for the orientation offset from every large blob
    votes, areas, frames = [], [], []
    for hh in hours:
        for mm in range(0, 60, step_min):
            key = f"{date}T{hh:02d}-{mm:02d}"
            cands = [x for x in os.listdir(IMAGES)
                     if x.startswith("img-" + key) and x.endswith("devID1.jpg")]
            if not cands:
                continue
            nm = cands[0]
            stamp = nm.replace("img-", "").replace("devID1.jpg", "")
            when = dt.datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S")
            zen, az = sun_position(when)
            if zen > 88:
                continue
            g = cv2.imread(os.path.join(IMAGES, nm), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            bl = blobs(g)
            if not bl:
                continue
            for a, u, v in bl[:2]:
                ph = math.degrees(math.atan2(v - CENTRE, u - CENTRE)) % 360
                votes.append((az - ph) % 360)
                areas.append(a)
                frames.append((when, zen, az, a, u, v))
    if not votes:
        return None
    votes = np.array(votes)
    # the offset that most blobs agree on
    hist, edges = np.histogram(votes, bins=72, range=(0, 360))
    k = int(hist.argmax())
    a0 = 0.5 * (edges[k] + edges[k + 1])
    d = ((votes - a0 + 180) % 360) - 180
    sel = votes[np.abs(d) < 15]
    ang = np.radians(sel)
    a0 = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360

    # second pass: keep only the blob nearest the predicted azimuth, measure residuals
    az_res, radii, thetas, used_area = [], [], [], []
    for when, zen, az, a, u, v in frames:
        ph = math.degrees(math.atan2(v - CENTRE, u - CENTRE)) % 360
        pred = (az - a0) % 360
        r = abs((ph - pred + 180) % 360 - 180)
        if r <= 8:
            az_res.append(r)
            radii.append(math.hypot(u - CENTRE, v - CENTRE))
            thetas.append(math.radians(zen))
            used_area.append(a)
    if len(radii) < 5:
        return None
    th = np.array(thetas); rr = np.array(radii)
    # radial fit: r = f*theta + r0, separating f from a constant centroid bias
    A = np.vstack([th, np.ones_like(th)]).T
    sol, _res, _rank, _sv = np.linalg.lstsq(A, rr, rcond=None)
    rms = float(np.sqrt(np.mean((rr - A @ sol) ** 2)))
    return {
        "date": date, "n_frames_used": len(radii),
        "median_blob_area_all": float(np.median(areas)) if areas else None,
        "median_blob_area_used": float(np.median(used_area)) if used_area else None,
        "azimuth_residual_median_deg": float(np.median(az_res)),
        "azimuth_residual_p90_deg": float(np.percentile(az_res, 90)),
        "f_bias_px_per_rad": float(sol[0]), "r0_px": float(sol[1]),
        "radial_rms_px": rms, "f_through_origin": float(th @ rr / (th @ th)),
        "A0_deg": float(a0),
    }


def main():
    print("Is the cloudless day a usable Sun/infinity reference?\n")
    out = {}
    for date in ("2025-05-22", "2025-05-23"):
        r = measure(date)
        out[date] = r
        if r is None:
            print(f"{date}: no usable Sun detections")
            continue
        print(f"=== {date} ===")
        print(f"  frames used                 {r['n_frames_used']}")
        print(f"  median blob area (all/used) {r['median_blob_area_all']:.0f} / "
              f"{r['median_blob_area_used']:.0f} px")
        print(f"  azimuth residual  median {r['azimuth_residual_median_deg']:.2f} deg, "
              f"p90 {r['azimuth_residual_p90_deg']:.2f} deg")
        print(f"  radial fit  f = {r['f_bias_px_per_rad']:.1f} px/rad, "
              f"r0 = {r['r0_px']:.1f} px, rms {r['radial_rms_px']:.1f} px")
        print(f"  through-origin f = {r['f_through_origin']:.1f} px/rad "
              f"(assumed xi0/pi = {F0:.1f})")
        print()

    a, b = out.get("2025-05-22"), out.get("2025-05-23")
    if a and b:
        ra, rb = a["radial_rms_px"], b["radial_rms_px"]
        aa, ab = a["median_blob_area_used"], b["median_blob_area_used"]
        print("comparison")
        print(f"  blob area   clear {aa:.0f} px  vs cloudy {ab:.0f} px  "
              f"(ratio {aa/max(ab,1):.2f})")
        print(f"  radial rms  clear {ra:.1f} px  vs cloudy {rb:.1f} px  "
              f"(ratio {ra/max(rb,1e-9):.2f})")
        print(f"  azimuth p90 clear {a['azimuth_residual_p90_deg']:.2f} deg vs "
              f"cloudy {b['azimuth_residual_p90_deg']:.2f} deg")
        better = ra < rb
        print(f"\n  R9 hypothesis {'SUPPORTED' if better else 'NOT supported'}: the clear")
        print(f"  day's Sun position is {'tighter' if better else 'not tighter'} than the "
              f"cloudy day's.")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
