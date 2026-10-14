# PURPOSE: Estimate the projection constant f from the Sun independently in each
#          camera and each session, and measure how well f is actually
#          identifiable. f = px/rad is the one geometry constant that scales
#          every height (D = f*B/disparity), so its uncertainty is the dominant
#          term in the absolute error budget.
#
#          Method: the Sun's image radius is r = f*theta + bias(theta). Two
#          estimators are compared on the same data:
#            * free-intercept fit   r = f*theta + r0   (f and a constant radial
#              bias of the blob centroid are separated)
#            * through-origin fit   r = f*theta        (assumes the centroid is
#              unbiased)
#          They disagree systematically, which is the point of the script: the
#          spread ACROSS cameras and sessions is an upper bound on how well f is
#          known, and the disagreement WITHIN one stream shows the degeneracy.
#
#          The Sun's image azimuth needs a per-stream orientation offset, which
#          is estimated here by voting (A0 = ephemeris_azimuth - image_azimuth)
#          and refined by a circular mean. This is also an independent check of
#          the timebase: a wrong clock cannot produce a consistent A0.
# INPUTS: images under ~/clouds_data/images for the two 2025 sessions.
# OUTPUTS: experiments/w1/f_estimates.json and a table on stdout.
# KEYWORDS: calibration, focal_constant, identifiability, sun, ephemeris
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from sun_reference import sun_position  # noqa: E402

IMAGES = "/home/iustimov/clouds_data/images"
XI0 = 1920.0
F_ASSUMED = XI0 / math.pi          # 611.15 px/rad, the value the pipeline assumes
CENTRE = 960.0


# PURPOSE: Collect saturated blobs larger than a threshold in one frame.
# INPUTS: gray (2-D uint8), min_area (px), max_radius (px).
# OUTPUTS: list of (area, u, v, radius, image_azimuth_deg), largest first.
# KEYWORDS: blob, saturation, detection
def blobs(gray, min_area=800, max_radius=900.0):
    mask = (gray >= 252).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        area = stats[i, 4]
        if area < min_area:
            continue
        u, v = float(cent[i][0]), float(cent[i][1])
        r = math.hypot(u - CENTRE, v - CENTRE)
        if r > max_radius:
            continue
        az = math.degrees(math.atan2(v - CENTRE, u - CENTRE)) % 360
        out.append((int(area), u, v, r, az))
    return sorted(out, reverse=True)


# PURPOSE: Sample one camera stream over a day and attach the solar ephemeris.
# INPUTS: date ("YYYY-MM-DD"), dev (1 or 2), step_min.
# OUTPUTS: list of (datetime, zenith_deg, azimuth_deg, blobs).
# KEYWORDS: sampling, session, ephemeris
def stream(date, dev, step_min=20):
    frames = []
    for hh in range(4, 20):
        for mm in range(0, 60, step_min):
            key = f"{date}T{hh:02d}-{mm:02d}"
            cands = [x for x in os.listdir(IMAGES)
                     if x.startswith(f"img-{key}") and x.endswith(f"devID{dev}.jpg")]
            if not cands:
                continue
            name = cands[0]
            stamp = name.replace("img-", "").replace(f"devID{dev}.jpg", "")
            when = dt.datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S")
            zen, az = sun_position(when)
            if zen > 88:
                continue
            g = cv2.imread(os.path.join(IMAGES, name), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            frames.append((when, zen, az, blobs(g)))
    return frames


# PURPOSE: Estimate the orientation offset A0 and f for one camera stream.
# INPUTS: date, dev.
# OUTPUTS: dict, or None when too few Sun detections.
# KEYWORDS: orientation, focal_constant, regression
def estimate(date, dev):
    frames = stream(date, dev)
    votes = []
    for _w, _z, az, bl in frames:
        for _a, _u, _v, _r, ph in bl[:3]:
            votes.append((az - ph) % 360)
    if not votes:
        return None
    votes = np.array(votes)
    hist, edges = np.histogram(votes, bins=72, range=(0, 360))
    k = int(hist.argmax())
    a0 = 0.5 * (edges[k] + edges[k + 1])
    d = ((votes - a0 + 180) % 360) - 180
    sel = votes[np.abs(d) < 15]
    ang = np.radians(sel)
    a0 = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360

    thetas, radii, azres = [], [], []
    for _w, zen, az, bl in frames:
        if not bl:
            continue
        pred = (az - a0) % 360
        cand = min(bl, key=lambda t: abs((t[4] - pred + 180) % 360 - 180))
        dd = abs((cand[4] - pred + 180) % 360 - 180)
        if dd <= 10:
            thetas.append(math.radians(zen))
            radii.append(cand[3])
            azres.append(dd)
    if len(thetas) < 6:
        return None

    th = np.array(thetas)
    r = np.array(radii)
    A = np.vstack([th, np.ones_like(th)]).T
    sol, _res, _rank, _sv = np.linalg.lstsq(A, r, rcond=None)
    rms = float(np.sqrt(np.mean((r - A @ sol) ** 2)))
    f_thru = float(th @ r / (th @ th))
    return {
        "date": date, "dev": dev, "n": len(th), "A0_deg": float(a0),
        "f_bias": float(sol[0]), "r0_px": float(sol[1]), "rms_px": rms,
        "f_through_origin": f_thru,
        "azimuth_residual_deg": float(np.mean(azres)),
        "theta_range_rad": [float(th.min()), float(th.max())],
    }


# PURPOSE: CLI entry point: run every stream and report identifiability.
# INPUTS: none.
# OUTPUTS: table on stdout, experiments/w1/f_estimates.json.
# KEYWORDS: report, identifiability
def main():
    print(f"assumed f = xi0/pi = {F_ASSUMED:.1f} px/rad\n")
    print(f"{'stream':>16} {'n':>3} {'A0(deg)':>8} {'f_bias':>7} {'r0(px)':>7} "
          f"{'rms(px)':>7} {'f_r0=0':>7} {'bias/xi':>8} {'thru/xi':>8} {'az_res':>7}")
    rows = []
    for date in ("2025-05-22", "2025-05-23"):
        for dev in (1, 2):
            res = estimate(date, dev)
            if res is None:
                print(f"{date + ' dev' + str(dev):>16}   insufficient detections")
                continue
            rows.append(res)
            print(f"{date + ' dev' + str(dev):>16} {res['n']:>3} {res['A0_deg']:>8.2f} "
                  f"{res['f_bias']:>7.1f} {res['r0_px']:>7.1f} {res['rms_px']:>7.1f} "
                  f"{res['f_through_origin']:>7.1f} {res['f_bias']/F_ASSUMED:>8.3f} "
                  f"{res['f_through_origin']/F_ASSUMED:>8.3f} "
                  f"{res['azimuth_residual_deg']:>7.2f}")
    if rows:
        fb = np.array([r["f_bias"] for r in rows])
        ft = np.array([r["f_through_origin"] for r in rows])
        print(f"\n f (free intercept) : {fb.min():.0f} .. {fb.max():.0f} px/rad")
        print(f" f (r0 forced to 0) : {ft.min():.0f} .. {ft.max():.0f} px/rad")
        print(f" the two estimators differ by a factor {ft.mean()/fb.mean():.3f}: the")
        print(f" degeneracy between f and a radial blob bias is systematic, not noise.")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f_estimates.json")
        json.dump(rows, open(out, "w"), indent=1)
        print(f" saved {out}")


if __name__ == "__main__":
    main()
