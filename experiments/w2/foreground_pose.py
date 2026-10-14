# PURPOSE: Estimate the stereo relative pose (R, t) from the rigid static foreground and
#          test whether the static spatial pattern found in R-15/R-16 is a calibration
#          artefact that an independently estimated pose collapses.
#
#          WHY THE FOREGROUND. The clouds move and the pattern does not, so the clouds
#          cannot tell an instrument error from a real spatial structure in the sky.
#          The mast, the brick building and the roof panels are rigid, stationary and
#          cloud-independent: they are the only geometric control in the dataset that
#          the atmosphere cannot influence.
#
#          WHERE IT IS. The pipeline processes a 640x640 centre crop that is pure zenith
#          sky. The foreground lies at v > 1500 of the 1920x1920 frame, i.e. at zenith
#          angles of roughly 50-90 degrees. The control therefore runs on the FULL
#          frames, and the pose it yields is global while any residual principal-point or
#          distortion error is not. That extrapolation is stated as a limit in the
#          pre-registration and not hidden here.
#
#          THE MODEL. Camera 1 is brought into camera 2's frame by the shipped infinity
#          affine; both rays are then unit vectors about the common centre, built with
#          the measured cubic distortion. From ray correspondences the essential matrix
#          gives the residual relative rotation. If the affine rectifies correctly, that
#          rotation is the identity - so a significant departure from identity is itself
#          the measurement, and its size is the rectification error.
#
#          The primary metric is in PIXELS and needs no baseline: the magnitude of the
#          translation is unobservable from rays alone, and the structured disparity
#          residual is scale-invariant (pre-registration section 5.2).
# INPUTS: full-frame image pairs; experiments/w2/config.yaml.
# OUTPUTS: experiments/w2/results/<run>/foreground_pose.json and a printed diagnosis.
# KEYWORDS: foreground, essential_matrix, pose, calibration, R-15, R-16, control
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

IMAGES = "/home/iustimov/clouds_data/images"
import yaml                                                                   # noqa: E402
CFG = (yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
       if os.path.exists(os.path.join(HERE, "config.yaml")) else {}) or {}
F = float(CFG.get("camera", {}).get("f_sun", 654.0))
A_DIST = float(CFG.get("camera", {}).get("distortion_a", -0.30))
CENTRE = (960.0, 960.0)
NAME = re.compile(r"^img-(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})devID(\d)\.jpg$")


# PURPOSE: Build unit direction vectors for image points, with the measured cubic
#          radial distortion applied.
# INPUTS: uv (N,2) pixels; f (float) px/rad; centre (2,); a (float) cubic coefficient.
# OUTPUTS: rays (N,3), unit norm. A point at the principal point gives (0,0,1).
# KEYWORDS: ray_space, fisheye, distortion
def to_rays(uv, f=F, centre=CENTRE, a=A_DIST):
    d = np.asarray(uv, float) - np.asarray(centre, float)
    r = np.hypot(d[:, 0], d[:, 1])
    t = r / f
    theta = t * (1.0 + a * t * t)
    s = np.sin(theta); c = np.cos(theta)
    safe = np.where(r > 1e-9, r, 1.0)
    return np.stack([s * d[:, 0] / safe, s * d[:, 1] / safe, c], axis=1)


# PURPOSE: Mask the rigid foreground of one frame - the building, roof panels and mast.
# INPUTS: shape of the frame.
# OUTPUTS: boolean mask over the full 1920x1920 frame. The region is the lower band,
#          clipped to the lens circle so the black corners outside the field are
#          excluded; it deliberately omits the sky above and the trees at the sides,
#          which are not rigid.
# KEYWORDS: mask, foreground, rigid
def foreground_mask(shape):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - CENTRE[0], yy - CENTRE[1])
    return (yy >= float(CFG.get("foreground", {}).get("v_min", 1560))) & (r <= 950.0)


# PURPOSE: Estimate the relative pose from foreground correspondences of one pair.
# INPUTS: paths of the two frames.
# OUTPUTS: dict with R (3,3), the rotation angle from identity in degrees, the
#          translation direction, inlier count and epipolar RMS in pixels; or a dict
#          carrying `reason` when the gate is not met. Never raises on a bad frame.
# KEYWORDS: sift, essential_matrix, ransac, pose
def pose_from_frame(path1, path2):
    """Estimate the relative pose from the foreground counterpart of one pair.

    SIFT was tried first and rejected on measurement: the building sits at a few
    hundred metres from a 45 m baseline, so the same feature appears ~200 px apart in
    the two frames, and no descriptor matcher bridges that - the ratio test yielded 11
    correspondences at 0.75 against 327 at 0.90, i.e. almost all of them ambiguous.
    RoMa handles large displacements by construction and is the matcher already
    validated for this rig, so it is used here too.
    """
    from clouds.matching.matchers import create_matcher
    i1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if i1 is None or i2 is None:
        return {"reason": "unreadable"}
    cal = json.load(open(os.path.join(REPO, "configs", "calib",
                                      "2025-05-22_affine_old.json")))
    M = np.array(cal["matrix_2x3"], float)
    i1r = cv2.warpAffine(i1, M, dsize=(i1.shape[1], i1.shape[0]))
    v0 = int(CFG.get("foreground", {}).get("v_min", 1560))
    b1 = i1r[v0:, :]
    b2 = i2[v0:, :]
    matcher = create_matcher("roma", {"grid_stride": 6,
                                      "certainty_threshold": float(
                                          CFG.get("foreground", {})
                                          .get("certainty_threshold", 0.5))},
                             CFG.get("matcher", {}).get("device", "cuda"))
    k0, k1 = matcher(b1, b2)
    k0 = np.asarray(k0, float).reshape(-1, 2)
    k1 = np.asarray(k1, float).reshape(-1, 2)
    if len(k0) < 30:
        return {"reason": "too few foreground matches", "n_good": len(k0)}
    # back into full-frame coordinates
    p1 = k0 + np.array([0.0, float(v0)])
    p2 = k1 + np.array([0.0, float(v0)])
    # findEssentialMat takes 2-D points with a camera matrix, so a unit ray is passed
    # as its normalised projection (x/z, y/z) with K = I. Passing the 3-vector is an
    # error, not a convenience: the solver asserts on the point dimension.
    R1 = to_rays(p1); R2 = to_rays(p2)
    r1 = (R1[:, :2] / R1[:, 2:3]).astype(np.float32)   # (x/z, y/z), K = I
    r2 = (R2[:, :2] / R2[:, 2:3]).astype(np.float32)
    E, inl = cv2.findEssentialMat(r1, r2, cameraMatrix=np.eye(3),
                                 method=cv2.RANSAC, prob=0.999,
                                 threshold=float(CFG.get("foreground", {})
                                                 .get("ransac_threshold", 0.01)))
    if E is None or inl is None or int(inl.sum()) < 20:
        return {"reason": "essential matrix failed", "n_good": len(k0),
                "n_inliers": 0 if inl is None else int(inl.sum())}
    inl = inl.ravel().astype(bool)
    n_in, R, t, _ = cv2.recoverPose(E, r1[inl], r2[inl], cameraMatrix=np.eye(3))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))
    x1 = np.hstack([r1[inl], np.ones((int(inl.sum()), 1))])
    x2 = np.hstack([r2[inl], np.ones((int(inl.sum()), 1))])
    Ex1 = (E @ x1.T).T
    num = np.abs(np.sum(x2 * Ex1, axis=1))
    den = np.sqrt(Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Ex1[:, 2] ** 2)
    # the Sampson distance is in NORMALISED units; multiplying by f converts it to an
    # approximate pixel residual, which is what the pre-registered gate is stated in
    sampson = num / np.maximum(den, 1e-12)
    rms = float(np.sqrt(np.mean(sampson ** 2))) * F
    return {"R": R.tolist(), "t": (t.ravel() / max(np.linalg.norm(t), 1e-12)).tolist(),
            "rotation_from_identity_deg": ang, "n_good": len(k0),
            "n_inliers": int(n_in), "epipolar_rms_norm": rms,
            "epipolar_rms_deg": math.degrees(rms)}


def frames_for(date, limit, hour_lo=8, hour_hi=17):
    out = []
    for f in sorted(glob.glob(os.path.join(IMAGES, f"img-{date}T*devID1.jpg"))):
        m = NAME.match(os.path.basename(f))
        if not m:
            continue
        hh = int(m.group(2))
        if not (hour_lo <= hh <= hour_hi):
            continue
        p2 = f.replace("devID1.jpg", "devID2.jpg")
        if os.path.exists(p2):
            out.append((f, p2))
        if limit and len(out) >= limit:
            break
    return out


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"f = {F} px/rad, distortion a = {A_DIST}, centre = {CENTRE}")
    print("foreground = the lower band of the full frame (the crop is pure sky)\n")
    out = {}
    for date in ("2025-05-22", "2025-05-23"):
        rows = []
        for p1, p2 in frames_for(date, limit):
            r = pose_from_frame(p1, p2)
            r["frame"] = os.path.basename(p1)
            rows.append(r)
        ok = [r for r in rows if "reason" not in r]
        print(f"{date}: {len(rows)} pairs, {len(ok)} produced a pose")
        if ok:
            angs = np.array([r["rotation_from_identity_deg"] for r in ok])
            inl = np.array([r["n_inliers"] for r in ok])
            rms = np.array([r["epipolar_rms_norm"] for r in ok])
            tvec = np.array([r["t"] for r in ok])
            print(f"  inliers      median {np.median(inl):.0f} "
                  f"(min {inl.min()})")
            print(f"  epipolar rms median {np.median(rms):.5f} (sin) = "
                  f"{math.degrees(np.median(rms)):.3f} deg")
            print(f"  rotation from IDENTITY: median {np.median(angs):.3f} deg, "
                  f"p10-p90 {np.percentile(angs,10):.3f}-{np.percentile(angs,90):.3f} deg")
            tm = np.median(tvec, axis=0)
            print(f"  median translation direction (unit) "
                  f"[{tm[0]:+.4f}, {tm[1]:+.4f}, {tm[2]:+.4f}]")
            print(f"    -> the epipole is {math.degrees(math.atan2(tm[1], tm[0])):.1f} deg "
                  f"from the +u axis in the image plane")
            out[date] = {"n_ok": len(ok), "n_pairs": len(rows),
                         "median_inliers": float(np.median(inl)),
                         "median_rotation_from_identity_deg": float(np.median(angs)),
                         "median_epipolar_rms_sin": float(np.median(rms)),
                         "median_t": tm.tolist(),
                         "reasons": {r["reason"]: sum(1 for x in rows
                                                      if x.get("reason") == r["reason"])
                                     for r in rows if "reason" in r}}
        else:
            reasons = {}
            for r in rows:
                reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
            print(f"  no poses: {reasons}")
            out[date] = {"n_ok": 0, "n_pairs": len(rows), "reasons": reasons}
        print()
    # drift between the two sessions
    if out.get("2025-05-22", {}).get("n_ok") and out.get("2025-05-23", {}).get("n_ok"):
        a = out["2025-05-22"]["median_t"]; b = out["2025-05-23"]["median_t"]
        ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(a, b))))))
        print(f"translation-direction drift between sessions: {ang:.3f} deg")
        out["translation_drift_deg"] = ang
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    dst = os.path.join(HERE, "foreground_pose.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
