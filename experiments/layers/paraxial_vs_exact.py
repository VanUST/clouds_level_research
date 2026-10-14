# PURPOSE: Reconcile two measurements that disagree. Using the SAME matches, the
#          paraxial formula D = f*B/disp_x gives an almost flat height-versus-radius
#          profile (7 % spread), while the exact unit-sphere triangulation gives a
#          strongly falling one (26 %). Both cannot be describing the same geometry,
#          so one of them is wrong or they are sensitive to different parts of the
#          data.
#
#          The likely culprit is documented here rather than guessed at: the
#          calibration is a 2-D AFFINE warp. A 2-D warp can correct a rotation about
#          the optical axis and a translation, but it CANNOT correct a relative TILT
#          between the two cameras. After such a warp the pair is therefore not truly
#          rectified away from the principal point, the epipolar lines are not
#          horizontal, and the disparity acquires a vertical component that the
#          paraxial formula discards and the exact triangulation does not.
#
#          This script measures that directly: the disparity components and both
#          height estimates, all as functions of position.
# INPUTS: experiments/layers/heights.npz.
# OUTPUTS: docs/report/paraxial_vs_exact.png, printed profiles.
# KEYWORDS: rectification, affine, tilt, paraxial, triangulation, correctness
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.geometry.triangulation import heights_from_matches               # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    f = float(d["f"]); B = float(d["baseline"]); cx, cy = d["centre"]
    print(f"f = {f:.1f} px/rad, B = {B} m, centre = ({cx}, {cy})\n")

    rows = []
    for a, b in zip(K1, K2):
        if len(a) == 0:
            continue
        du = (a[:, 0] - b[:, 0]).astype(np.float64)
        dv = (a[:, 1] - b[:, 1]).astype(np.float64)
        r = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
        pa = f * B / np.abs(du)
        ex = heights_from_matches(a, b, f, B, centre=(cx, cy))["heights"]
        keep = np.isfinite(ex) & np.isfinite(pa) & (ex > 50) & (ex < 20000) & (np.abs(du) > 0.5)
        rows.append((r[keep], du[keep], dv[keep], pa[keep], ex[keep], a[keep]))
    r = np.concatenate([x[0] for x in rows])
    du = np.concatenate([x[1] for x in rows])
    dv = np.concatenate([x[2] for x in rows])
    pa = np.concatenate([x[3] for x in rows])
    ex = np.concatenate([x[4] for x in rows])
    kp1 = np.concatenate([x[5] for x in rows])
    print(f"matches {len(r)}")
    print(f"overall: |du| median {np.median(np.abs(du)):.2f}, "
          f"|dv| median {np.median(np.abs(dv)):.2f}, "
          f"paraxial median {np.median(pa):.0f} m, exact median {np.median(ex):.0f} m")
    print(f"vertical/horizontal disparity ratio: median "
          f"{np.median(np.abs(dv)/np.abs(du)):.3f}\n")

    print(f"{'radius':>10} {'|du|':>7} {'|dv|':>7} {'tan(tilt)':>10} "
          f"{'paraxial':>9} {'exact':>8} {'ratio':>7}")
    prof = []
    for lo in np.arange(0, 425, 25):
        m = (r >= lo) & (r < lo + 25)
        if m.sum() < 100:
            continue
        prof.append((lo + 12.5, float(np.median(np.abs(du[m]))),
                     float(np.median(np.abs(dv[m]))),
                     float(np.median(pa[m])), float(np.median(ex[m]))))
        p = prof[-1]
        print(f"{p[0]:>10.0f} {p[1]:>7.2f} {p[2]:>7.2f} {p[2]/p[1]:>10.3f} "
              f"{p[3]:>9.0f} {p[4]:>8.0f} {p[3]/p[4]:>7.3f}")

    x = np.array([p[0] for p in prof])
    par = np.array([p[3] for p in prof])
    exc = np.array([p[4] for p in prof])
    print(f"\nprofile spread (max-min)/mean:")
    print(f"  paraxial: {100*(par.max()-par.min())/par.mean():.1f} %")
    print(f"  exact   : {100*(exc.max()-exc.min())/exc.mean():.1f} %")
    print(f"  paraxial/exact ratio: {par[0]/exc[0]:.3f} at r={x[0]:.0f} "
          f"-> {par[-1]/exc[-1]:.3f} at r={x[-1]:.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.5))
    ax1.plot(x, [p[1] for p in prof], "o-", color="#1f6feb", lw=1.7, label="|du| horizontal")
    ax1.plot(x, [p[2] for p in prof], "s-", color="#c1440e", lw=1.7, label="|dv| vertical")
    ax1.set_xlabel("radius from principal point (px)")
    ax1.set_ylabel("median |disparity component| (px)")
    ax1.set_title("A vertical disparity that grows with radius")
    ax1.legend(fontsize=8)
    ax2.plot(x, par, "o-", color="#c1440e", lw=1.8, label="paraxial f·B/|du|")
    ax2.plot(x, exc, "s-", color="#1a7f37", lw=1.8, label="exact triangulation")
    ax2.set_xlabel("radius from principal point (px)")
    ax2.set_ylabel("median height (m)")
    ax2.set_title("The two estimators disagree off-axis")
    ax2.legend(fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "paraxial_vs_exact.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/paraxial_vs_exact.png")


if __name__ == "__main__":
    main()
