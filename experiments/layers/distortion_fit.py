# PURPOSE: Test the hypothesis that the radial trend in apparent height is fisheye
#          LENS DISTORTION, as opposed to one of the four ideal projection laws.
#
#          The earlier test compared equidistant, equisolid, stereographic and
#          orthographic and found that each leaves 20-26 % of the trend. Those are
#          four specific closed forms, not distortion: a real lens obeys a smooth
#          radial deviation from any ideal law, which is what the Kannala-Brandt
#          polynomial r(theta) = k1*theta + k2*theta^3 + k3*theta^5 + ... describes.
#          The leading deviation from equidistant is the cubic term, so this script
#          fits  theta = (r/f) * (1 + a * (r/f)^2)  and asks whether ANY value of a
#          flattens the height-versus-radius profile.
#
#          If some a does flatten it, the lens distortion was the cause and its
#          coefficient is measured. If none does, the trend is not radial distortion
#          and the search must move elsewhere. Either answer is a result.
# INPUTS: experiments/layers/heights.npz.
# OUTPUTS: docs/report/distortion_fit.png, distortion_fit.json, printed grid.
# KEYWORDS: distortion, kannala_brandt, fisheye, radial_trend, falsification
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})


# PURPOSE: Height profile after applying a cubic distortion to the angle mapping.
# INPUTS: kp1, kp2 lists (crop coordinates), f, centre, baseline, distortion a.
#          theta = (r/f) * (1 + a*(r/f)^2) is applied identically to both cameras,
#          as one shared lens model must be.
# OUTPUTS: (bin centres, median heights) for bins with enough matches.
# KEYWORDS: distortion, profile, triangulation
def profile(K1, K2, f, centre, baseline, a, nbin=17, step=25.0):
    cx, cy = centre
    hs, rs = [], []
    for k1, k2 in zip(K1, K2):
        if len(k1) == 0:
            continue
        rays = []
        for k in (k1, k2):
            dvec = k - np.array([cx, cy], float)
            r = np.hypot(dvec[:, 0], dvec[:, 1])
            t = r / f
            th = t * (1.0 + a * t * t)
            s, c = np.sin(th), np.cos(th)
            safe = np.where(r > 0, r, 1.0)
            rays.append(np.stack([s * dvec[:, 0] / safe, s * dvec[:, 1] / safe, c], axis=1))
        d1, d2 = rays
        o2 = np.array([baseline, 0.0, 0.0])
        w = -o2
        A = np.einsum("ij,ij->i", d1, d1)
        Bb = np.einsum("ij,ij->i", d1, d2)
        C = np.einsum("ij,ij->i", d2, d2)
        Dd = d1 @ w
        E = d2 @ w
        den = A * C - Bb * Bb
        ok = np.abs(den) > 1e-12
        t1 = np.where(ok, (Bb * E - C * Dd) / np.where(ok, den, 1.0), np.nan)
        t2 = np.where(ok, (A * E - Bb * Dd) / np.where(ok, den, 1.0), np.nan)
        p1 = t1[:, None] * d1
        p2 = o2[None, :] + t2[:, None] * d2
        z = 0.5 * (p1[:, 2] + p2[:, 2])
        keep = ok & np.isfinite(z) & (z > 50) & (z < 20000)
        hs.append(z[keep])
        rs.append(np.hypot(k1[keep, 0] - cx, k1[keep, 1] - cy))
    h = np.concatenate(hs); r = np.concatenate(rs)
    centres, med = [], []
    for lo in np.arange(0, 425, step):
        m = (r >= lo) & (r < lo + step)
        if m.sum() >= 200:
            centres.append(lo + step / 2)
            med.append(float(np.median(h[m])))
    return np.array(centres), np.array(med)


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    B = float(d["baseline"]); centre = tuple(d["centre"])
    # f and the distortion coefficient are jointly determined: flatness fixes only
    # the SHAPE of the angle mapping, so f must come from elsewhere (the Sun or the
    # image circle) and `a` is then what the sky says. Allow f to be overridden to
    # show how the pair trades off.
    f = float(sys.argv[1]) if len(sys.argv) > 1 else float(d["f"])
    print(f"f = {f:.1f} px/rad, B = {B} m, {len(K1)} frames\n")
    print(f"{'a':>10} {'spread':>9} {'median H':>10} {'inner':>8} {'outer':>8}")
    rows = []
    grid = [round(x, 3) for x in np.arange(-0.80, 0.021, 0.05)]
    for a in grid:
        x, m = profile(K1, K2, f, centre, B, a)
        if len(m) < 5:
            continue
        spread = float((m.max() - m.min()) / m.mean())
        rows.append(dict(a=a, spread=spread, median=float(np.median(m)),
                         inner=float(m[0]), outer=float(m[-1]), x=x.tolist(), m=m.tolist()))
        print(f"{a:>10.2f} {100*spread:>8.1f}% {np.median(m):>10.0f} "
              f"{m[0]:>8.0f} {m[-1]:>8.0f}")
    best = min(rows, key=lambda r: r["spread"])
    print(f"\nflattest over this grid: a = {best['a']:+.2f} with spread "
          f"{100*best['spread']:.1f} % (compare a = 0, equidistant: "
          f"{100*[r for r in rows if r['a']==0.0][0]['spread']:.1f} %)")
    if best["spread"] < 0.05:
        print("-> distortion in this family DOES flatten the profile; "
              "the coefficient is the measurement.")
    else:
        print("-> no cubic distortion in this family removes the trend. The radial "
              "profile is not explained by the lens law, distortion, radial scale, "
              "or the relative orientation within its physically allowed range.")

    json.dump(rows, open(os.path.join(HERE, "distortion_fit.json"), "w"), indent=1)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for r in rows:
        style = "-" if r["a"] in (0.0, best["a"]) else "--"
        lw = 2.2 if r["a"] in (0.0, best["a"]) else 1.0
        ax.plot(r["x"], r["m"], style, lw=lw, ms=4, marker="o" if r["a"] in (0.0, best["a"]) else "",
                label=f"a={r['a']:+.2f}  spread {100*r['spread']:.0f}%")
    ax.set_xlabel("radius from principal point (px)")
    ax.set_ylabel("median height (m)")
    ax.set_title("Can a distortion polynomial flatten the radial trend?")
    ax.legend(fontsize=7.5, ncol=2)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "distortion_fit.png"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT}/distortion_fit.png")


if __name__ == "__main__":
    main()
