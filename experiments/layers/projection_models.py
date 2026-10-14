# PURPOSE: Test which lens law the cameras actually obey, by exploiting a property
#          that does not depend on knowing the cloud: the apparent height of a real
#          cloud layer must not depend systematically on where in the image the
#          match lies. The measured height DOES fall with radius (1849 m at the
#          centre to 1254 m at the edge), which is either a real radial structure in
#          the sky - physically implausible over a 5-hour mean - or the wrong
#          mapping from image radius to viewing angle.
#
#          Four candidate mappings are compared, all normalized to pass through the
#          same focal constant so that only the SHAPE of theta(r) differs:
#            equidistant   theta = r/f          (what the pipeline assumes)
#            equisolid     theta = 2 asin(r/2f)
#            stereographic theta = 2 atan(r/2f)
#            orthographic  theta = asin(r/f)
#          The correct model is the one whose radial trend is flattest.
#
#          This is a projection-model test, not a height calibration: it uses no
#          ground truth and no ceilometer.
# INPUTS: experiments/layers/heights.npz (both keypoint sets).
# OUTPUTS: docs/report/projection_models.png and a printed table of radial trends.
# KEYWORDS: projection_model, fisheye, equidistant, lens_law, radial_trend
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

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})

MODELS = {
    "equidistant": lambda t: t,
    "equisolid": lambda t: 2.0 * np.arcsin(np.clip(t / 2.0, -1, 1)),
    "stereographic": lambda t: 2.0 * np.arctan(t / 2.0),
    "orthographic": lambda t: np.arcsin(np.clip(t, -1, 1)),
}


# PURPOSE: Turn image points into unit rays under a named lens law.
#          `t` is the normalized image radius r/f; each model maps it to the angle
#          from the optical axis. Equidistant is the identity, so the four models
#          differ only in how the periphery is compressed.
# INPUTS: uv (N,2); f (px/rad); centre (2,); model (str key of MODELS).
# OUTPUTS: (N,3) unit vectors. r = 0 gives (0,0,1); a ray that a model cannot
#          represent (arcsin argument out of range) is clipped rather than raising.
# KEYWORDS: ray, projection, fisheye
def rays(uv, f, centre, model):
    d = np.asarray(uv, float) - np.asarray(centre, float)
    r = np.hypot(d[:, 0], d[:, 1])
    t = r / f
    th = MODELS[model](t)
    s, c = np.sin(th), np.cos(th)
    safe = np.where(r > 0, r, 1.0)
    return np.stack([s * d[:, 0] / safe, s * d[:, 1] / safe, c], axis=1)


# PURPOSE: Triangulate two ray sets and return the height of the closest approach.
# INPUTS: d1, d2 (N,3) unit rays; baseline (m).
# OUTPUTS: (N,) heights in metres, NaN where the rays are parallel or behind.
# KEYWORDS: triangulation, height
def heights(d1, d2, baseline):
    w = np.array([-baseline, 0.0, 0.0])
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    dd = d1 @ w
    e = d2 @ w
    den = a * c - b * b
    ok = np.abs(den) > 1e-12
    t1 = np.where(ok, (b * e - c * dd) / np.where(ok, den, 1.0), np.nan)
    p1 = t1[:, None] * d1
    p2 = np.array([baseline, 0, 0])[None, :] + np.where(ok, (a * e - b * dd) /
                                                      np.where(ok, den, 1.0), np.nan)[:, None] * d2
    z = 0.5 * (p1[:, 2] + p2[:, 2])
    return np.where(ok & np.isfinite(z) & (z > 0), z, np.nan)


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    f = float(d["f"])
    B = float(d["baseline"])
    cx, cy = d["centre"]
    print(f"f = {f:.1f} px/rad, baseline {B} m, {len(K1)} frames\n")

    results = {}
    for model in MODELS:
        hs, rs = [], []
        for a, b in zip(K1, K2):
            if len(a) == 0:
                continue
            h = heights(rays(a, f, (cx, cy), model), rays(b, f, (cx, cy), model), B)
            rad = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
            keep = np.isfinite(h) & (h > 50) & (h < 20000)
            hs.append(h[keep])
            rs.append(rad[keep])
        h = np.concatenate(hs)
        r = np.concatenate(rs)
        prof = []
        for lo in np.arange(0, 425, 25):
            m = (r >= lo) & (r < lo + 25)
            if m.sum() >= 100:
                prof.append((lo + 12.5, float(np.median(h[m])), int(m.sum())))
        results[model] = dict(h=h, r=r, prof=prof, median=float(np.median(h)))
        inner = h[r < 150]
        outer = h[r > 350]
        print(f"{model:>14}: median {np.median(h):6.0f} m   "
              f"inner(r<150) {np.median(inner):6.0f}   outer(r>350) {np.median(outer):6.0f}   "
              f"drop {100*(1-np.median(outer)/np.median(inner)):5.1f} %")

    # the model whose central-to-outer drop is smallest is the better lens law
    best = min(results, key=lambda k: abs(np.median(results[k]["h"][results[k]["r"] > 350])
                                          / np.median(results[k]["h"][results[k]["r"] < 150]) - 1))
    print(f"\nflattest radial trend: {best}")
    for model, res in results.items():
        x = [p[0] for p in res["prof"]]
        y = [p[1] for p in res["prof"]]
        plt.plot(x, y, "o-", ms=3.5, lw=1.7,
                 label=f"{model}  (drop {100*(1-y[-1]/y[0]):.0f} %)")
    plt.xlabel("radius from principal point (px)")
    plt.ylabel("median match height (m)")
    plt.title("Which lens law makes the apparent height independent of radius?")
    plt.legend(fontsize=8.5)
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, "projection_models.png"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT}/projection_models.png")


if __name__ == "__main__":
    main()
