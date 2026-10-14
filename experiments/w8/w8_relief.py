# PURPOSE: W8 injection 1 — 3-D relief. Does a rough cloud base break the recovery that a
#          flat plane passes?
#
#          WHY THIS IS FIRST. The W7 control rendered a flat plane and the matcher
#          recovered its height to 0.07 px. A real cloud base is not flat: it has billows
#          and gaps on the scale of the footprint. Relief matters for a specific reason -
#          with a flat plane the two cameras see the SAME surface point and the match is
#          geometrically well posed; with relief they sample genuinely different points,
#          and the disparity field acquires a variance that has nothing to do with the mean
#          height. That is a mechanism no flat-plane control can exercise.
#
#          The other injections are ordered after this one. Atmospheric haze is placed LAST
#          because W6 already measured the photometric coupling and found it runs the wrong
#          way to explain the inversion (rho(brightness, du) = -0.176 while the high regime
#          is brighter). Relief and broken fields are untested.
#
#          THE PRE-REGISTERED TRIGGER. The real-world effect is a +1.8 px disparity shift
#          that makes a HIGHER cloud look LOWER. So an injection "triggers the inversion"
#          only if the recovered height FAILS to increase with the true height - inverting,
#          or compressing by more than 50 % - across at least three heights. A single
#          render can show a bias; a bias is not an inversion.
# INPUTS: the validated renderer (w7_renderer.py), whose gate must pass first.
# OUTPUTS: experiments/w8/w8_relief.json, docs/report/w8_relief.png.
# KEYWORDS: W8, relief, 3D, injection, bisection, inversion, pre_registered
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "w7"))

import w7_renderer as R                                                       # noqa: E402
from w7_synthetic import cloud_texture                                        # noqa: E402


# PURPOSE: A relief field in metres over the cloud plane, at a chosen roughness.
# INPUTS: X, Y world coordinates (m); rms_m the desired RMS relief; seed.
# OUTPUTS: relief in metres with approximately the requested RMS, zero mean.
#          Built from the same procedural generator as the texture so the two share a
#          spectrum; a white-noise relief would put all its power at the pixel scale and
#          would not resemble a cloud base.
# KEYWORDS: relief, cloud_base, roughness
def relief_field(X, Y, rms_m, extent_m=4000.0, seed=11):
    tex = cloud_texture(512, 60.0, seed=seed)
    tx = np.mod((X + extent_m / 2) / extent_m, 1.0)
    ty = np.mod((Y + extent_m / 2) / extent_m, 1.0)
    px = np.clip((tx * (tex.shape[1] - 1)).astype(np.int32), 0, tex.shape[1] - 1)
    py = np.clip((ty * (tex.shape[0] - 1)).astype(np.int32), 0, tex.shape[0] - 1)
    z = tex[py, px]
    z = z - z.mean()
    s = z.std()
    return z * (rms_m / s) if s > 0 else z


# PURPOSE: Cast camera 1's rays onto a RELIEVED surface z = H + relief(x, y).
# INPUTS: H, rms relief in metres, grid.
# OUTPUTS: (grid*grid, 3) surface points, the radius map, and the residual after the
#          fixed-point iteration - a large residual would mean the intersection failed and
#          any disparity computed from it is meaningless.
# KEYWORDS: ray_cast, relieved_surface, fixed_point
def cast_relieved(H, rms_m, grid=R.ORIG, iters=12):
    xs = np.arange(grid, dtype=np.float64)
    uu, vv = np.meshgrid(xs, xs)
    du = uu - R.C1[0]; dv = vv - R.C1[1]
    r = np.hypot(du, dv)
    t = r / R.F_PX
    theta = t * (1.0 + R.A_DIST * t * t)
    phi = np.arctan2(dv, du)
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    z = np.full_like(dx, float(H))
    for _ in range(iters):
        s = z / np.maximum(dz, 1e-9)
        X = s * dx; Y = s * dy
        z = H + relief_field(X, Y, rms_m)
    s = z / np.maximum(dz, 1e-9)
    P = np.stack([s * dx, s * dy, z], axis=-1).reshape(-1, 3)
    return P, r


def main():
    from clouds.config import load_config
    from clouds.geometry.triangulation import heights_from_matches
    from clouds.matching.matchers import create_matcher
    from clouds.pipeline.analyze import lowest_prominent_mode

    if not R.validate():
        print("RENDERER GATE FAILED - no result may be reported")
        return

    cfg = load_config(os.path.join(REPO, "configs", "paper_dlcp2026_roll.yaml"))
    matcher = create_matcher("roma", {"grid_stride": 8, "certainty_threshold": 0.0},
                             "cuda")
    tex = cloud_texture(1024, 90.0, seed=3)
    y0 = (R.ORIG - R.CROP) // 2
    heights = [1000.0, 1500.0, 2000.0, 3000.0, 4000.0]

    rows = []
    for rms_frac in (0.0, 0.05, 0.10, 0.20):
        print(f"\n--- relief {100*rms_frac:.0f}% of height ---")
        print(f"{'H_true':>7} {'gt du':>8} {'RoMa du':>9} {'bias px':>8} "
              f"{'median H':>9} {'mode H':>8}")
        series = {"H": [], "gt": [], "roma": [], "median": [], "mode": []}
        for H in heights:
            rms_m = rms_frac * H
            P, r = cast_relieved(H, rms_m)
            p2 = R.project_cam2(P).reshape(R.ORIG, R.ORIG, 2)
            M = R._affine()
            xs = np.arange(R.ORIG, dtype=np.float64)
            uu, vv = np.meshgrid(xs, xs)
            u_w = M[0, 0] * uu + M[0, 1] * vv + M[0, 2]
            gt = u_w - p2[:, :, 0]
            gt[r > 940] = np.nan
            # texture sampled at the surface points, then camera 1 warped as the pipeline does
            shape = tex.shape
            extent = 4000.0
            txf = np.mod((P[:, 0] + extent / 2) / extent, 1.0)
            tyf = np.mod((P[:, 1] + extent / 2) / extent, 1.0)
            g1 = tex[np.clip((tyf * (shape[0] - 1)).astype(np.int32), 0, shape[0] - 1),
                     np.clip((txf * (shape[1] - 1)).astype(np.int32), 0, shape[1] - 1)]
            g1 = g1.reshape(R.ORIG, R.ORIG)
            g1[r > 940] = 0.0
            g1w = __import__("cv2").warpAffine(g1.astype(np.float32), M,
                                               dsize=(R.ORIG, R.ORIG))
            # camera 2's texture: sample at the points camera 2 sees
            p2i = R.project_cam2(P)
            # invert camera 2's projection by casting its own rays onto the same surface
            g2 = np.zeros((R.ORIG, R.ORIG), np.float32)
            for _ in range(1):
                pass
            # camera 2's rays
            uu2, vv2 = np.meshgrid(xs, xs)
            ph2 = (np.arctan2(vv2 - R.C2[1], uu2 - R.C2[0]) - math.pi / 2.0)
            rr2 = np.hypot(uu2 - R.C2[0], vv2 - R.C2[1])
            tt2 = rr2 / R.F_PX
            th2 = tt2 * (1.0 + R.A_DIST * tt2 * tt2)
            z2 = np.full_like(th2, float(H))
            for _ in range(12):
                s2 = z2 / np.maximum(np.cos(th2), 1e-9)
                X2 = s2 * np.sin(th2) * np.cos(ph2)
                Y2 = -R.B_M + s2 * np.sin(th2) * np.sin(ph2)
                z2 = H + relief_field(X2, Y2, rms_m)
            s2 = z2 / np.maximum(np.cos(th2), 1e-9)
            X2 = s2 * np.sin(th2) * np.cos(ph2)
            Y2 = -R.B_M + s2 * np.sin(th2) * np.sin(ph2)
            tx2 = np.mod((X2 + extent / 2) / extent, 1.0)
            ty2 = np.mod((Y2 + extent / 2) / extent, 1.0)
            g2 = tex[np.clip((ty2 * (shape[0] - 1)).astype(np.int32), 0, shape[0] - 1),
                     np.clip((tx2 * (shape[1] - 1)).astype(np.int32), 0, shape[1] - 1)]
            g2[rr2 > 940] = 0.0

            c1 = (g1w[y0:y0 + R.CROP, y0:y0 + R.CROP] * 255).astype(np.uint8)
            c2 = (g2[y0:y0 + R.CROP, y0:y0 + R.CROP] * 255).astype(np.uint8)
            k0, k1 = matcher(c1, c2)
            k0 = np.asarray(k0, float).reshape(-1, 2)
            k1 = np.asarray(k1, float).reshape(-1, 2)
            rec = k0[:, 0] - k1[:, 0]
            ui = np.clip(k0[:, 0].astype(int), 0, R.CROP - 1)
            vi = np.clip(k0[:, 1].astype(int), 0, R.CROP - 1)
            truth = gt[y0:y0 + R.CROP, y0:y0 + R.CROP][vi, ui]
            ok = np.isfinite(truth)
            if ok.sum() < 100:
                continue
            hh = heights_from_matches(k0, k1, R.F_PX, R.B_M,
                                      centre=(R.CROP / 2.0, R.CROP / 2.0),
                                      distortion_a=R.A_DIST)["heights"]
            hf = hh[np.isfinite(hh)]
            hmode, _ = lowest_prominent_mode(hf, None)
            series["H"].append(H); series["gt"].append(float(np.median(truth[ok])))
            series["roma"].append(float(np.median(rec[ok])))
            series["median"].append(float(np.median(hf)))
            series["mode"].append(None if hmode is None else float(hmode))
            print(f"{H:>7.0f} {np.median(truth[ok]):>8.2f} {np.median(rec[ok]):>9.2f} "
                  f"{np.median(rec[ok]-truth[ok]):>+8.3f} {np.median(hf):>9.0f} "
                  f"{(hmode if hmode else float('nan')):>8.0f}")
        from scipy.stats import spearmanr
        Hv = np.array(series["H"]); gtv = np.array(series["gt"]); rcv = np.array(series["roma"])
        rho, p = spearmanr(gtv, rcv)
        mv = np.array([x if x is not None else np.nan for x in series["mode"]], float)
        m = np.isfinite(mv)
        rho_m = spearmanr(Hv[m], mv[m])[0] if m.sum() >= 3 else np.nan
        print(f"   matcher rho = {rho:+.3f} (p={p:.3f});  lowest_mode rho = {rho_m:+.3f}")
        rows.append({"rms_frac": rms_frac, "series": series,
                     "matcher_rho": float(rho), "mode_rho": float(rho_m)})

    print("\nPRE-REGISTERED TRIGGER: does the recovered height fail to increase with H_true?")
    print(f"{'relief':>8} {'matcher rho':>13} {'lowest_mode rho':>17} {'triggered':>11}")
    for r in rows:
        trig = (r["matcher_rho"] < 0.9) or (r["mode_rho"] < 0.9)
        print(f"{100*r['rms_frac']:>7.0f}% {r['matcher_rho']:>+13.3f} "
              f"{r['mode_rho']:>+17.3f} {'YES' if trig else 'no':>11}")
    hit = [r for r in rows if r["matcher_rho"] < 0.9 or r["mode_rho"] < 0.9]
    verdict = ("RELIEF TRIGGERS THE FAILURE at "
               + ", ".join(f"{100*r['rms_frac']:.0f}%" for r in hit)
               if hit else
               "RELIEF DOES NOT TRIGGER THE FAILURE at any tested roughness - the "
               "recovery survives 3-D relief")
    print(f"\nVERDICT: {verdict}")
    json.dump({"rows": rows, "verdict": verdict},
              open(os.path.join(HERE, "w8_relief.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(heights, heights, "o-", color="#1b2430", label="truth")
    for r, c in zip(rows, ("#5c6672", "#1f6feb", "#c1440e", "#1a7f37")):
        mv = [x if x is not None else np.nan for x in r["series"]["mode"]]
        ax.plot(r["series"]["H"], mv, "s--", color=c,
                label=f"lowest_mode, relief {100*r['rms_frac']:.0f}%")
    ax.set_xlabel("true cloud height (m)"); ax.set_ylabel("reported height (m)")
    ax.set_title("W8 injection 1: 3-D relief", fontsize=10); ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "w8_relief.png"), bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}/w8_relief.png and {os.path.join(HERE, 'w8_relief.json')}")


if __name__ == "__main__":
    main()
