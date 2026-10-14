# PURPOSE: Characterise the static spatial artefact found in R11, directly, as a
#          geometric error field.
#
#          R11 showed that the estimated height is dominated by a pattern fixed to the
#          image rather than to the cloud (map correlation r = 0.98 between the session's
#          two halves, against a shuffled null of 0). Since the cloud moves and the
#          pattern does not, the STATIC part of the disparity field is not the cloud: it
#          is the calibration and lens error, readable directly.
#
#          This separates the field into its static and varying parts and asks what
#          geometric form the static part takes. Candidates, each with a signature:
#
#            constant offset      -> a uniform disparity shift (translation error)
#            linear in (u, v)     -> a plane (scale/aspect error)
#            radial, even in r    -> lens distortion or a principal-point error
#            odd in (u, v)        -> a rotation (roll)
#
#          A correct diagnosis says which correction removes it; a wrong one leaves the
#          pattern and the estimator still measures the instrument.
#
#          Ground-truth-free: no reference is used anywhere.
# INPUTS: experiments/layers/heights.npz (per-match image positions in both frames).
# OUTPUTS: docs/report/r8_static_field.png, experiments/layers/r8_static_field.json.
# KEYWORDS: R8, static_field, calibration_error, distortion, diagnosis, disparity
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
CROP = 640
CENTRE = CROP / 2.0
CELL = 20
GRID = CROP // CELL


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    K1, K2 = list(d["uv"]), list(d["kp2"])
    f = float(d["f"]); B = float(d["baseline"])

    # per-match disparity components and positions, pooled across the session
    du_all, dv_all, u_all, v_all, frame_all = [], [], [], [], []
    for fi, (a, b) in enumerate(zip(K1, K2)):
        if len(a) == 0:
            continue
        du_all.append((a[:, 0] - b[:, 0]))
        dv_all.append((a[:, 1] - b[:, 1]))
        u_all.append(a[:, 0]); v_all.append(a[:, 1])
        frame_all.append(np.full(len(a), fi))
    du = np.concatenate(du_all); dv = np.concatenate(dv_all)
    u = np.concatenate(u_all); v = np.concatenate(v_all)
    fr = np.concatenate(frame_all)
    nfr = fr.max() + 1
    print(f"frames {nfr}, matches {len(du)}")

    ci = np.clip((u / CELL).astype(int), 0, GRID - 1)
    cj = np.clip((v / CELL).astype(int), 0, GRID - 1)

    # static component: the within-cell median is by construction the part that does not
    # change between frames (a changing part is averaged out by the median only if it is
    # small; to be strict we also compute the temporal spread per cell below)
    stat = np.full((GRID, GRID), np.nan)
    cell_temporal_iqr = np.full((GRID, GRID), np.nan)
    for j in range(GRID):
        for i in range(GRID):
            m = (ci == i) & (cj == j)
            if m.sum() < 50:
                continue
            stat[j, i] = float(np.median(du[m]))
            per_frame = [np.median(du[m & (fr == k)]) for k in np.unique(fr[m])]
            per_frame = [x for x in per_frame if np.isfinite(x)]
            if len(per_frame) >= 10:
                cell_temporal_iqr[j, i] = float(np.percentile(per_frame, 75)
                                                - np.percentile(per_frame, 25))
    ok = np.isfinite(stat)
    print(f"cells {int(ok.sum())}/{GRID*GRID}")
    print(f"static disparity field: mean {np.nanmean(stat):.2f} px, "
          f"std {np.nanstd(stat):.2f} px, range "
          f"{np.nanmin(stat):.2f}-{np.nanmax(stat):.2f} px")
    print(f"per-cell temporal IQR  : median {np.nanmedian(cell_temporal_iqr):.2f} px")
    ratio = np.nanstd(stat) / max(np.nanmedian(cell_temporal_iqr), 1e-9)
    print(f"  static spatial spread / temporal spread = {ratio:.2f}")
    print(f"  -> {'STATIC DOMINATES' if ratio > 1.5 else 'time dominates'}")

    # what geometric form does the static field take?
    uu, vv = np.meshgrid((np.arange(GRID) + 0.5) * CELL,
                         (np.arange(GRID) + 0.5) * CELL)
    du_c = uu - CENTRE
    dv_c = vv - CENTRE
    r = np.hypot(du_c, dv_c)
    y = stat[ok]
    terms = {
        "constant":            np.ones_like(y),
        "linear u":            du_c[ok],
        "linear v":            dv_c[ok],
        "radial r":            r[ok],
        "r^2":                 (r[ok] ** 2) / CENTRE,
        "radial cubic r^3":    (r[ok] ** 3) / (CENTRE ** 2),
        "tangential (roll)":   np.arctan2(dv_c[ok], du_c[ok]),
    }
    print(f"\n{'model':>26} {'rms residual (px)':>18} {'explains':>9}")
    fits = {}
    for name, t in terms.items():
        A = np.vstack([t, np.ones_like(t)]).T
        sol, _res, _rk, _sv = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ sol
        rms = float(np.sqrt(np.mean(res ** 2)))
        fits[name] = {"coef": float(sol[0]), "rms": rms}
        print(f"{name:>26} {rms:>18.3f} {100*(1-rms/np.std(y)):>8.1f}%")

    # a model with everything radial, to test whether radial structure suffices
    A = np.vstack([terms["constant"], terms["radial r"], terms["r^2"],
                   terms["radial cubic r^3"]]).T
    sol, _res, _rk, _sv = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ sol
    rms_radial = float(np.sqrt(np.mean(res ** 2)))
    A2 = np.vstack([terms["constant"], terms["linear u"], terms["linear v"],
                    terms["r^2"], terms["radial cubic r^3"],
                    terms["tangential (roll)"]]).T
    sol2, _res, _rk, _sv = np.linalg.lstsq(A2, y, rcond=None)
    rms_full = float(np.sqrt(np.mean((y - A2 @ sol2) ** 2)))
    print(f"\n{'radial terms only':>26} {rms_radial:>18.3f} "
          f"{100*(1-rms_radial/np.std(y)):>8.1f}%")
    print(f"{'radial + linear + roll':>26} {rms_full:>18.3f} "
          f"{100*(1-rms_full/np.std(y)):>8.1f}%")
    print(f"\n(the field's own std is {np.nanstd(stat):.2f} px, so an 'explains' of 100%)")
    print(f" means the model accounts for the whole static pattern)")

    json.dump({"n_frames": int(nfr), "static_std_px": float(np.nanstd(stat)),
               "static_mean_px": float(np.nanmean(stat)),
               "cell_temporal_iqr_px": float(np.nanmedian(cell_temporal_iqr)),
               "space_time_ratio": float(ratio),
               "model_rms_px": {k: v["rms"] for k, v in fits.items()},
               "radial_only_rms_px": rms_radial,
               "full_model_rms_px": rms_full},
              open(os.path.join(HERE, "r8_static_field.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    im0 = axes[0].imshow(stat, origin="upper", cmap="RdBu_r",
                         vmin=np.nanmean(stat) - 2 * np.nanstd(stat),
                         vmax=np.nanmean(stat) + 2 * np.nanstd(stat))
    axes[0].set_title("Static disparity field (px)", fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(cell_temporal_iqr, origin="upper", cmap="magma")
    axes[1].set_title("Per-cell temporal IQR (px)", fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].scatter(r[ok].ravel(), y, s=6, color="#c1440e", alpha=0.6)
    axes[2].set_xlabel("radius from principal point (px)")
    axes[2].set_ylabel("static disparity (px)")
    axes[2].set_title("Is the static field radial?", fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(OUT, "r8_static_field.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/r8_static_field.png")


if __name__ == "__main__":
    main()
