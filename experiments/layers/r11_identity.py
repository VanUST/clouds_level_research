# PURPOSE: Answer R11 — is the estimated height a property of the CLOUD or of the IMAGE
#          POSITION?
#
#          This is the sharpest available discriminator for the target-identity question,
#          and it needs no reference at all (Tier 1). The logic:
#
#            * a correct retrieval assigns a height to a piece of cloud, so as the cloud
#              moves across the field the height map moves with it and the map measured
#              over the first half of the session should look DIFFERENT from the second
#              half;
#            * an artefact assigns a height to a place in the image, so the map is
#              STATIC: the same cells carry the same heights all session, and the first
#              and second halves agree closely.
#
#          The measured cloud motion (median 3.27 m/s at the estimated height, azimuth
#          ~213 deg) moves cloud about 10 px per 10 s frame, so across five hours the same
#          air passes through the field many times over. A static map is therefore not
#          explicable by the cloud standing still.
#
#          Ground-truth-free: heights come from exact triangulation of real matches.
# INPUTS: experiments/layers/heights.npz (per-match heights and image positions).
# OUTPUTS: docs/report/r11_identity.png, experiments/layers/r11_identity.json.
# KEYWORDS: R11, temporal, target_identity, spatial_map, static_artefact, Tier1
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
GRID = 32                      # 20 px cells


# PURPOSE: Build a map of median height per image cell over a set of frames.
# INPUTS: list of (heights, uv) per frame; grid resolution; minimum matches per cell.
# OUTPUTS: (map, counts) arrays of shape (GRID, GRID), NaN where a cell is too sparse.
#          A median rather than a mean, because the height field is heavy-tailed.
# KEYWORDS: map, grid, median, spatial
def height_map(frames, grid=GRID, min_n=20):
    acc = np.full((grid, grid, 0), 0.0)
    buckets = [[[] for _ in range(grid)] for _ in range(grid)]
    for h, uv in frames:
        if len(h) == 0:
            continue
        ui = np.clip((uv[:, 0] / CROP * grid).astype(int), 0, grid - 1)
        vi = np.clip((uv[:, 1] / CROP * grid).astype(int), 0, grid - 1)
        for a, b, hv in zip(ui, vi, h):
            if np.isfinite(hv):
                buckets[b][a].append(hv)
    m = np.full((grid, grid), np.nan)
    c = np.zeros((grid, grid), int)
    for b in range(grid):
        for a in range(grid):
            v = buckets[b][a]
            c[b, a] = len(v)
            if len(v) >= min_n:
                m[b, a] = float(np.median(v))
    return m, c


def main():
    d = np.load(os.path.join(HERE, "heights.npz"), allow_pickle=True)
    H, UV = list(d["heights"]), list(d["uv"])
    frames = [(h, uv) for h, uv in zip(H, UV) if len(h) > 0]
    n = len(frames)
    print(f"frames {n}, matches {sum(len(h) for h, _ in frames)}")
    half = n // 2

    m_all, c = height_map(frames)
    m_a, _ = height_map(frames[:half])
    m_b, _ = height_map(frames[half:])
    ok = np.isfinite(m_all) & np.isfinite(m_a) & np.isfinite(m_b)
    print(f"cells with enough matches in all three maps: {int(ok.sum())}/{GRID*GRID}")

    def corr(x, y):
        x, y = x[ok], y[ok]
        if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    r_halves = corr(m_a, m_b)
    # null for the same statistic: shuffle the cells of the second half
    rng = np.random.default_rng(3)
    null = []
    for _ in range(400):
        x = m_a[ok].copy()
        y = m_b[ok].copy()
        rng.shuffle(y)
        if np.std(x) and np.std(y):
            null.append(float(np.corrcoef(x, y)[0, 1]))
    null = np.array(null)

    # spatial variance vs temporal variance: how much of a cell's height is explained by
    # WHICH cell it is, versus by WHEN it was measured
    kh = np.array([np.median(h) for h, _ in frames])
    print(f"\nper-frame median height: std {np.std(kh):.0f} m "
          f"(the temporal signal available)")
    cell_means = np.nanmedian(m_all)
    print(f"spatial map of median height: spread across cells "
          f"{np.nanstd(m_all):.0f} m, mean {cell_means:.0f} m")

    print(f"\nR11 core test")
    print(f"  correlation of the height map between the session's two halves:")
    print(f"    observed r = {r_halves:+.3f}")
    print(f"    null (cells shuffled) mean {null.mean():+.3f}, "
          f"95th pct {np.percentile(null,95):+.3f}")
    p = float(np.mean(null >= r_halves))
    if r_halves > 0.5 and p < 0.05:
        verdict = "STATIC MAP: the height is largely a property of image position"
    elif p > 0.05:
        verdict = "no evidence of a static map"
    else:
        verdict = "weakly static"
    print(f"    -> {verdict}")

    # how stable is each cell over time? a static map has a small temporal spread
    print(f"\n  a static map is also implied by low per-cell temporal variability:")
    per_cell_iqr = []
    for k in range(0, n, 6):
        m, _ = height_map(frames[k:k + 6])
        per_cell_iqr.append(m)
    arr = np.array(per_cell_iqr)
    cell_temporal_iqr = np.nanmedian(
        [np.nanpercentile(arr[:, b, a], 75) - np.nanpercentile(arr[:, b, a], 25)
         for b in range(GRID) for a in range(GRID)
         if np.isfinite(arr[:, b, a]).sum() >= 5] or [np.nan])
    print(f"    median per-cell temporal IQR {cell_temporal_iqr:.0f} m")
    print(f"    median spatial spread across cells {np.nanstd(m_all):.0f} m")
    ratio = np.nanstd(m_all) / max(cell_temporal_iqr, 1e-9)
    print(f"    spatial / temporal ratio {ratio:.2f}  "
          f"({'space dominates - consistent with a static pattern' if ratio > 1.5 else 'time dominates - the map is not fixed'})")

    json.dump({"n_frames": n, "r_halves": r_halves,
               "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
               "p": p, "verdict": verdict,
               "spatial_std_m": float(np.nanstd(m_all)),
               "cell_temporal_iqr_m": float(cell_temporal_iqr),
               "ratio_space_time": float(ratio),
               "frame_median_std_m": float(np.std(kh))},
              open(os.path.join(HERE, "r11_identity.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    for ax, m, t in ((axes[0], m_a, f"first half (n={half} frames)"),
                     (axes[1], m_b, f"second half (n={n-half} frames)"),
                     (axes[2], m_all, "whole session")):
        im = ax.imshow(m, cmap="viridis", origin="upper")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, label="median height (m)")
    fig.suptitle(f"Height by image cell — halves correlate r = {r_halves:+.2f} "
                 f"(shuffled null {null.mean():+.2f})", fontsize=10, y=1.04)
    fig.savefig(os.path.join(OUT, "r11_identity.png"), bbox_inches="tight",
                facecolor="white")
    print(f"\nwrote {OUT}/r11_identity.png")


if __name__ == "__main__":
    main()
