#!/usr/bin/env python3
"""Per-window diagnostic with Reliability Index (RI) validation.

Splits data into 30-min sections, computes per-window r and RI,
stitches all windows, validates RI as a predictor of ceilometer r.
"""
import os, sys, re, numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as sp_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.temporal_filter import TemporalFilter, ClusterMeasurement
from src.camera import StereoCameraSystem
from config import CameraModel

OUTPUT_DIR = "report_window_diagnostic"
WINDOW_MIN = 30
SLIDE_MIN = 15
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
stereo = StereoCameraSystem(45.0, 180.0, 1920, CameraModel.FISHEYE)

ceilo = []
with open("data/height data/Zve_Sci_250523.txt") as f:
    for line in f:
        m = re.match(r"^(\d{12})\s+MSK\s+H1\s+=\s+b'(\d+)'", line)
        if not m: continue
        h = int(m.group(2))
        if h == 9999: continue
        ceilo.append((datetime.strptime(m.group(1), "%y%m%d%H%M%S"), h))
ceilo.sort(key=lambda x: x[0])

log_dir = "logs/250523/kornia_loftr"
algo_frames_raw = []
for fname in sorted(os.listdir(log_dir)):
    m = re.match(r"img-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.npz", fname)
    if not m: continue
    ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
    d = np.load(os.path.join(log_dir, fname), allow_pickle=True)
    kp1, labels, shifts = d["kp1"], d["labels"], d["shifts"]
    clusters = []
    for lbl in sorted(set(labels)):
        if lbl < 0: continue
        mask = labels == lbl; n = int(np.sum(mask))
        if n < 5: continue
        csh, ckp = shifts[mask], kp1[mask]
        dx = float(np.mean(csh[:, 0])); ds = float(np.std(csh[:, 0]))
        h, _ = stereo.compute_distance(dx, 0.0)
        _, err_sem = stereo.compute_distance(dx, ds / np.sqrt(max(n, 1)))
        if not np.isfinite(h) or h <= 0: continue
        clusters.append(ClusterMeasurement(
            h=float(h), h_err=float(err_sem),
            u=float(np.mean(ckp[:, 0])), v=float(np.mean(ckp[:, 1])),
            size=n, disp_x=dx, disp_std=ds, cluster_id=int(lbl)))
    if clusters: algo_frames_raw.append((ts, clusters, kp1, labels, shifts))
algo_frames_raw.sort(key=lambda x: x[0])

algo_t0 = algo_frames_raw[0][0]; algo_t1 = algo_frames_raw[-1][0]
ceilo = [(t, h) for t, h in ceilo if algo_t0 <= t <= algo_t1]
print(f"Algo: {len(algo_frames_raw)} frames ({algo_t0.strftime('%H:%M')} - {algo_t1.strftime('%H:%M')})")
print(f"Ceilo: {len(ceilo)} points (truncated)")
print(f"Windows: {WINDOW_MIN}min, slide: {SLIDE_MIN}min")

# ── run filter ─────────────────────────────────────────────────────────
config = {
    "mode": "simple", "cbh_strategy": "largest",
    "spatial_filter_enabled": False,
    "min_cluster_size_for_tracking": 5,
    "process_noise": {"sigma_h_accel": 5.0},
    "measurement_noise_scale": 50.0,
    "ri_weights": {"competition": 0.30, "dominance": 0.25, "precision": 0.20, "cardinality": 0.15, "spatial": 0.10},
}
tf = TemporalFilter(config, stereo)
filtered_series, ri_history = [], []
for ts, clusters, kp1, labels, shifts in algo_frames_raw:
    cbh = tf.process_frame(clusters, ts.timestamp(), kp1, labels, shifts)
    ri = tf.get_last_tqi()  # uses same history buffer
    filtered_series.append((ts, cbh, ri))
    ri_history.append(ri)
    if len(filtered_series) % 60 == 0:
        print(f"  processed {len(filtered_series)}/{len(algo_frames_raw)}")

raw_series = []
for ts, clusters, _, _, _ in algo_frames_raw:
    sc = sorted(clusters, key=lambda c: c.size, reverse=True)
    raw_series.append((ts, sc[0].h))

# ── window analysis ────────────────────────────────────────────────────
dw = timedelta(minutes=WINDOW_MIN); ds = timedelta(minutes=SLIDE_MIN)
ceilo_ts_arr = np.array([t.timestamp() for t, _ in ceilo])
ceilo_h_arr = np.array([h for _, h in ceilo])
current = algo_t0; end = algo_t1

window_results = []
all_stitched_t, all_stitched_ce, all_stitched_al = [], [], []
current_ris = []  # per-frame RI within current window for better stats

while current + dw <= end:
    nxt = current + dw
    f_slice = [(ts, cbh, ri) for ts, cbh, ri in filtered_series
               if cbh is not None and current <= ts < nxt]
    r_slice = [(ts, h) for ts, h in raw_series if current <= ts < nxt]
    c_mask = (ceilo_ts_arr >= current.timestamp()) & (ceilo_ts_arr < nxt.timestamp())
    c_sts, c_sh = ceilo_ts_arr[c_mask], ceilo_h_arr[c_mask]
    if len(f_slice) < 5 or len(c_sts) < 5: current += ds; continue

    f_ts = np.array([ts.timestamp() for ts, _, _ in f_slice])
    f_h = np.array([cbh for _, cbh, _ in f_slice])
    ri_vals = np.array([ri for _, _, ri in f_slice if ri is not None])
    r_ts = np.array([ts.timestamp() for ts, _ in r_slice])
    r_h = np.array([h for _, h in r_slice])

    best_r_filt, best_s_filt = -2, 0; best_r_raw, best_s_raw = -2, 0
    for s in range(-300, 301, 10):
        pa, pc = [], []
        for i, t in enumerate(f_ts):
            idx = np.searchsorted(c_sts, t + s); idx = np.clip(idx, 0, len(c_sts)-1)
            if abs(c_sts[idx] - (t + s)) <= 10: pa.append(f_h[i]); pc.append(c_sh[idx])
        if len(pa) >= 5:
            r = float(np.corrcoef(pc, pa)[0, 1])
            if r > best_r_filt: best_r_filt, best_s_filt = r, s
        pa2, pc2 = [], []
        for i, t in enumerate(r_ts):
            idx = np.searchsorted(c_sts, t + s); idx = np.clip(idx, 0, len(c_sts)-1)
            if abs(c_sts[idx] - (t + s)) <= 10: pa2.append(r_h[i]); pc2.append(c_sh[idx])
        if len(pa2) >= 5:
            r2 = float(np.corrcoef(pc2, pa2)[0, 1])
            if r2 > best_r_raw: best_r_raw, best_s_raw = r2, s
    if best_r_filt <= -1: current += ds; continue

    mean_ri = float(np.mean(ri_vals)) if len(ri_vals) > 0 else 0.0
    gap = abs(np.median(f_h) - np.median(c_sh))
    window_results.append({
        "center": current + dw/2, "r_filt": best_r_filt, "r_raw": best_r_raw,
        "shift": best_s_filt, "mean_ri": mean_ri, "gap": gap,
        "n_algo": len(f_slice), "n_ceilo": len(c_sts),
        "algo_mean": float(np.mean(f_h)), "ceilo_mean": float(np.mean(c_sh)),
    })
    # Store per-window pairs for deduplication
    for i, t in enumerate(f_ts):
        idx = np.searchsorted(c_sts, t + best_s_filt); idx = np.clip(idx, 0, len(c_sts)-1)
        if abs(c_sts[idx] - (t + best_s_filt)) <= 10:
            all_stitched_t.append(t)
            all_stitched_al.append(f_h[i])
            all_stitched_ce.append(c_sh[idx])
            window_results[-1].setdefault("_algo_ts", []).append(t)
            window_results[-1].setdefault("_algo_h", []).append(f_h[i])
            window_results[-1].setdefault("_ceilo_h", []).append(c_sh[idx])
    current += ds

# ── Deduplicate stitched pairs: for each algo timestamp, keep pair from
#    the window whose center is closest to that timestamp.
if all_stitched_t:
    at_arr = np.array(all_stitched_t)
    centers = np.array([w["center"].timestamp() for w in window_results])
    dedup_t, dedup_al, dedup_ce = [], [], []
    for i, t_raw in enumerate(at_arr):
        # Find which window center is closest to this timestamp
        closest_win = int(np.argmin(np.abs(centers - t_raw)))
        # Check if this timestamp was contributed by that window
        w_ts = np.array(window_results[closest_win].get("_algo_ts", []))
        if len(w_ts) > 0 and np.min(np.abs(w_ts - t_raw)) < 1e-6:
            dedup_t.append(datetime.fromtimestamp(t_raw))
            dedup_al.append(all_stitched_al[i])
            dedup_ce.append(all_stitched_ce[i])
    all_stitched_t, all_stitched_al, all_stitched_ce = dedup_t, dedup_al, dedup_ce

# ── stats ──────────────────────────────────────────────────────────────
r_filts = np.array([w["r_filt"] for w in window_results])
r_raws = np.array([w["r_raw"] for w in window_results])
ri_means = np.array([w["mean_ri"] for w in window_results])
gaps = np.array([w["gap"] for w in window_results])

ri_r_corr = float(np.corrcoef(ri_means, r_filts)[0, 1]) if len(ri_means) >= 3 else 0.0
# Spearman for robustness
ri_r_spearman, _ = sp_stats.spearmanr(ri_means, r_filts) if len(ri_means) >= 3 else (0, 1)

print(f"\n{'='*70}")
print(f"  RESULTS: {WINDOW_MIN}min windows, slide={SLIDE_MIN}min")
print(f"  Windows:           {len(window_results)}")
print(f"  Mean r (filtered): {np.mean(r_filts):.4f}")
print(f"  Mean r (raw):      {np.mean(r_raws):.4f}")
print(f"  r>0.5 windows:     {sum(r_filts>0.5)}/{len(window_results)} ({100*sum(r_filts>0.5)/len(window_results):.0f}%)")
print(f"  Mean RI:           {np.mean(ri_means):.3f}")
print(f"  RI-r Pearson:      {ri_r_corr:.4f}")
print(f"  RI-r Spearman:     {ri_r_spearman:.4f}")
print(f"  Gap-r correlation: {np.corrcoef(gaps, r_filts)[0,1]:.4f}")
print(f"{'='*70}")

# RI classification effectiveness
ri_hi = ri_means >= 0.6; ri_lo = ri_means < 0.4
if sum(ri_hi) > 0 and sum(ri_lo) > 0:
    print(f"\n  RI ≥ 0.6: mean r = {np.mean(r_filts[ri_hi]):.3f} ({sum(ri_hi)} windows)")
    print(f"  RI < 0.4: mean r = {np.mean(r_filts[ri_lo]):.3f} ({sum(ri_lo)} windows)")

# ── PLOT 1: Stitched timeseries ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 13), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})

# Sort stitched data by time
if all_stitched_t:
    order = np.argsort([t.timestamp() for t in all_stitched_t])
    st = [all_stitched_t[i] for i in order]
    sc = [all_stitched_ce[i] for i in order]
    sa = [all_stitched_al[i] for i in order]
    ax1.plot(st, sc, "-", color="black", lw=1.5, alpha=0.8, label="Ceilometer (truncated, per-window aligned)")
    ax1.plot(st, sa, "-", color="#3498db", lw=1.8, alpha=0.95, label="Filtered CBH (per-window optimal shift)")

# Raw overlay (deduplicated too)
raw_pairs = {}
for w in window_results:
    cur = w["center"] - dw/2; nxt = w["center"] + dw/2
    rs = [(ts, h) for ts, h in raw_series if cur <= ts < nxt]
    if len(rs) < 5: continue
    rt_arr = np.array([ts.timestamp() for ts,_ in rs]); rh_arr = np.array([h for _,h in rs])
    cm = (ceilo_ts_arr >= cur.timestamp()) & (ceilo_ts_arr < nxt.timestamp())
    cst, csh2 = ceilo_ts_arr[cm], ceilo_h_arr[cm]
    for i, t_raw in enumerate(rt_arr):
        idx = np.searchsorted(cst, t_raw + w["shift"]); idx = np.clip(idx, 0, len(cst)-1)
        if abs(cst[idx] - (t_raw + w["shift"])) <= 10:
            key = round(t_raw, 0)
            if key not in raw_pairs:
                raw_pairs[key] = (t_raw, rh_arr[i])
if raw_pairs:
    rp_sorted = sorted(raw_pairs.values(), key=lambda x: x[0])
    rp_t = [datetime.fromtimestamp(t) for t,_ in rp_sorted]
    rp_h = [h for _,h in rp_sorted]
    ax1.plot(rp_t, rp_h, "-", color="#e74c3c", lw=0.8, alpha=0.35, label="Raw dominant cluster")

ax1.set_ylabel("Cloud Base Height (m)", fontsize=13)
ax1.set_title(f"Per-Window Optimal-Shift Alignment: {len(window_results)} x {WINDOW_MIN}min windows stitched", fontsize=15)
ax1.legend(loc="upper right", fontsize=10); ax1.grid(True, alpha=0.3)
ax1.set_ylim(1000, 4500)

# Bottom: RI overlaid on r
centers = [w["center"] for w in window_results]
colors_r = ["#2ecc71" if r>0.5 else "#f39c12" if r>0.2 else "#e74c3c" for r in r_filts]
ax2.bar(centers, r_filts, width=0.006, color=colors_r, alpha=0.7, label=f"r per {WINDOW_MIN}min window")
ax2.axhline(0.5, color="#2ecc71", ls="--", lw=1.5); ax2.axhline(0.0, color="red", ls="-", lw=0.5)
ax2_twin = ax2.twinx()
ax2_twin.plot(centers, ri_means, "s-", color="#8e44ad", ms=6, lw=2, label="Mean RI (unsupervised)")
ax2_twin.set_ylabel("Reliability Index (RI)", fontsize=12, color="#8e44ad"); ax2_twin.set_ylim(0, 1.05)
ax2.set_ylabel("Pearson r per window", fontsize=12)
lines1, labels1 = ax2.get_legend_handles_labels(); lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc="lower left", fontsize=9)
ax2.grid(True, alpha=0.3)
nt = sum(r_filts > 0.5)
ax2.text(0.02, 0.95, f"Mean r={np.mean(r_filts):.3f} | r>0.5: {nt}/{len(r_filts)} ({100*nt/len(r_filts):.0f}%)\nRI-r Pearson={ri_r_corr:.3f} Spearman={ri_r_spearman:.3f}",
         transform=ax2.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax2.set_xlabel("Time (HH:MM)", fontsize=12)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); fig.autofmt_xdate(); plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_stitched_windows.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved: 01_stitched_windows.png")

# ── PLOT 2: RI vs r scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors_ri = ["#2ecc71" if r>0.5 else "#f39c12" if r>0.2 else "#e74c3c" for r in r_filts]
ax.scatter(ri_means, r_filts, c=colors_ri, s=120, alpha=0.8, edgecolors="white", linewidth=1.5, zorder=5)
if len(ri_means) >= 3:
    coeffs = np.polyfit(ri_means, r_filts, 1)
    xl = np.linspace(min(ri_means), max(ri_means), 100)
    ax.plot(xl, np.polyval(coeffs, xl), "-", color="#8e44ad", lw=2.5, zorder=4,
            label=f"RI={coeffs[0]:.3f}*r + {coeffs[1]:.3f} (Pearson)")
    # Also fit r as function of RI
    coeffs2 = np.polyfit(r_filts, ri_means, 1)
    ax.plot(np.polyval(coeffs2, xl), xl, "--", color="gray", lw=1, alpha=0.5)

ax.axvline(0.6, color="#2ecc71", ls="--", lw=1.5, alpha=0.5, label="RI=0.6")
ax.axvline(0.4, color="#f39c12", ls="--", lw=1.5, alpha=0.5, label="RI=0.4")
ax.axhline(0.5, color="#2ecc71", ls=":", lw=1.5, alpha=0.5, label="r=0.5")
ax.axhline(0.0, color="red", ls="-", lw=0.5)

# Annotate with window time
for i, w in enumerate(window_results):
    ax.annotate(w["center"].strftime("%H:%M"), (ri_means[i], r_filts[i]),
                fontsize=5, ha="center", va="bottom", alpha=0.6)

ax.set_xlabel("Mean Reliability Index (RI) per window", fontsize=13)
ax.set_ylabel("Pearson r vs Ceilometer", fontsize=13)
ax.set_title(f"RI Validation: RI-r Pearson={ri_r_corr:.3f}, Spearman={ri_r_spearman:.3f}\n{len(window_results)} x {WINDOW_MIN}min windows", fontsize=14)
ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1.05)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_ri_vs_r_scatter.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved: 02_ri_vs_r_scatter.png")

# ── PLOT 3: Per-window small multiples ──────────────────────────────────
n_cols = 4; n_rows = (len(window_results) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4.5, n_rows*2.8))
axes = axes.flatten()
for i, w in enumerate(window_results):
    ax = axes[i]; cur = w["center"] - dw/2; nxt = w["center"] + dw/2
    cm = (ceilo_ts_arr >= cur.timestamp()) & (ceilo_ts_arr < nxt.timestamp())
    ct, ch = ceilo_ts_arr[cm], ceilo_h_arr[cm]
    ax.plot([datetime.fromtimestamp(t) for t in ct], ch, ".", color="black", ms=2, alpha=0.5)
    fs = [(ts, cbh, ri) for ts, cbh, ri in filtered_series if cbh is not None and cur <= ts < nxt]
    if fs:
        ft2 = np.array([ts.timestamp() for ts,_,_ in fs]); fh2 = np.array([cbh for _,cbh,_ in fs])
        ax.plot([datetime.fromtimestamp(t+w["shift"]) for t in ft2], fh2, "-", color="#3498db", lw=1.2)
    qual = "TRUSTED" if w["r_filt"]>0.5 else "WEAK" if w["r_filt"]>0.2 else "BAD"
    ax.set_title(f"{cur.strftime('%H:%M')} r={w['r_filt']:.2f} RI={w['mean_ri']:.2f} {qual}", fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for lbl in ax.get_xticklabels(): lbl.set_fontsize(5)
    for lbl in ax.get_yticklabels(): lbl.set_fontsize(5)
    ax.grid(True, alpha=0.15)
    if i == 0: ax.legend(fontsize=5, loc="upper right")
for j in range(len(window_results), len(axes)): axes[j].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_per_window_small_multiples.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved: 03_per_window_small_multiples.png")

# ── PLOT 4: RI histogram by r quality ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
bins = np.linspace(0, 1, 15)
for label, mask, color in [("r>0.5", r_filts>0.5, "#2ecc71"), ("r≤0.5", r_filts<=0.5, "#e74c3c")]:
    if sum(mask) > 0:
        ax1.hist(ri_means[mask], bins=bins, alpha=0.6, color=color, label=f"{label} (n={sum(mask)})")
ax1.set_xlabel("Reliability Index"); ax1.set_ylabel("Count"); ax1.set_title("RI Distribution by Window Quality"); ax1.legend()
# Right: gap vs r, colored by RI
sc = ax2.scatter(gaps, r_filts, c=ri_means, cmap="RdYlGn", s=80, alpha=0.8, edgecolors="white", linewidth=1)
ax2.set_xlabel("Height gap algo-ceilo (m)"); ax2.set_ylabel("Pearson r")
ax2.set_title("Gap vs r (colored by RI)"); ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax2); cbar.set_label("RI")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_ri_distribution.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved: 04_ri_distribution.png")

# ── table ───────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  {'Time':>7s}  {'r(filt)':>8s}  {'r(raw)':>8s}  {'RI':>6s}  {'Shift':>6s}  {'Gap':>7s}  {'Algo':>6s}  {'Ceilo':>6s}  {'Quality':>12s}")
print(f"  {'-'*90}")
for w in window_results:
    if w['r_filt'] > 0.5: q = "TRUSTED"
    elif w['r_filt'] > 0.2: q = "MODERATE"
    elif w['gap'] > 1500: q = "DIFF-LAYER"
    else: q = "LOW"
    print(f"  {w['center'].strftime('%H:%M'):>7s}  {w['r_filt']:8.3f}  {w['r_raw']:8.3f}  {w['mean_ri']:6.3f}  {w['shift']:+6d}  {w['gap']:7.0f}  {w['algo_mean']:6.0f}  {w['ceilo_mean']:6.0f}  {q:>12s}")
print(f"{'='*100}")
