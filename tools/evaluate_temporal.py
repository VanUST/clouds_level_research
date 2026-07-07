#!/usr/bin/env python3
"""End-to-end evaluation of temporal + spatial filter for CBH estimation.

Loads .npz algorithm logs, runs filter, compares against ceilometer with
constant-shift and dynamic-shift alignment. Generates publication-ready plots
and computes Trust Quality Index (TQI) diagnostics.
"""
import os, sys, re, logging, argparse, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.temporal_filter import TemporalFilter, ClusterMeasurement
from src.camera import StereoCameraSystem
from config import CameraModel

logger = logging.getLogger("evaluate")


# ── logging ────────────────────────────────────────────────────────────
def setup_logging(log_dir, verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level); root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(level)
    sh.setFormatter(logging.Formatter(fmt, "%H:%M:%S")); root.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, "evaluate_temporal.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ── data loading ───────────────────────────────────────────────────────
def load_ceilometer(filepath):
    logger.info("Loading ceilometer: %s", filepath)
    data, pat = [], re.compile(r"^(\d{12})\s+MSK\s+H1\s+=\s+b'(\d+)'")
    if not os.path.exists(filepath): logger.error("File not found: %s", filepath); return []
    with open(filepath) as f:
        for line in f:
            m = pat.match(line)
            if not m: continue
            h = int(m.group(2))
            if h == 9999: continue
            data.append((datetime.datetime.strptime(m.group(1), "%y%m%d%H%M%S"), h))
    logger.info("  Loaded %d ceilometer measurements", len(data))
    return sorted(data, key=lambda x: x[0])


def load_algorithm_logs(log_dir, stereo_system):
    logger.info("Loading algorithm logs from: %s", log_dir)
    if not os.path.exists(log_dir): logger.error("Log dir not found"); return []
    pat, frames = re.compile(r"img-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.npz"), []
    for fname in sorted(os.listdir(log_dir)):
        m = pat.match(fname)
        if not m: continue
        ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        try: d = np.load(os.path.join(log_dir, fname), allow_pickle=True)
        except Exception: continue
        kp1, labels, shifts = d["kp1"], d["labels"], d["shifts"]
        clusters = []
        for label in sorted(set(labels)):
            if label < 0: continue
            mask, n = labels == label, int(np.sum(labels == label))
            if n < 1: continue
            csh, ckp = shifts[mask], kp1[mask]
            dx, ds = float(np.mean(csh[:, 0])), float(np.std(csh[:, 0]))
            try:
                h, _ = stereo_system.compute_distance(dx, 0.0)
                _, err_sem = stereo_system.compute_distance(dx, ds / np.sqrt(max(n, 1)))
            except Exception: h, err_sem = 0.0, 10000.0
            if not np.isfinite(h) or h <= 0: continue
            clusters.append(ClusterMeasurement(
                h=float(h), h_err=float(err_sem),
                u=float(np.mean(ckp[:, 0])), v=float(np.mean(ckp[:, 1])),
                size=n, disp_x=dx, disp_std=ds, cluster_id=int(label)))
        if clusters: frames.append((ts.timestamp(), clusters, kp1, labels, shifts))
    logger.info("  Loaded %d frames", len(frames))
    return sorted(frames, key=lambda x: x[0])


def build_raw_series(frames, strategy="largest"):
    raw = []
    for ts, clusters, _, _, _ in frames:
        if not clusters: continue
        if strategy == "lowest": sc = sorted(clusters, key=lambda c: c.h)
        elif strategy == "lowest_reliable": sc = sorted(clusters, key=lambda c: (c.h, -c.size))
        else: sc = sorted(clusters, key=lambda c: c.size, reverse=True)
        raw.append((ts, sc[0].h))
    return raw


# ── filter runner ──────────────────────────────────────────────────────
def run_filter(frames, config, stereo_system):
    logger.info("Running filter over %d frames...", len(frames))
    tf = TemporalFilter(config, stereo_system)
    results = []
    for ts, clusters, kp1, labels, shifts in frames:
        cbh = tf.process_frame(clusters, ts, kp1, labels, shifts)
        results.append((ts, cbh))
    if config.get("enable_rts_smoother", True) and config.get("mode") != "simple":
        tf.rts_smooth()
        cbh_map = {round(e["timestamp"], 1): e["cbh"] for e in tf.get_cbh_series()}
        results = [(ts, cbh_map.get(round(ts, 1))) for ts, _ in results]
    logger.info("Filter complete.")
    return tf, results


# ── pairing and shift search ───────────────────────────────────────────
def pair_with_ceilometer(algo_series, ceilo_data, shift_s=0.0, tolerance_s=10.0):
    if not algo_series or not ceilo_data: return []
    cts = np.array([t.timestamp() for t, _ in ceilo_data])
    chs = np.array([h for _, h in ceilo_data])
    paired = []
    for a_ts, a_h in algo_series:
        if a_h is None: continue
        idx = np.searchsorted(cts, a_ts + shift_s)
        idx = np.clip(idx, 0, len(cts) - 1)
        if abs(cts[idx] - (a_ts + shift_s)) <= tolerance_s:
            paired.append((a_ts, a_h, float(chs[idx])))
    return paired


def find_best_shift(algo_series, ceilo_data, search_range_s=300, tolerance_s=10.0):
    best_shift, best_r, best_paired = 0, -1.0, []
    for shift in range(-search_range_s, search_range_s + 1, 5):
        paired = pair_with_ceilometer(algo_series, ceilo_data, shift, tolerance_s)
        if len(paired) < 5: continue
        _, ah, ch = zip(*paired)
        r = float(np.corrcoef(ch, ah)[0, 1])
        if r > best_r: best_r, best_shift, best_paired = r, shift, paired
    return best_shift, best_r, best_paired


# ── dynamic window analysis ────────────────────────────────────────────
def compute_dynamic_windows(algo_series, ceilo_data, window_min=20, slide_min=5):
    if len(algo_series) < 10 or len(ceilo_data) < 10:
        return {"pairs": [], "window_data": [], "mean_r": float("nan")}
    cts = np.array([t.timestamp() for t, _ in ceilo_data])
    chs = np.array([h for _, h in ceilo_data])
    ats_sorted = sorted([t for t, h in algo_series if h is not None])
    if not ats_sorted: return {"pairs": [], "window_data": [], "mean_r": float("nan")}
    t0, t_end = ats_sorted[0], ats_sorted[-1]
    dw, ds = window_min * 60, slide_min * 60
    all_pairs, wdata = [], []
    cur = t0
    while cur + dw < t_end:
        nxt = cur + dw
        si = [i for i, (t, h) in enumerate(algo_series) if h is not None and cur <= t < nxt]
        if len(si) < 5: cur += ds; continue
        sts = np.array([algo_series[i][0] for i in si])
        sh = np.array([algo_series[i][1] for i in si])
        ci = np.where((cts >= cur) & (cts < nxt))[0]
        if len(ci) < 5: cur += ds; continue
        scts, sch = cts[ci], chs[ci]
        best_r, best_s = -2, 0
        for s in range(-300, 301, 10):
            pa, pc = [], []
            for i, t in enumerate(sts):
                idx = np.searchsorted(scts, t + s); idx = np.clip(idx, 0, len(scts) - 1)
                if abs(scts[idx] - (t + s)) <= 10: pa.append(sh[i]); pc.append(sch[idx])
            if len(pa) < 5: continue
            r = float(np.corrcoef(pc, pa)[0, 1])
            if r > best_r: best_r, best_s = r, s
        if best_r <= -1: cur += ds; continue
        for i, t in enumerate(sts):
            idx = np.searchsorted(scts, t + best_s); idx = np.clip(idx, 0, len(scts) - 1)
            if abs(scts[idx] - (t + best_s)) <= 10: all_pairs.append((sh[i], sch[idx]))
        wdata.append({"center": cur + dw / 2, "r": best_r, "shift": best_s,
                       "gap": abs(np.median(sh) - np.median(sch)),
                       "algo_mean": float(np.mean(sh)), "ceilo_mean": float(np.mean(sch))})
        cur += ds
    return {"pairs": all_pairs, "window_data": wdata,
            "mean_r": float(np.mean([w["r"] for w in wdata])) if wdata else float("nan")}


# ── metrics ────────────────────────────────────────────────────────────
def compute_metrics(raw_results, filtered_results, ceilo_data, algo_frames):
    raw_p0 = pair_with_ceilometer(raw_results, ceilo_data, 0)
    filt_p0 = pair_with_ceilometer(filtered_results, ceilo_data, 0)
    m = {"n_raw": len(raw_p0), "n_filt": len(filt_p0)}
    # zero-shift
    for label, paired, key in [("raw", raw_p0, "r_raw_0"), ("filt", filt_p0, "r_filt_0")]:
        if len(paired) >= 3:
            _, ah, ch = zip(*paired); m[key] = float(np.corrcoef(ch, ah)[0, 1])
        else: m[key] = float("nan")
    m["r_delta"] = (m["r_filt_0"] - m["r_raw_0"]) if not np.isnan(m.get("r_raw_0", float("nan"))) and not np.isnan(m.get("r_filt_0", float("nan"))) else float("nan")
    # constant shift
    if len(raw_results) >= 10:
        raw_s, raw_r, raw_al = find_best_shift(raw_results, ceilo_data)
        filt_s, filt_r, filt_al = find_best_shift(filtered_results, ceilo_data)
        m["raw_shift_s"], m["filt_shift_s"] = raw_s, filt_s
        m["r_raw_shifted"], m["r_filt_shifted"] = raw_r, filt_r
        m["r_delta_shifted"] = (filt_r - raw_r) if not np.isnan(raw_r) and not np.isnan(filt_r) else float("nan")
        raw_paired, filt_paired = raw_al, filt_al
    else: raw_paired, filt_paired = raw_p0, filt_p0
    # dynamic windows
    dyn = compute_dynamic_windows(filtered_results, ceilo_data, 20, 5)
    m["window_r_mean"] = dyn["mean_r"]
    m["n_windows"] = len(dyn["window_data"])
    n_trusted = sum(1 for w in dyn["window_data"] if w["r"] > 0.5)
    n_diff = sum(1 for w in dyn["window_data"] if w["gap"] > 1500 and w["r"] < 0.3)
    m["window_trusted_pct"] = 100.0 * n_trusted / max(m["n_windows"], 1)
    m["window_diff_layer_pct"] = 100.0 * n_diff / max(m["n_windows"], 1)
    if dyn["pairs"]:
        a_d, c_d = zip(*dyn["pairs"]); m["r_dynamic_pooled"] = float(np.corrcoef(c_d, a_d)[0, 1])
    else: m["r_dynamic_pooled"] = float("nan")
    # RMSE
    for label, paired, key in [("raw", raw_paired, "rmse_raw"), ("filt", filt_paired, "rmse_filt")]:
        if len(paired) >= 1:
            m[key] = float(np.sqrt(np.mean((np.array([h for _, h, _ in paired]) - np.array([c for _, _, c in paired])) ** 2)))
        else: m[key] = float("nan")
    # jitter, autocorr, outlier rate
    for label, series, prefix in [("raw", raw_results, "raw"), ("filt", filtered_results, "filt")]:
        hs = np.array([h for _, h in series if h is not None])
        m[f"jitter_{prefix}"] = float(np.mean(np.abs(np.diff(hs)))) if len(hs) >= 2 else float("nan")
        m[f"autocorr_{prefix}"] = float(np.corrcoef(hs[:-1], hs[1:])[0, 1]) if len(hs) >= 3 else float("nan")
        if len(hs) >= 5:
            sm = np.convolve(hs, np.ones(5) / 5, mode="same")
            res = np.abs(hs - sm); mad = np.median(np.abs(res - np.median(res)))
            m[f"outlier_rate_{prefix}"] = float(np.mean(res > 3 * mad)) if mad > 1e-8 else 0.0
        else: m[f"outlier_rate_{prefix}"] = float("nan")
    return m


# ── plots ──────────────────────────────────────────────────────────────
def plot_timeseries_comparison(output_dir, ceilo_data, raw_results, filtered_results, metrics):
    fig, ax = plt.subplots(figsize=(16, 8))
    if ceilo_data:
        ax.plot([t for t, _ in ceilo_data], [h for _, h in ceilo_data], ".", color="black", ms=3, alpha=0.5, label="Ceilometer")
    if raw_results:
        rt = [datetime.datetime.fromtimestamp(ts) for ts, _ in raw_results if _ is not None]
        rh = [h for _, h in raw_results if h is not None]
        if rt: ax.plot(rt, rh, "-", color="red", alpha=0.4, lw=1, label="Raw (dominant cluster)")
    if filtered_results:
        ft = [datetime.datetime.fromtimestamp(ts) for ts, _ in filtered_results if _ is not None]
        fh = [h for _, h in filtered_results if h is not None]
        if ft: ax.plot(ft, fh, "-", color="blue", lw=1.5, label="Temporal Filter CBH")
    ax.set_title(f"Cloud Base Height: Ceilometer vs Raw (r={metrics.get('r_raw_0',0):.3f}) vs Filtered (r={metrics.get('r_filt_0',0):.3f})", fontsize=14)
    ax.set_xlabel("Time"); ax.set_ylabel("Height (m)"); ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); fig.autofmt_xdate(); plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_timeseries_comparison.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_correlation_scatter(output_dir, ceilo_data, raw_results, filtered_results, metrics):
    rp = pair_with_ceilometer(raw_results, ceilo_data); fp = pair_with_ceilometer(filtered_results, ceilo_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    for ax, paired, label, rv in [(ax1, rp, "Raw", metrics.get("r_raw_0",0)), (ax2, fp, "Temporal Filter", metrics.get("r_filt_0",0))]:
        if not paired: ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes); continue
        _, ah, ch = zip(*paired); ax.scatter(ch, ah, alpha=0.5, s=15)
        lims = [min(min(ch),min(ah))*0.9, max(max(ch),max(ah))*1.1]
        ax.plot(lims, lims, "r--", lw=1, label="y=x"); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Ceilometer (m)"); ax.set_ylabel("Algorithm (m)")
        ax.set_title(f"{label}: r={rv:.4f} (n={len(paired)})", fontsize=12); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Zero-Shift Correlation", fontsize=14); plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_correlation_scatter.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_error_distribution(output_dir, ceilo_data, raw_results, filtered_results):
    rp = pair_with_ceilometer(raw_results, ceilo_data); fp = pair_with_ceilometer(filtered_results, ceilo_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, paired, label in [(ax1, rp, "Raw"), (ax2, fp, "Temporal Filter")]:
        if not paired: ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes); continue
        _, ah, ch = zip(*paired); errs = np.array(ah) - np.array(ch)
        ax.hist(errs, bins=60, color="steelblue", edgecolor="white", alpha=0.8, density=True)
        ax.axvline(0, color="red", ls="--", lw=1.5); ax.axvline(np.mean(errs), color="orange", ls="-", lw=1.5, label=f"Mean={np.mean(errs):.0f}m")
        ax.set_title(f"{label}: Std={np.std(errs):.0f}m, MAE={np.mean(np.abs(errs)):.0f}m", fontsize=11)
        ax.set_xlabel("Error Algo-Ceilo (m)"); ax.set_ylabel("Density"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(output_dir, "03_error_distribution.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_window_correlation_diagnostic(output_dir, ceilo_data, filtered_results, metrics):
    dyn = compute_dynamic_windows(filtered_results, ceilo_data, 20, 5)
    wdata = dyn.get("window_data", [])
    if len(wdata) < 3: return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    centers = [datetime.datetime.fromtimestamp(w["center"]) for w in wdata]
    rs = [w["r"] for w in wdata]; shifts = [w["shift"] for w in wdata]; gaps = [w["gap"] for w in wdata]
    colors = ["#2ecc71" if r>0.5 else "#f39c12" if r>0.2 else "#e74c3c" for r in rs]
    ax1.bar(centers, rs, width=0.002, color=colors, alpha=0.7)
    ax1.axhline(0.5, color="#2ecc71", ls="--", lw=1.5, label="r=0.5 (trusted)")
    ax1.axhline(0.2, color="#f39c12", ls="--", lw=1.5, label="r=0.2 (weak)")
    ax1.axhline(0, color="red", ls="-", lw=0.5)
    for i, (c, r, g) in enumerate(zip(centers, rs, gaps)):
        if r < 0.2 and g > 1500:
            ax1.annotate("DIFF\nLAYER", (c, r), fontsize=6, color="red", ha="center", va="bottom")
    ax1.set_ylabel("Pearson r per 20-min window", fontsize=12)
    ax1.legend(fontsize=8, loc="lower left"); ax1.grid(True, alpha=0.3)
    nt = sum(1 for r in rs if r>0.5)
    ax1.set_title(f"Dynamic-Shift Window Analysis: Mean r={np.mean(rs):.3f}, Trusted={100*nt/len(rs):.0f}% ({nt}/{len(rs)})", fontsize=13)
    ax2.plot(centers, shifts, "o-", color="#8e44ad", ms=3, lw=1); ax2.axhline(0, color="black", ls="-", lw=0.5)
    ax2.set_ylabel("Best Shift (s)", fontsize=12); ax2.set_xlabel("Time (HH:MM)")
    ax2.set_title("Optimal Time Shift per Window", fontsize=13); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); fig.autofmt_xdate(); plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_window_correlation_diagnostic.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_error_timeseries(output_dir, ceilo_data, raw_results, filtered_results):
    rp = pair_with_ceilometer(raw_results, ceilo_data); fp = pair_with_ceilometer(filtered_results, ceilo_data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ax, paired, color in [(ax1, rp, "#e74c3c"), (ax2, fp, "#3498db")]:
        if not paired: ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes); continue
        ts = [datetime.datetime.fromtimestamp(t) for t,_,_ in paired]; errs = [a-c for _,a,c in paired]
        ax.bar(ts, errs, width=0.003, color=color, alpha=0.7); ax.axhline(0, color="black", ls="-", lw=0.5)
        ax.set_ylabel("Error (m)"); ax.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (HH:MM)"); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); fig.autofmt_xdate()
    fig.suptitle("Per-Frame Height Error (Algo - Ceilometer)", fontsize=14); plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_error_timeseries.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_spatial_filter_report(output_dir, filter_inst, metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rejected, total = filter_inst.spatial_reject_count, filter_inst.total_clusters_seen
    accepted = total - rejected
    if total > 0:
        ax1.pie([accepted, rejected], labels=["Accepted","Rejected"], colors=["#2ecc71","#e74c3c"],
                autopct="%1.1f%%", startangle=90, explode=(0,0.05))
        ax1.set_title(f"Spatial Filter: {total} total clusters", fontsize=13)
    else: ax1.text(0.5,0.5,"No data",ha="center",va="center",transform=ax1.transAxes)
    lm = ["Pearson r", "RMSE (m)", "Jitter (m)", "Autocorr"]
    rv = [metrics.get("r_raw_0",0), metrics.get("rmse_raw",0), metrics.get("jitter_raw",0), metrics.get("autocorr_raw",0)]
    fv = [metrics.get("r_filt_0",0), metrics.get("rmse_filt",0), metrics.get("jitter_filt",0), metrics.get("autocorr_filt",0)]
    x = np.arange(len(lm)); w = 0.35
    ax2.bar(x-w/2, rv, w, label="Raw", color="#e74c3c", alpha=0.7)
    ax2.bar(x+w/2, fv, w, label="Filtered", color="#3498db", alpha=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(lm, fontsize=11)
    ax2.set_title("Internal Metrics Comparison", fontsize=13); ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(os.path.join(output_dir, "06_spatial_filter_report.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_tqi_diagnostic(output_dir, filter_inst, filtered_results, ceilo_data):
    tqis = filter_inst._tqi_history
    if not tqis or len(tqis) < 2: return
    ts_list = [datetime.datetime.fromtimestamp(t) for t, _ in filtered_results if _ is not None]
    n = min(len(tqis), len(ts_list)); tqis = tqis[:n]; ts_list = ts_list[:n]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    colors = ["#2ecc71" if q>=0.7 else "#f39c12" if q>=0.4 else "#e74c3c" for q in tqis]
    ax1.bar(ts_list, tqis, width=0.002, color=colors, alpha=0.8)
    ax1.axhline(0.70, color="#2ecc71", ls="--", lw=2, label="TRUSTED (TQI$\geq$0.7)")
    ax1.axhline(0.40, color="#f39c12", ls="--", lw=2, label="CONDITIONAL (0.4$\leq$TQI<0.7)")
    ax1.set_ylabel("Trust Quality Index", fontsize=12)
    nt = sum(1 for q in tqis if q>=0.7); nc = sum(1 for q in tqis if 0.4<=q<0.7); nu = sum(1 for q in tqis if q<0.4)
    ax1.set_title(f"Trust Quality Index — Unsupervised CBH Confidence (Trusted={100*nt/n:.0f}%, Cond={100*nc/n:.0f}%, Untrusted={100*nu/n:.0f}%)", fontsize=13)
    ax1.legend(loc="upper right", fontsize=10); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)
    paired = pair_with_ceilometer(filtered_results[:n], ceilo_data)
    if paired:
        p_ts = [datetime.datetime.fromtimestamp(t) for t, _, _ in paired]
        errs = [abs(a-c) for _, a, c in paired]
        ax2.plot(p_ts, errs, ".", color="#3498db", ms=3, alpha=0.6)
        ax2.set_ylabel("|Algo - Ceilo| (m)", fontsize=12)
        ax2.set_title("Absolute Error vs Ceilometer (ground truth)", fontsize=12)
    ax2.set_xlabel("Time (HH:MM)"); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); fig.autofmt_xdate(); plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_tqi_diagnostic.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal + spatial filter for CBH estimation")
    parser.add_argument("--algo-dir", default="logs/250523/kornia_loftr")
    parser.add_argument("--ceilo-path", default="data/height data/Zve_Sci_250523.txt")
    parser.add_argument("--output", default="report_temporal")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--no-rts", action="store_true")
    parser.add_argument("--hours", default="12-17")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    setup_logging(args.output, args.verbose)
    logger.info("=" * 60); logger.info("TEMPORAL FILTER EVALUATION"); logger.info("=" * 60)
    h_start, h_end = args.hours.split("-")
    logger.info("Time window: %s:00 - %s:00", h_start, h_end)

    config = {
        "mode": "simple", "cbh_strategy": "largest",
        "spatial_filter_enabled": not args.no_spatial,
        "spatial_dbscan_eps": 30.0, "spatial_dbscan_min_samples": 3,
        "spatial_density_min": 0.002, "spatial_inconsistency_max": 0.30,
        "spatial_k_neighbors": 5, "spatial_disparity_std_mult": 2.0,
        "max_height_rate": 200.0, "max_apparent_speed": 1000.0,
        "min_size_ratio": 0.02, "max_size_ratio": 50.0,
        "min_cluster_size_for_tracking": 8, "max_clusters_per_frame": 5,
        "cluster_merge_height_tol_m": 300,
        "process_noise": {"sigma_h_accel": 5.0, "sigma_u_accel": 5.0, "sigma_v_accel": 5.0, "sigma_s_walk": 10.0},
        "measurement_noise_scale": 50.0,
        "centroid_uncertainty_scale": 2.0, "image_width": 320,
        "chi2_gate": 25.0, "innovation_outlier_thresh": 8.0,
        "max_consecutive_outliers": 10, "min_track_length": 2,
        "min_points_for_cbh": 10, "quality_threshold": 0.15,
        "quality_weights": {"w_len": 0.3, "w_innov": 0.4, "w_size": 0.3},
        "quality_max_len": 50, "quality_innov_max": 15.0, "quality_size_max": 200,
        "enable_rts_smoother": not args.no_rts, "smooth_window": 5,
        "tqi_window": 5,
        "ri_weights": {"competition": 0.30, "dominance": 0.25, "precision": 0.20, "cardinality": 0.15, "spatial": 0.10},
    }

    stereo = StereoCameraSystem(base=45.0, angle_of_view=180.0, image_width=1920, model=CameraModel.FISHEYE)
    ceilo_data = load_ceilometer(args.ceilo_path)
    algo_frames = load_algorithm_logs(args.algo_dir, stereo)
    if not algo_frames: logger.error("No algorithm frames loaded."); return
    if args.stride > 1: algo_frames = algo_frames[::args.stride]; logger.info("Stride %d: %d frames", args.stride, len(algo_frames))
    h_start_dt = datetime.datetime.strptime(h_start.lstrip("0") or "0", "%H")
    h_end_dt = datetime.datetime.strptime(h_end.lstrip("0") or "0", "%H")
    af = [(ts, cl, kp, lb, sh) for ts, cl, kp, lb, sh in algo_frames if h_start_dt.hour <= datetime.datetime.fromtimestamp(ts).hour <= h_end_dt.hour]
    algo_frames = af; logger.info("After time filter: %d frames", len(algo_frames))

    raw_results = build_raw_series(algo_frames, config.get("cbh_strategy", "largest"))
    filter_inst, filtered_results = run_filter(algo_frames, config, stereo)
    metrics = compute_metrics(raw_results, filtered_results, ceilo_data, algo_frames)

    w_mean = metrics.get("window_r_mean", float("nan"))
    w_trust = metrics.get("window_trusted_pct", float("nan"))
    w_diff = metrics.get("window_diff_layer_pct", float("nan"))
    print(f"\n{'='*70}"); print(f"  EVALUATION RESULTS"); print(f"{'='*70}")
    print(f"  Frames: {len(algo_frames)} raw, {len([r for _,r in filtered_results if r is not None])} filtered")
    print(f"  {'─'*40}")
    print(f"  --- Global Constant Shift ---")
    print(f"  r (raw, zero-shift):      {metrics.get('r_raw_0',0):.4f}")
    print(f"  r (filt, zero-shift):     {metrics.get('r_filt_0',0):.4f}")
    print(f"  r (raw, best-shift):      {metrics.get('r_raw_shifted',0):.4f}")
    print(f"  r (filt, best-shift):     {metrics.get('r_filt_shifted',0):.4f}")
    print(f"  {'─'*40}")
    print(f"  --- Dynamic Shift (20-min windows) ---")
    print(f"  Mean window r:            {w_mean:.4f}")
    print(f"  Trusted windows (r>0.5):  {w_trust:.0f}%")
    print(f"  Different-layer windows:  {w_diff:.0f}%")
    print(f"  Pooled dynamic-shift r:   {metrics.get('r_dynamic_pooled',0):.4f}")
    print(f"  {'─'*40}")
    print(f"  --- Internal Consistency ---")
    print(f"  Jitter raw:               {metrics.get('jitter_raw',0):.0f} m")
    print(f"  Jitter filtered:          {metrics.get('jitter_filt',0):.0f} m")
    print(f"  Autocorr raw:             {metrics.get('autocorr_raw',0):.4f}")
    print(f"  Autocorr filtered:        {metrics.get('autocorr_filt',0):.4f}")
    print(f"  {'─'*40}")
    print(f"  Spatial rejected:         {filter_inst.spatial_reject_count}/{filter_inst.total_clusters_seen}")
    if not np.isnan(w_diff) and w_diff > 5:
        print(f"  {'─'*40}")
        print(f"  NOTE: {w_diff:.0f}% of windows see DIFFERENT cloud layers than")
        print(f"  ceilometer. Full 5-h r is low because constant shift can't align")
        print(f"  data across changing wind + different cloud decks.")
    print(f"{'='*70}")
    if config.get("mode") == "simple": print(f"  Mode: Simple 1D Kalman on {config.get('cbh_strategy','largest')} cluster\n{'='*70}")

    logger.info("Generating plots...")
    plot_timeseries_comparison(args.output, ceilo_data, raw_results, filtered_results, metrics)
    plot_correlation_scatter(args.output, ceilo_data, raw_results, filtered_results, metrics)
    plot_error_distribution(args.output, ceilo_data, raw_results, filtered_results)
    plot_window_correlation_diagnostic(args.output, ceilo_data, filtered_results, metrics)
    plot_error_timeseries(args.output, ceilo_data, raw_results, filtered_results)
    plot_spatial_filter_report(args.output, filter_inst, metrics)
    plot_tqi_diagnostic(args.output, filter_inst, filtered_results, ceilo_data)
    logger.info("All plots saved to: %s", args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
