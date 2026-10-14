# PURPOSE: Build the findings report: generate the charts that carry the evidence
#          and assemble them, with the photographic figures, into one
#          self-contained HTML document (images embedded, no external assets).
# INPUTS: the JSON artefacts under experiments/w1, the correspondence cache, and
#         the figures under experiments/audit.
# OUTPUTS: docs/report/findings.html and the chart PNGs beside it.
# KEYWORDS: report, visualisation, findings, html
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
W1 = os.path.join(REPO, "experiments", "w1")
AUDIT = os.path.join(REPO, "experiments", "audit")
OUT = os.path.join(REPO, "docs", "report")
XI0 = 1920.0
F_ASSUMED = XI0 / math.pi
B = 45.0

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.autolayout": True,
})
INK = "#1b2430"
ACC = "#c1440e"
BLUE = "#1f6feb"
GOOD = "#1a7f37"
GREY = "#8b949e"


def b64(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


# 1 --- geometry: why calibration precision is the binding constraint -----------
def chart_geometry():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    D = np.linspace(300, 4000, 400)
    disp = F_ASSUMED * B / D
    ax1.plot(D, disp, color=INK, lw=2)
    for d0, lbl in ((1000, "1 km"), (1500, "1.5 km"), (3000, "3 km")):
        dd = F_ASSUMED * B / d0
        ax1.plot([d0], [dd], "o", color=ACC, ms=6)
        ax1.annotate(f"{lbl}\n{dd:.1f} px", (d0, dd), textcoords="offset points",
                     xytext=(8, 8), fontsize=8, color=ACC)
    ax1.set_xlabel("cloud base height (m)")
    ax1.set_ylabel("stereo disparity (px)")
    ax1.set_title("Disparity shrinks with height")
    ax1.set_ylim(0, 60)

    pct = 100.0 / disp            # % height error per 1 px of disparity error
    ax2.plot(D, pct, color=ACC, lw=2)
    ax2.axhline(10, color=GREY, ls="--", lw=1)
    ax2.annotate("10 % height error", (3000, 10), textcoords="offset points",
                 xytext=(-4, 6), fontsize=8, color=GREY, ha="right")
    ax2.set_xlabel("cloud base height (m)")
    ax2.set_ylabel("height error per 1 px (%)")
    ax2.set_title("One pixel of calibration error, in metres")
    ax2.set_ylim(0, 40)
    return save(fig, "chart_geometry.png")


# 2 --- f: the sun-radius fits --------------------------------------------------
def chart_f():
    rows = json.load(open(os.path.join(W1, "f_estimates.json")))
    fig, ax = plt.subplots(figsize=(9.4, 3.3))
    labels = [f"{r['date'][5:]}\ndev{r['dev']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fb = [r["f_bias"] for r in rows]
    ft = [r["f_through_origin"] for r in rows]
    ax.bar(x - w / 2, fb, w, label="free-intercept fit  (f, + radial bias)", color=GREY)
    ax.bar(x + w / 2, ft, w, label="through-origin fit  (assumes centroid unbiased)",
           color=BLUE)
    ax.axhline(F_ASSUMED, color=ACC, lw=2, ls="--",
               label=f"assumed  ξ₀/π = {F_ASSUMED:.0f}")
    for xi, v in zip(x + w / 2, ft):
        ax.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=8, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("f  (px / rad)")
    ax.set_ylim(0, 760)
    ax.set_title("Focal constant measured four independent ways — the two estimators disagree systematically")
    ax.legend(fontsize=7.5, loc="upper left")
    return save(fig, "chart_f.png")


# 3 --- roll: the disparity distribution before and after -----------------------
def chart_roll_hist():
    d = np.load(os.path.join(W1, "matches_cache_2025-05-23.npz"), allow_pickle=True)
    kp1s, kp2s = d["kp1"], d["kp2"]

    def du_for(angle):
        out = []
        for k1, k2 in zip(kp1s, kp2s):
            a = math.radians(angle)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            k2r = (k2 - 960.0) @ R.T + 960.0
            out.append(k1[:, 0] - k2r[:, 0])
        return np.concatenate(out)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.4))
    du0 = du_for(0.0)
    du1 = du_for(1.5)
    bins = np.linspace(-5, 40, 120)
    for ax, du, t in ((ax1, du0, "as shipped"), (ax2, du1, "with +1.5° roll correction")):
        q1, q3 = np.percentile(du, [25, 75])
        ax.hist(du, bins=bins, color=ACC if ax is ax1 else GOOD, alpha=0.85)
        ax.axvspan(q1, q3, color=INK, alpha=0.10)
        ax.set_xlabel("horizontal disparity  (px)")
        ax.set_ylabel("matches")
        ax.set_title(f"{t}\nIQR = {q3-q1:.2f} px")
        ax.set_xlim(-5, 40)
    fig.suptitle("Applying the measured roll collapses the disparity spread  "
                 "(38 frames, 79,982 correspondences, 2025-05-23)",
                 fontsize=10, fontweight="bold", y=1.03)
    return save(fig, "chart_roll_hist.png")


# 4 --- roll: the sweep curve ---------------------------------------------------
def chart_roll_sweep():
    d = np.load(os.path.join(W1, "matches_cache_2025-05-23.npz"), allow_pickle=True)
    kp1s, kp2s = d["kp1"], d["kp2"]
    angles = np.arange(-4, 4.01, 0.25)
    iqrs = []
    for ang in angles:
        a = math.radians(ang)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        du = np.concatenate([k1[:, 0] - ((k2 - 960.0) @ R.T + 960.0)[:, 0]
                             for k1, k2 in zip(kp1s, kp2s)])
        q1, q3 = np.percentile(du, [25, 75])
        iqrs.append(q3 - q1)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(angles, iqrs, color=INK, lw=2)
    ibest = int(np.argmin(iqrs))
    ax.plot([angles[ibest]], [iqrs[ibest]], "o", color=GOOD, ms=8)
    ax.annotate(f"optimum {angles[ibest]:+.2f}°\nIQR {iqrs[ibest]:.2f} px",
                (angles[ibest], iqrs[ibest]), textcoords="offset points",
                xytext=(12, 18), fontsize=8, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD))
    ax.axvline(0, color=ACC, lw=1.5, ls="--")
    ax.annotate("as shipped", (0, max(iqrs) * 0.92), fontsize=8, color=ACC,
                ha="right", rotation=90, textcoords="offset points", xytext=(-4, 0))
    ax.set_xlabel("roll applied to camera 2 (deg)")
    ax.set_ylabel("disparity IQR (px)")
    ax.set_title("The optimum is sharp and its sign is unambiguous")
    return save(fig, "chart_roll_sweep.png")


# 5 --- session usability -------------------------------------------------------
def chart_sessions():
    hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    d22 = [12, 13, 2, 2, 4, 1, 14, 12, 0, 1919]
    d23 = [1104, 1867, 3002, 183, 3719, 1088, 1067, 2991, 1442, 1553]
    fig, ax = plt.subplots(figsize=(9.4, 3.1))
    x = np.arange(len(hours))
    ax.bar(x - 0.2, d22, 0.4, label="2025-05-22", color=ACC)
    ax.bar(x + 0.2, d23, 0.4, label="2025-05-23", color=BLUE)
    ax.axhline(100, color=GREY, ls="--", lw=1)
    ax.annotate("below ~100 correspondences there is no usable stereo signal",
                (0.1, 100), textcoords="offset points", xytext=(0, 6),
                fontsize=8, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours])
    ax.set_xlabel("hour (UTC)")
    ax.set_ylabel("LoFTR correspondences")
    ax.set_title("2025-05-22 is cloud-free — the project has one usable session")
    ax.legend()
    return save(fig, "chart_sessions.png")


# 5b --- performance -------------------------------------------------------------
def chart_perf():
    stages = ["JPEG decode", "rectify + crop", "LoFTR matching", "HDBSCAN clustering"]
    ms = [51.0, 3.5, 168.4, 185.5]
    cols = [GREY, GREY, BLUE, ACC]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(stages))
    ax1.barh(y, ms, color=cols)
    for i, v in enumerate(ms):
        ax1.text(v + 4, i, f"{v:.0f} ms  ({100*v/sum(ms):.0f} %)", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(stages)
    ax1.invert_yaxis()
    ax1.set_xlabel("ms per stereo pair")
    ax1.set_xlim(0, 260)
    ax1.set_title("Where one pair's 408 ms goes")
    names = ["as shipped\n(batch 1)", "matcher\nbatched 1.70x", "batch +\nparallel decode"]
    fps = [2.45, 2.98, 3.27]
    ax2.bar(names, fps, color=[GREY, BLUE, GOOD])
    for i, v in enumerate(fps):
        ax2.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(1.65, color=ACC, ls="--", lw=1.5)
    ax2.annotate("measured end-to-end,\nunbatched", (2.35, 1.65),
                 textcoords="offset points", xytext=(-8, -30), fontsize=7.5,
                 color=ACC, ha="right")
    ax2.set_ylabel("frames per second")
    ax2.set_ylim(0, 3.8)
    ax2.set_title("Projected gains, and why batching alone failed")
    return save(fig, "chart_perf.png")


# 6 --- infinity test -----------------------------------------------------------
def chart_infinity():
    rows = json.load(open(os.path.join(W1, "infinity_test.json")))
    z = np.array([r.get("zen", r.get("sun_zenith_deg")) for r in rows])
    m = np.array([r["mag"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.scatter(z, m, color=ACC, s=28, zorder=3)
    ax.set_xlabel("solar zenith angle (deg)")
    ax.set_ylabel("infinity residual (px)")
    ax.set_title("Sun must rectify to zero — residual vs zenith angle")
    ax.axhline(0, color=GREY, lw=1)
    for zi, mi in zip(z, m):
        if mi > 100:
            ax.annotate("blob clipped by\nthe image circle",
                        (zi, mi), textcoords="offset points", xytext=(-70, -6),
                        fontsize=7.5, color=GREY,
                        arrowprops=dict(arrowstyle="->", color=GREY))
            break
    return save(fig, "chart_infinity.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    g = chart_geometry()
    cf = chart_f()
    rh = chart_roll_hist()
    rs = chart_roll_sweep()
    cs = chart_sessions()
    ci = chart_infinity()
    pf = chart_perf()
    cmp_imgs = {}
    for key, fn in (("cmp_ts", "comparison_timeseries.png"),
                    ("cmp_m", "comparison_metrics.png"),
                    ("layers", "ceilometer_layers.png"),
                    ("layerstruct", "layer_structure.png"),
                    ("radial", "height_by_radius.png"),
                    ("distortion", "distortion_fit.png"),
                    ("parexact", "paraxial_vs_exact.png")):
        fp = os.path.join(OUT, fn)
        if os.path.exists(fp):
            cmp_imgs[key] = b64(fp)
    sheets = {
        "sun": b64(os.path.join(AUDIT, "sun_contact_sheet.png")),
        "sessions": b64(os.path.join(AUDIT, "session_compare.png")),
        "fov": b64(os.path.join(AUDIT, "probe_full_frame_grid.png")),
    }
    imgs = {k: b64(os.path.join(OUT, v)) for k, v in
            dict(geometry=g, f=cf, roll_hist=rh, roll_sweep=rs,
                 sessions_chart=cs, infinity=ci, perf=pf).items()}
    html = build_html(imgs, sheets, cmp_imgs)
    path = os.path.join(OUT, "findings.html")
    open(path, "w").write(html)
    print(f"wrote {path}  ({len(html)/1e6:.1f} MB)")


def build_html(imgs, sheets, cmp_imgs=None):
    cmp_imgs = cmp_imgs or {}
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Cloud base height — findings</title>
<style>
 :root {{ --ink:#1b2430; --acc:#c1440e; --blue:#1f6feb; --good:#1a7f37;
         --grey:#5c6672; --line:#e3e6ea; --bg:#fbfbfc; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
        font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1080px; margin:0 auto; padding:0 28px 96px; }}
 header {{ padding:56px 0 28px; border-bottom:3px solid var(--ink); margin-bottom:8px; }}
 h1 {{ font-size:34px; line-height:1.2; margin:0 0 10px; letter-spacing:-.5px; }}
 .sub {{ color:var(--grey); font-size:15px; }}
 h2 {{ font-size:23px; margin:52px 0 6px; letter-spacing:-.3px; }}
 h2 .n {{ color:var(--acc); font-variant-numeric:tabular-nums; margin-right:10px; }}
 h3 {{ font-size:16px; margin:26px 0 4px; }}
 p, li {{ font-size:15.2px; }}
 figure {{ margin:22px 0; }}
 figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }}
 figcaption {{ color:var(--grey); font-size:13px; margin-top:8px; }}
 table {{ border-collapse:collapse; width:100%; margin:18px 0; font-size:14.5px; }}
 th,td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); }}
 th {{ background:#f2f4f7; font-weight:600; }}
 td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .big {{ display:flex; gap:14px; flex-wrap:wrap; margin:24px 0; }}
 .kpi {{ flex:1 1 190px; background:#fff; border:1px solid var(--line);
         border-left:4px solid var(--acc); border-radius:6px; padding:14px 16px; }}
 .kpi.good {{ border-left-color:var(--good); }}
 .kpi .v {{ font-size:25px; font-weight:700; font-variant-numeric:tabular-nums; }}
 .kpi .l {{ color:var(--grey); font-size:13px; margin-top:2px; }}
 .note {{ background:#fff7ed; border-left:4px solid var(--acc); padding:12px 16px;
          border-radius:0 6px 6px 0; margin:20px 0; font-size:14.6px; }}
 .neg {{ background:#f0f6ff; border-left:4px solid var(--blue); }}
 blockquote {{ margin:20px 0; padding:2px 0 2px 18px; border-left:4px solid var(--ink);
               font-size:16px; }}
 code {{ background:#f2f4f7; padding:1px 5px; border-radius:3px; font-size:13.5px; }}
 .toc {{ columns:2; font-size:14.6px; }}
 footer {{ margin-top:64px; padding-top:20px; border-top:1px solid var(--line);
           color:var(--grey); font-size:13.5px; }}
</style></head><body><div class="wrap">

<header>
  <h1>Cloud base height from a stereo fisheye pair</h1>
  <div class="sub">Findings from the step-one reassessment &middot; 2025-05-23 session
  &middot; rig: two 180&deg; fisheye cameras, 45 m baseline, 1920&times;1920</div>
</header>

<div class="toc">
  <a href="#s1">1. The constraint that governs everything</a><br>
  <a href="#s2">2. The focal constant is not what the code assumes</a><br>
  <a href="#s3">3. The infinity test, and why it is not enough</a><br>
  <a href="#s4">4. The calibration error, measured</a><br>
  <a href="#s5">5. Why this explains the fragmentation</a><br>
  <a href="#s6">6. One usable session, not two</a><br>
  <a href="#s7">7. What was ruled out</a><br>
  <a href="#s8">8. Where this leaves the algorithm</a><br>
  <a href="#s9">9. Inference cost, and a wrong optimisation</a><br>
  <a href="#s10">10. Corrections applied, and what they are worth</a><br>
  <a href="#s11">11. Temporal alignment: the lag is not the main problem</a><br>
  <a href="#s12">12. The layer hypothesis: what holds and what does not</a><br>
  <a href="#s13">13. The radial trend is lens distortion</a>
</div>

<h2 id="s1"><span class="n">1</span>The constraint that governs everything</h2>
<p>The rig turns a tiny horizontal shift between the two images into a height:
<code>D = f&middot;B/&Delta;&xi;</code>, with the baseline <code>B = 45 m</code>. Because
that baseline is short, the shift is small, and a fixed error in pixels becomes a
large error in metres. This is a property of the hardware, not of the algorithm.</p>
<figure><img src="{imgs['geometry']}" alt="disparity and sensitivity vs height"></figure>
<div class="big">
  <div class="kpi"><div class="v">18.3 px</div><div class="l">disparity at 1.5 km</div></div>
  <div class="kpi"><div class="v">5.5 %</div><div class="l">height error at 1.5 km, per 1 px of calibration error</div></div>
  <div class="kpi"><div class="v">10.9 %</div><div class="l">the same, at 3 km</div></div>
  <div class="kpi"><div class="v">0.18 px</div><div class="l">calibration accuracy needed for 1 % at 1.5 km</div></div>
</div>
<div class="note">The written work quotes &ldquo;&asymp;30 m per pixel&rdquo;. That is
true only near 550 m. At 1.5 km it is 82 m/px and at 3 km, 327 m/px — the sensitivity
claim understates the problem by about 3&times; at typical heights. This single fact
sets the accuracy budget for the whole project.</div>

<h2 id="s2"><span class="n">2</span>The focal constant is not what the code assumes</h2>
<p>The pipeline computes <code>f = &xi;<sub>0</sub>/&pi; = 611 px/rad</code>, i.e. it
<em>assumes</em> the 180&deg; field maps exactly onto the circle inscribed in the
frame. Nothing measures it. Two independent lines of evidence say it is too small:</p>
<ul>
<li>the radius at which the frame actually goes black is <strong>1078&ndash;1156 px</strong>,
not the 960 px that assumption implies, and it is soft (fading between about 1000 and
1150 px);</li>
<li>the Sun's image radius against its known zenith angle, fitted through four
independent streams.</li>
</ul>
<figure><img src="{imgs['f']}" alt="f estimates"></figure>
<p>The two estimators on the <em>same data</em> disagree, and that disagreement is the
result: <code>f</code> and a radial bias of the blob centroid are degenerate over this
zenith range. The free-intercept fit says 597&nbsp;px/rad with a 68 px outward bias;
forcing the bias to zero says 664. The four through-origin estimates are far more
consistent with each other (spread 3.5 %) than the four biased ones (9 %), which is
why they are the more credible set.</p>
<div class="big">
  <div class="kpi"><div class="v">&asymp;654 &plusmn; 11</div><div class="l">f, px/rad (through-origin, 4 streams)</div></div>
  <div class="kpi"><div class="v">1.070 &plusmn; 0.018</div><div class="l">ratio to the assumed &xi;<sub>0</sub>/&pi;</div></div>
  <div class="kpi"><div class="v">6&ndash;8 %</div><div class="l">implied downward bias of every height</div></div>
</div>
<p>This bias is purely multiplicative and independent of cloud height. It therefore does
<em>not</em> distort dynamics — which is exactly the pattern seen in the data: the
estimated series tracks the ceilometer's motion while sitting systematically below it.</p>

<h2 id="s3"><span class="n">3</span>The infinity test, and why it is not enough</h2>
<p>The Sun is the only object at infinity in these images, so its two images must
coincide after rectification. A solar ephemeris plus <em>one constant</em> orientation
offset predicts the Sun's image azimuth to better than 0.5&deg; across 14 hours
&mdash; which independently confirms the filenames are UTC, and gives the rig's
azimuth orientation, which the calibration matrix does not record.</p>
<figure><img src="{sheets['sun']}" alt="sun detected across the day with radius rings"></figure>
<figcaption>Left to right, 04:00&ndash;18:00 UTC. The red ring is the Sun's predicted
radius from the ephemeris: it visibly grows and shrinks correctly across the day. The
Sun is found as the saturated region nearest the predicted azimuth.</figcaption>
<figure><img src="{imgs['infinity']}" alt="infinity residual vs zenith"></figure>
<p>But the residual is <strong>4.3&ndash;175 px, median 24.7 px</strong>, and it must
<em>not</em> be read as calibration error: the detected point is the centroid of a
saturated region far larger than the ~6 px solar disc, and that region changes shape
between the two views. The residual blows up near the horizon where the blob is
clipped, and is small and stable in the middle of the day &mdash; a pattern consistent
with a shape artefact. <strong>The sun-blob cannot resolve calibration errors below
about 4&deg;, i.e. ~30 px at the crop corner. The target is ~1 px.</strong></p>

<h2 id="s4"><span class="n">4</span>The calibration error, measured</h2>
<p>Abandoning the infinity reference and using the images themselves: fit the disparity
field of each frame with the exact geometry plus the calibration terms, and let the
scene have a plane height, a roll of camera 2, and a translation. No Sun, no ground
truth.</p>
<table>
<tr><th>model</th><th class="num">median rms residual</th></tr>
<tr><td>D only</td><td class="num">13.18 px</td></tr>
<tr><td>D + translation</td><td class="num">8.46 px</td></tr>
<tr><td><strong>D + translation + roll</strong></td><td class="num"><strong>2.64 px</strong></td></tr>
</table>
<p>Over 38 frames the fitted roll is <strong>&minus;1.57&deg; &plusmn; 0.10&deg;</strong>
&mdash; stable to a tenth of a degree over an hour. This also settles the earlier
puzzle: the Sun-blob azimuths had suggested ~4&deg;, and they were wrong because of the
blob-shape bias, exactly as suspected.</p>

<h3>Making the prediction falsifiable, then testing it</h3>
<p>A roll about the principal point displaces points <em>tangentially</em>, so its
horizontal component is <code>&delta;&middot;&Delta;v</code> and it <strong>changes sign
across the centre</strong>. The prediction was: correcting it should collapse the
disparity spread, and the wrong sign should widen it.</p>
<figure><img src="{imgs['roll_hist']}" alt="disparity distribution before and after"></figure>
<figure><img src="{imgs['roll_sweep']}" alt="IQR vs applied roll"></figure>
<div class="big">
  <div class="kpi good"><div class="v">8.14 &rarr; 1.92 px</div><div class="l">disparity IQR from the cached correspondences (38 frames, 4.2&times; tighter)</div></div>
  <div class="kpi good"><div class="v">7.68 &rarr; 1.33 px</div><div class="l">the same correction run through the real pipeline (5.8&times; tighter)</div></div>
  <div class="kpi"><div class="v">&minus;1.0 %</div><div class="l">movement of the median disparity &mdash; level is protected</div></div>
</div>
<p>The wrong sign <em>doubles</em> the spread. The optimum is sharp, and the same
magnitude comes out of three independent methods (the field fit, the spread
minimisation, and the infinity residuals). The median barely moves, which is why the
pipeline's heights looked plausible while its spread was wrong.</p>

<h2 id="s5"><span class="n">5</span>Why this explains the fragmentation</h2>
<p>The original audit found that <code>HDBSCAN</code> with
<code>min_cluster_size = 3</code> produces <strong>478 clusters</strong> on a single
frame and that a median cluster holds only 5 matches, with a 300 m height-merge doing
all the real work. The reason is now visible:</p>
<blockquote>A 1.57&deg; roll swings the horizontal disparity by
<strong>&plusmn;8.8 px</strong> across the crop. For a 1500 m cloud, whose true
disparity is 18.3 px, that spreads the matches from <strong>1014 m to 2883 m</strong>.</blockquote>
<p>Most of the within-frame spread being clustered was calibration error. After the
correction the same 10 frames give a match height span of <strong>1748&ndash;1909 m</strong>
instead of <strong>1442&ndash;2415 m</strong>.</p>
<div class="note neg"><strong>Consequence for the plan:</strong> the layer-extraction
step (W4) cannot succeed before this is corrected. It was segmenting an artefact. The
roll term must be removed, or carried as a known nuisance parameter, before any layer
extraction is meaningful.</div>

<h2 id="s6"><span class="n">6</span>One usable session, not two</h2>
<p>Frame counts suggested two sessions of equal size (17,264 and 17,265 images).
Usability is a different matter: stereo needs cloud texture for the matcher to lock
onto, and <strong>2025-05-22 is cloud-free</strong>.</p>
<figure><img src="{imgs['sessions_chart']}" alt="matches per hour by session"></figure>
<figure><img src="{sheets['sessions']}" alt="clear sky vs cloud on the two days"></figure>
<figure><img src="{sheets['fov']}" alt="full frame with field of view rings"></figure>
<figcaption>The last figure also shows what the 640 px square crop includes: the
fisheye circle is inscribed, the corners fall outside it, and near-field occluders
(trees, building, mast) reach into the crop's corners.</figcaption>
<p>2025-05-22 returns single digits of correspondences at every hour except 17:00 UTC.
This is corroborated independently by the ceilometer &mdash; <strong>610 valid
readings for 250522 against 2,341 for 250523</strong>, which is what a clear day looks
like. There is therefore <strong>one usable session</strong>.</p>
<p>That has consequences which are recorded rather than worked around:</p>
<ul>
<li><strong>session drift cannot be measured</strong> &mdash; there is no second
session to compare against;</li>
<li><strong>splits must be temporal, within one day</strong> &mdash; session-level
hold-out is impossible, so a held-out block is correlated with the tuning block;</li>
<li><strong>validation on an unseen session is not possible</strong> with this data.</li>
</ul>

<h2 id="s7"><span class="n">7</span>What was ruled out</h2>
<p>Negative results are recorded so they are not re-attempted.</p>
<table>
<tr><th>Attempt</th><th>Outcome</th></tr>
<tr><td>Brightest saturated blob as the Sun</td><td>finds saturated <em>cloud</em> (10<sup>4</sup>&ndash;4&times;10<sup>4</sup> px against ~26 px for the disc)</td></tr>
<tr><td>Gradient-convergence glow centring</td><td>fails &mdash; r/&theta; scattered 492&ndash;945; the glow has no consistent centre</td></tr>
<tr><td>Star-field calibration at night</td><td>closed &mdash; the stretched night frames are pure noise; the exposure is far too short</td></tr>
<tr><td>Moon as an angular reference</td><td>closed &mdash; no Moon in the night frames</td></tr>
<tr><td>2025-05-22 as an independent session</td><td>closed &mdash; cloud-free, no stereo signal</td></tr>
</table>
<p>Two defects in my own tooling were also found and fixed, because both produced
plausible-looking wrong numbers: the solar azimuth branch needs the <em>signed</em>
hour angle (folding it into 0&ndash;360&deg; mirrors every morning measurement by
180&deg;), and the calibration matrix must be loaded from the registry rather than
retyped (dropping leading zeros silently invalidated a run of residuals).</p>

<h2 id="s8"><span class="n">8</span>Where this leaves the algorithm</h2>
<div class="big">
  <div class="kpi good"><div class="v">&plusmn;7 %</div><div class="l">height spread for a 1500 m cloud after the roll correction (was &plusmn;27 %)</div></div>
  <div class="kpi"><div class="v">~2 px</div><div class="l">remaining within-frame disparity floor, part of it real cloud relief</div></div>
  <div class="kpi"><div class="v">1.8 %</div><div class="l">height uncertainty contributed by the f measurement alone</div></div>
</div>
<p>Everything here is falsifiable and states its evidence. The three claims that
matter:</p>
<ul>
<li><strong>Dynamics</strong> &mdash; supportable, and the necessary condition for the
algorithm's internal logic being right.</li>
<li><strong>Level</strong> &mdash; carries a measured ~7 % multiplicative bias from
<code>f</code>, correctable; the roll leaves the median nearly untouched.</li>
<li><strong>Spread</strong> &mdash; was dominated by calibration error, and is now
close to the physical floor.</li>
</ul>
<p>The correction is derived entirely from the images, by an internal criterion
(minimising disparity spread, on the physical expectation that a cloud layer has small
intrinsic spread), and never from the ceilometer. It does not change the fixed
calibration; it compensates a measured residual. It must nevertheless be declared as a
measured correction applied on top of that calibration, which is what the configuration
now makes explicit.</p>

<h2 id="s9"><span class="n">9</span>Inference cost, and a wrong optimisation</h2>
<p>Profiling the full evaluation produced one surprise that changed the plan: the
clustering stage costs as much as the matcher.</p>
<figure><img src="{imgs['perf']}" alt="stage breakdown and projected gains"></figure>
<table>
<tr><th>stage</th><th class="num">ms / frame</th><th class="num">share</th></tr>
<tr><td>JPEG decode (two 1920&times;1920)</td><td class="num">51</td><td class="num">12 %</td></tr>
<tr><td>rectify + crop</td><td class="num">3.5</td><td class="num">1 %</td></tr>
<tr><td><strong>LoFTR matching</strong></td><td class="num"><strong>168</strong></td><td class="num"><strong>41 %</strong></td></tr>
<tr><td><strong>HDBSCAN clustering</strong></td><td class="num"><strong>185</strong></td><td class="num"><strong>45 %</strong></td></tr>
</table>
<p>The matcher was batched into single forward passes &mdash; verified equivalent
to 0.0000 px per match, with match counts differing by a few out of ~3700 &mdash;
and it did speed that stage up <strong>1.70&times;</strong>. But end-to-end the
evaluation went from 1.65 fps to about 1.5&ndash;1.6 fps: <strong>no gain</strong>.
The arithmetic explains it: matching is 41 % of the time, so a 1.70&times; matcher
speed-up can improve the total by at most 1.20&times;, and only if the freed GPU
time is used &mdash; which it is not, because the CPU blocks on decoding and
clustering between batches and nothing overlaps.</p>
<div class="note">The GPU sits at <strong>~3 % utilisation</strong>. That single
number says the win is <em>pipelining</em> (prepare batch N+1 while the GPU runs
batch N) and removing the clustering cost &mdash; which is also the stage the plan
already intends to replace, since it is fragmenting calibration error into
hundreds of clusters. Speeding it up and replacing it are the same task.</div>

<h2 id="s10"><span class="n">10</span>Corrections applied, and what they are worth</h2>
<p>Both configurations were run over the full validation window on <strong>identical
code</strong>, differing only in the roll correction, so the difference is
attributable to the correction and nothing else. 1,776 and 1,784 frames scored
against 906 ceilometer readings.</p>
<figure><img src="{cmp_imgs.get('cmp_ts','')}" alt="algorithm and ceilometer height series"></figure>
<figure><img src="{cmp_imgs.get('cmp_m','')}" alt="dynamics, smoothness and level"></figure>
<table>
<tr><th>variant</th><th class="num">median height</th><th class="num">r, best shift</th>
<th class="num">r, fixed shift</th><th class="num">r of frame differences</th>
<th class="num">median jump</th></tr>
<tr><td>as shipped</td><td class="num">1398 m</td><td class="num">0.328</td>
<td class="num">&minus;0.159</td><td class="num">0.048</td><td class="num">27.0 m</td></tr>
<tr><td><strong>roll corrected</strong></td><td class="num">1837 m</td><td class="num">0.181</td>
<td class="num">&minus;0.149</td><td class="num">&minus;0.007</td><td class="num"><strong>5.2 m</strong></td></tr>
<tr><td>roll + f corrected (&times;1.070)</td><td class="num">1966 m</td><td class="num">0.181</td>
<td class="num">&minus;0.149</td><td class="num">&minus;0.007</td><td class="num">5.5 m</td></tr>
</table>

<h3>Three things this says, one of them uncomfortable</h3>
<p><strong>The correction is doing what it should.</strong> Removing the measured
roll makes the series <strong>5&times; smoother</strong> (median frame-to-frame jump
27.0 m &rarr; 5.2 m) and raises the level by 31&nbsp;%, moving it toward the
ceilometer rather than away. Raising the level is the right direction: the roll was
inflating the disparity and so depressing the computed height.</p>

<p><strong>The empirical agreement got worse, and that is worth understanding
rather than hiding.</strong> The best-shift windowed correlation falls from 0.328 to
0.181. Two facts explain it. First, the metric searches a per-window time shift and
takes the largest correlation found; a noisier series gets more chances to find a
flattering alignment, so the metric is biased <em>in favour of</em> noise. Second,
the fixed-shift correlation is <strong>negative for both</strong> versions
(&minus;0.159 and &minus;0.149), and the correlation of frame-to-frame differences
is <strong>~zero for both</strong> (0.048 and &minus;0.007).</p>

<blockquote>Neither the old method nor the corrected one tracks the ceilometer's
dynamics. The positive numbers in the best-shift metric come from the shift search,
not from agreement.</blockquote>

<p><strong>The level does not move with the ceilometer at all.</strong> In the right
panel the estimate is pinned between roughly 1000 and 2500 m whether the ceilometer
reads 1900 m or 4100 m. The algorithm is not failing to be accurate; it is reporting
something else. The likeliest explanation is the one the stage audit already
flagged: the 640 px <em>square</em> crop includes near-field trees, a building and a
mast, and a near-field object at 100 m produces a large disparity that the estimator
faithfully converts into a low cloud. That would place a floor under the estimated
height — exactly the saturating behaviour seen here, and exactly the artefact that a
circular field mask plus an occluder mask exists to remove.</p>

<h3>The lag is real but is not advection</h3>
<p>Cloud motion estimated from the imagery alone (block matching on 1,798 consecutive
pairs, with the translating-plane fisheye model) gives a median speed of
<strong>3.27 m/s</strong> at the measured height (p10&ndash;p90 2.02&ndash;6.03), image
azimuth 213&deg; &plusmn; 9&deg;, smooth over five hours. Validated against static
foreground patches, against phase correlation, and across four high-pass bands.
At that speed the observed 205&ndash;300 s shifts would require a separation of
<strong>700&ndash;1000 m</strong> &mdash; and <strong>the instrument separation is
unrecorded</strong>: there are no site coordinates anywhere in the project, and no
meteorology for 2025-05-23.</p>
<p>More tellingly, a fixed-baseline advection model fitted to the shift sequence
explains <strong>R&sup2; = 0.10</strong> (p = 0.18) and adjacent windows jump by
200&ndash;250 s, while changing only the roll correction &mdash; 2&ndash;3 px &mdash;
moves the best shift by 168 s on average. <strong>A smooth measured wind cannot
produce a shift sequence like that.</strong> The shifts are not advection; they are
the same selection artefact.</p>

<h3>And the sampling geometry cannot be rescued by any shift</h3>
<p>At 1837 m the camera footprint is <strong>1956 m across</strong> against the
ceilometer's ~10 m column &mdash; a ratio of about 200:1 &mdash; spanning 8.5 texture
decorrelation lengths (229 m each) and advancing 9&ndash;45 of them per window. Real
within-frame relief exists (~12 % of the height), so a footprint average cannot equal
a point column. Time-shifting cannot repair a spatial mismatch.</p>

<h3>What a properly-powered test actually finds</h3>
<p>With both series resampled so their samples are independent and no shift search,
the correlation is <strong>negative and significant at the level</strong>
(r = &minus;0.376 on 470 samples, 95&nbsp;% CI [&minus;0.653, &minus;0.262]). An
earlier version of this report also claimed the <em>dynamics</em> were
anti-correlated (r = &minus;0.248). <strong>That was a protocol artefact</strong>:
the differences were taken on a masked grid, and 30 of the 469 spanned a missing
epoch. Restricted to genuinely adjacent samples the value is
<strong>r = &minus;0.045, CI [&minus;0.100, &minus;0.007]</strong> &mdash; nil.
<strong>The failure is in level, not in dynamics.</strong></p>
<p>Two further measurements reframe the problem entirely:</p>
<table>
<tr><th>hour (MSK)</th><th class="num">12</th><th class="num">13</th><th class="num">14</th>
<th class="num">15</th><th class="num">16</th></tr>
<tr><td>estimate</td><td class="num">1590 m</td><td class="num">1704 m</td>
<td class="num">1573 m</td><td class="num">1571 m</td><td class="num">1570 m</td></tr>
<tr><td>ceilometer H1</td><td class="num">3150 m</td><td class="num">2520 m</td>
<td class="num">3920 m</td><td class="num">3930 m</td><td class="num">2480 m</td></tr>
</table>
<p>The reference moves through ~1450 m while the estimate moves through ~130 m. And
where the estimate <em>does</em> respond, the sign is inverted: stratified by the
reference's regime it sits at 1944 m when the cloud is <em>low</em> and 1839 m when it
is <em>high</em> &mdash; a shift of &minus;105 m (KS p = 3&times;10<sup>&minus;24</sup>),
strengthening to &minus;154 m when restricted to the inner field, with an identical
match-radius distribution across regimes.</p>
<div class="note"><strong>The flatness is in the field, not in the rule.</strong> Every
selection rule is flat over five hours (<code>lowest</code> 1156 m,
<code>largest</code> 1744 m, median-of-candidates 1878 m). The shipped
<code>lowest</code> rule reports a low-tail order statistic of a heavy-tailed field
&mdash; median 1156 m with an IQR of only 88 m &mdash; and excluding the top 128 px
lifts it only to 1463 m, so it is not one removable occluder. The dominant error is
the <strong>within-frame scatter of 1000&ndash;1400 m</strong>, 7&ndash;14&times; larger
than the estimator's entire response. The open question is therefore not "which
selection rule" but <strong>what the matched population actually is</strong>.</div>

<div class="note"><strong>What to do next, in order:</strong> mask the near-field
occluders and re-run this comparison. If the low-height population disappears, the
level offset was an artefact and the remaining gap to the ceilometer is the genuine
optical-versus-geometric difference. If it does not, the estimator is assigning
heights to matches that are not cloud, which is a matching problem (W3), not a
calibration one.</div>

<h2 id="s11"><span class="n">11</span>Temporal alignment: the lag is not the main problem</h2>
<p>The ceilometer and the cameras are physically separated, and wind advects cloud
between them, so the two series can be <em>dynamically</em> lagged — a constant
clock offset would not fix it. The project's metric already half-acknowledges this:
it searches a per-window time shift (configured &plusmn;300&nbsp;s) and keeps the
largest correlation found.</p>
<p>That search is itself suspect. In the stored window detail, <code>best_shift</code>
lands on the <strong>&plusmn;300&nbsp;s boundary in 15&ndash;25&nbsp;% of windows</strong>
(an earlier draft of this report said 45&nbsp;%; that was wrong &mdash; the maximum
across all stored runs is 30&nbsp;%, and 20&nbsp;% for this one). With the shift fixed
at zero the correlation is <em>negative</em>.</p>

<div class="note"><strong>Correction, and it is a serious one.</strong> A null test
has since been run: roll the algorithm series circularly against the fixed
ceilometer &mdash; which preserves both series' autocorrelation and destroys only
their relative timing &mdash; and run the <em>same</em> &plusmn;300&nbsp;s scan. Over
120 draws the <strong>null</strong> mean scan-max r is <strong>0.346</strong> raw and
<strong>0.421</strong> filtered, against observed values of 0.189 and 0.209. The
observation is <em>below</em> what random re-alignment produces. The cause is
statistical: a 30-minute window holds ~152 paired samples, but the ceilometer's
lag-1 autocorrelation is 0.906, leaving <strong>~3.4 effective independent
samples</strong>. Maximising a correlation over 121 shifts of three points is
selection, not measurement.</div>

<h3>The instruments are frequently watching different clouds</h3>
<figure><img src="{cmp_imgs.get('layers','')}" alt="ceilometer layers vs algorithm, and layer switching"></figure>
<p>The DOL-2 reports up to two layers. Over the validation window its lowest layer
<code>H1</code> is <strong>bimodal</strong>, not a single cloud slowly rising:</p>
<table>
<tr><th>H1 population</th><th class="num">samples</th><th class="num">share</th></tr>
<tr><td>2000&ndash;2750 m (lower layer)</td><td class="num">366</td><td class="num">40 %</td></tr>
<tr><td>2750&ndash;3500 m (almost empty)</td><td class="num">47</td><td class="num">5 %</td></tr>
<tr><td>3500&ndash;4250 m (upper layer)</td><td class="num">484</td><td class="num">53 %</td></tr>
</table>
<p>Only 3.8&nbsp;% of consecutive steps change by more than 500&nbsp;m, so this is
not noise: the instrument is reporting one cloud, then the other, in long stretches.
<strong>For about half the window the two instruments are not measuring the same
thing.</strong> No time shift can align two different clouds.</p>

<h3>The test that separates the two explanations</h3>
<p>A pure advection-lag problem would not care which layer is compared. A
layer-identity problem would change the answer completely. Restricting the
comparison to periods when the ceilometer sits on one layer:</p>
<table>
<tr><th>regime</th><th class="num">ceilometer samples</th>
<th class="num">r, fixed shift</th><th class="num">r, best shift &plusmn;300 s</th></tr>
<tr><td>all samples (as reported)</td><td class="num">906</td>
<td class="num">&minus;0.149</td><td class="num">0.181</td></tr>
<tr><td><strong>lower layer only</strong> (&lt; 2750 m)</td><td class="num">367</td>
<td class="num">&minus;0.069</td><td class="num"><strong>0.477</strong></td></tr>
<tr><td><strong>upper layer only</strong> (&gt; 3500 m)</td><td class="num">492</td>
<td class="num">+0.146</td><td class="num"><strong>0.571</strong></td></tr>
</table>

<blockquote><strong>RETRACTED.</strong> An earlier draft concluded from this table
that "the correlation triples when the comparison is restricted to a single layer".
That conclusion does not survive the same null test. Repeating the circular-roll
null <em>within each regime</em> (120 draws) gives:</blockquote>
<table>
<tr><th>regime</th><th class="num">observed mean r</th><th class="num">null mean r</th>
<th class="num">p</th><th>verdict</th></tr>
<tr><td>all samples</td><td class="num">0.181</td><td class="num">0.346</td>
<td class="num">0.992</td><td>not above null</td></tr>
<tr><td>lower layer only</td><td class="num">0.477</td><td class="num">0.546</td>
<td class="num">0.758</td><td><strong>not above null</strong></td></tr>
<tr><td>upper layer only</td><td class="num">0.571</td><td class="num">0.474</td>
<td class="num">0.067</td><td>weak</td></tr>
</table>
<p>Separating the layers raises the <em>apparent</em> correlation, but the null rises
just as much, because a shorter or more homogeneous series is easier to align by
chance. Only the upper-layer regime comes near significance, and at p = 0.067 it does
not reach it. <strong>The layer-regime improvement is not demonstrated.</strong></p>

<p>This also explains the boundary saturation. If the series being compared is a
mixture of two unrelated clouds, there is no single shift that aligns them, so the
maximisation wanders to wherever the noise is largest — often the edge of the
search range. The <code>best_shift</code> was never a lag estimate.</p>

<h3>What this changes</h3>
<ul>
<li><strong>The comparison must condition on layer identity</strong> before any lag
is discussed. Reporting a single correlation against a two-layer reference is
comparing a mixture to a mixture.</li>
<li><strong>The algorithm must report its layers</strong> (plan W5) so that a
layer-to-layer comparison is even possible. Today it reports one number chosen by
the <em>lowest cluster</em> rule, which is why it sits near 1800 m while the
ceilometer switches between 2400 m and 3800 m.</li>
<li><strong>The residual level offset is smaller than it looked.</strong> Against
the <em>lower</em> ceilometer layer the estimate is ~600 m low (1837 m vs ~2400 m),
about 23&nbsp;%; with the focal correction applied, ~18&nbsp;%. That is the right
order for the optical-versus-geometric difference the written work describes, and
it is a very different claim from "the algorithm is 50&nbsp;% low", which is what
comparing against a mixture suggested.</li>
<li><strong>Wind data for 2025-05-23 does not exist here</strong> (`meteo1`,
`meteo2`, `rvp`, `mtp` cover 2023&ndash;2024 only), and the repository records no
site coordinates. Any wind must therefore be estimated from the imagery, and the
instrument separation is currently unknown — so the advection lag can be bounded
but not yet computed. A dedicated investigation is running on exactly that.</li>
</ul>

<h2 id="s12"><span class="n">12</span>The layer hypothesis: what holds and what does not</h2>
<p>The hypothesis has two halves, and they come out opposite ways. To test them the
pipeline was bypassed entirely: every match's height was recomputed by exact
triangulation on 167 frames of 2025-05-23 (<strong>406,289 matches</strong>), with no
clustering choice in the loop.</p>
<figure><img src="{cmp_imgs.get('layerstruct','')}" alt="per-match height density over time"></figure>

<h3>On the inference side: not supported</h3>
<p>If the sky usually holds several layers, the per-match height distribution should
be multi-modal. It is not:</p>
<table>
<tr><th>significant modes per frame</th><th class="num">frames</th><th class="num">share</th></tr>
<tr><td>1 mode</td><td class="num">133</td><td class="num">80 %</td></tr>
<tr><td>2 modes</td><td class="num">34</td><td class="num">20 %</td></tr>
</table>
<p>Mean 1.20 modes per frame, and the strongest mode sits in a narrow band
(median 1675 m, p10&ndash;p90 1525&ndash;1845 m) for the whole five hours. So the stereo
is <em>not</em> merging several layers into one number. It repeatedly reports a single
thing. That is the opposite of the hypothesis, and it moves the question from
"separate the layers" to "what is that one thing, and why is it 1800 m while the
ceilometer reports 2400 m and 3800 m".</p>

<h3>On the validation side: the structure is real, the improvement is not</h3>
<p>The ceilometer's lowest layer <em>is</em> bi-modal &mdash; that measurement stands,
and it is a property of the reference, not of the metric. But the claim that
separating it improves the correlation <strong>failed its null test</strong>
(&sect;11). So the honest position is: the reference is a mixture of two clouds, which
is a real problem for any single-series comparison, and separating them is the
<i>right</i> thing to do &mdash; but doing so has not been shown to produce a
measurable improvement, because no correlation computed over these windows has been
shown to be distinguishable from chance.</p>

<h2 id="s13"><span class="n">13</span>The radial trend is lens distortion</h2>
<p>The same 406,289 matches also carry a systematic geometry signature: the apparent
height falls with distance from the principal point, from <strong>1849 m at the
centre to 1254 m at the edge</strong> of the crop &mdash; a 37&nbsp;% swing. A flat
cloud layer must give the same height everywhere, so this is an instrument
signature, and it was chased down by elimination.</p>
<figure><img src="{cmp_imgs.get('parexact','')}" alt="paraxial versus exact, and disparity components"></figure>
<table>
<tr><th>Candidate explanation</th><th>Test</th><th class="num">Trend remaining</th></tr>
<tr><td>Ideal equidistant projection</td><td>&mdash;</td><td class="num">36.7 %</td></tr>
<tr><td>Equisolid / stereographic / orthographic</td><td>four ideal lens laws</td><td class="num">20&ndash;29 %</td></tr>
<tr><td>A radial scale error between the cameras</td><td>fit and remove</td><td class="num">no change</td></tr>
<tr><td>A relative camera tilt (pitch/yaw)</td><td>bounded by the measured vertical disparity</td><td class="num">33.8&ndash;39.8 %</td></tr>
<tr><td><strong>Fisheye lens distortion</strong></td><td><strong>fit a cubic distortion term</strong></td><td class="num"><strong>3.2 %</strong></td></tr>
</table>
<figure><img src="{cmp_imgs.get('distortion','')}" alt="distortion coefficient versus radial spread"></figure>
<p>Fitting the leading deviation from equidistant,
<code>theta = (r/f)(1 + a (r/f)^2)</code>, gives a clean, smooth minimum at</p>
<blockquote><strong>a = &minus;0.30</strong>, taking the radial spread from
<strong>36.7&nbsp;% to 3.2&nbsp;%</strong>, and it is stable when the focal constant
is varied (a = &minus;0.30 at both f = 611 and f = 654, spread 3.2&nbsp;% and
3.6&nbsp;%).</blockquote>
<p>At the crop corner that coefficient compresses the angle by about 16&nbsp;%. The
pipeline ignores this distortion entirely, and it is not a small effect: it is the
difference between a height field that is radially inconsistent by 37&nbsp;% and one
that is consistent to 3&nbsp;%.</p>

<div class="note"><strong>Why this matters more than it looks.</strong> Distortion of
this size mimics cloud structure. Every match is assigned a height that depends on
where in the frame it happens to lie, so a single flat cloud produces a broad
distribution of heights &mdash; which is exactly the dispersion the clustering stage
then tries to segment, and exactly the kind of thing that can be mistaken for
multi-layer structure or for cloud relief. It is also invisible to any comparison
that uses the frame median, because the errors cancel across the field.</div>

<h3>What this does not fix</h3>
<p>Correcting the distortion makes the field geometrically consistent; it does not
move the level. The median stays near 1829 m (f = 611) or 1942 m (f = 654), against
a lower ceilometer layer near 2400 m. That remaining gap &mdash; roughly 20&nbsp;% after
the focal correction &mdash; is the genuine optical-versus-geometric difference the
written work describes, and it is now the smallest of the identified terms rather
than the largest.</p>

<h3>A fit that failed, recorded</h3>
<p>Before reaching distortion, a relative camera rotation was fitted to the same
flatness criterion and appeared to succeed spectacularly: pitch +0.99&deg;,
yaw +0.82&deg;, spread 36.7&nbsp;%&nbsp;&rarr;&nbsp;2.5&nbsp;%. It is wrong. That
rotation predicts about <strong>13.7 px</strong> of vertical disparity from the tilt
alone, while the measured vertical disparity is <strong>2.24 px</strong>. The fit was
exploiting the degeneracy between rotation and height scale, not recovering the
orientation. Restricting the tilt to what the vertical disparity allows
(|pitch| &le; 0.3&deg;) leaves the trend at 33.8&ndash;39.8&nbsp;%, i.e. unexplained
&mdash; which is what forced the search to the lens model, where it succeeded.</p>

<footer>Generated from <code>experiments/w1</code> and <code>experiments/audit</code>.
Source of record: <code>docs/plan/w1-calibration.md</code> and
<code>docs/plan/stage-audit.md</code>. Every number is reproducible from the run
artefacts; no step that produces an estimate consults the ceilometer.</footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
