# PURPOSE: Produce the current-state report as one self-contained HTML document, for a
#          reader who has seen the earlier findings report and nothing else.
#
#          It answers three questions: what can we now describe, what can we not, and
#          how was each verified. The second is the important one — the project has
#          characterised its instrument to a few percent while the retrieval itself
#          remains undemonstrated, and that distinction has to be legible.
#
#          Every number is taken from an artefact listed in docs/results/README.md.
#          Charts are generated here; photographic figures are reused from the
#          findings report.
# INPUTS: JSON and JSONL artefacts under experiments/ and ~/clouds_data/logs.
# OUTPUTS: docs/report/state.html plus its charts.
# KEYWORDS: report, state, status, verification, honesty
from __future__ import annotations

import base64
import datetime as dt
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "report")
sys.path.insert(0, os.path.join(REPO, "src"))
from clouds.sources import ceilometer as C                                 # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.autolayout": True,
})
INK, ACC, GOOD, GREY, BLUE = "#1b2430", "#c1440e", "#1a7f37", "#8b949e", "#1f6feb"

RUN = "/home/iustimov/clouds_data/logs/20261013T112649Z_evaluate_clouds_2025-05-23_3565c5"
LOG = "/home/iustimov/clouds_data/Звенигород/scilometer/Zve_Sci_250523.txt"


def b64(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def load_series(run=RUN):
    s = json.load(open(os.path.join(run, "series.json")))
    t = [dt.datetime.fromisoformat(x) for x in s["times_ceilometer_clock"]]
    h = np.array([np.nan if v is None else v for v in s["filtered_h"]], float)
    r = np.array([np.nan if v is None else v for v in s["raw_h"]], float)
    return t, h, r


def load_ceilo():
    ss = C.parse_ceilometer_log(LOG)
    lo, hi = dt.time(12, 0, 0), dt.time(17, 0, 0)
    ce = [x for x in ss if x.time.date() == dt.date(2025, 5, 23)
          and lo <= x.time.time() <= hi and x.h1 is not None]
    return [x.time for x in ce], np.array([x.h1 for x in ce], float)


# 1 --- the central figure: the estimate is flat while the reference swings ---------
def chart_flat_vs_swing():
    t, h, _ = load_series()
    ct, ch = load_ceilo()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.scatter(ct, ch, s=5, color=GREY, alpha=0.6, label="ceilometer H1 (per sample)")
    ax1.plot(t, h, color=GOOD, lw=1.6, label="stereo estimate (filtered)")
    ax1.set_ylabel("cloud base height (m)")
    ax1.set_xlabel("time (MSK)")
    ax1.set_ylim(0, 4400)
    ax1.set_title("Five hours: the reference moves ~1450 m, the estimate ~130 m")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    hours = [12, 13, 14, 15, 16]
    est, cei = [], []
    for hh in hours:
        m = np.array([x.hour == hh for x in t]) & np.isfinite(h)
        mc = np.array([x.hour == hh for x in ct])
        est.append(float(np.nanmedian(h[m])) if m.sum() else np.nan)
        cei.append(float(np.median(ch[mc])) if mc.sum() else np.nan)
    x = np.arange(len(hours))
    ax2.bar(x - 0.2, cei, 0.4, color=GREY, label="ceilometer H1")
    ax2.bar(x + 0.2, est, 0.4, color=GOOD, label="stereo estimate")
    for i, (a, b) in enumerate(zip(cei, est)):
        ax2.text(i - 0.2, a + 40, f"{a:.0f}", ha="center", fontsize=7.5, color=GREY)
        ax2.text(i + 0.2, b + 40, f"{b:.0f}", ha="center", fontsize=7.5, color=GOOD)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{hh:02d}h" for hh in hours])
    ax2.set_ylabel("median height (m)")
    ax2.set_ylim(0, 4600)
    ax2.set_title("Hourly medians")
    ax2.legend(fontsize=8)
    return save(fig, "state_flat_vs_swing.png")


# 2 --- the inverted response ------------------------------------------------------
def chart_inversion():
    d = json.load(open(os.path.join(REPO, "experiments", "planning",
                                    "anti_correlation_probe.json")))
    full = d["regime_response"]["full"]
    inner = d["regime_response"]["r_le_250"]
    labels = ["full field", "r ≤ 250 px"]
    lo = [full["median_lower_m"], inner["median_lower_m"]]
    up = [full["median_upper_m"], inner["median_upper_m"]]
    shift = [full["shift_upper_minus_lower_m"], inner["shift_upper_minus_lower_m"]]
    ps = [full["ks_p"], inner["ks_p"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar(x - 0.2, lo, 0.4, color=GREY, label="reference LOW (H1 < 2750 m)")
    ax.bar(x + 0.2, up, 0.4, color=ACC, label="reference HIGH (H1 > 3500 m)")
    for i in range(2):
        ax.text(i - 0.2, lo[i] + 25, f"{lo[i]:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, up[i] + 25, f"{up[i]:.0f}", ha="center", fontsize=8)
        ax.annotate(f"shift {shift[i]:+.0f} m\np = {ps[i]:.0e}",
                    (i, max(lo[i], up[i]) + 260), ha="center", fontsize=7.5, color=ACC)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("median estimated height (m)")
    ax.set_ylim(1500, 2500)
    ax.set_title("When the reference is HIGH the estimate is LOWER")
    ax.legend(fontsize=7.5, loc="lower right")
    return save(fig, "state_inversion.png")


def build(imgs, state):
    def row(name, status, how, evidence):
        cls = {"established": "ok", "not established": "bad",
               "withdrawn": "warn"}.get(status, "")
        return (f'<tr class="{cls}"><td><strong>{name}</strong></td>'
                f'<td class="st {cls}">{status}</td><td>{how}</td>'
                f'<td><code>{evidence}</code></td></tr>')

    rows = "".join(row(*r) for r in state)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Where the project stands</title><style>
 :root {{ --ink:#1b2430; --acc:#c1440e; --blue:#1f6feb; --good:#1a7f37;
         --grey:#5c6672; --line:#e3e6ea; --bg:#fbfbfc; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
        font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1060px; margin:0 auto; padding:0 28px 90px; }}
 header {{ padding:52px 0 24px; border-bottom:3px solid var(--ink); }}
 h1 {{ font-size:33px; line-height:1.2; margin:0 0 8px; letter-spacing:-.5px; }}
 .sub {{ color:var(--grey); font-size:15px; }}
 h2 {{ font-size:22px; margin:48px 0 6px; }}
 h2 .n {{ color:var(--acc); margin-right:9px; }}
 h3 {{ font-size:15.5px; margin:24px 0 4px; }}
 p,li,td {{ font-size:15px; }}
 figure {{ margin:20px 0; }}
 figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }}
 figcaption {{ color:var(--grey); font-size:13px; margin-top:7px; }}
 table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:14px; }}
 th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
         vertical-align:top; }}
 th {{ background:#f2f4f7; font-weight:600; }}
 td.st {{ font-weight:600; white-space:nowrap; }}
 td.st.ok {{ color:var(--good); }}
 td.st.bad {{ color:var(--acc); }}
 td.st.warn {{ color:#9a6700; }}
 code {{ background:#f2f4f7; padding:1px 5px; border-radius:3px; font-size:12.5px; }}
 .big {{ display:flex; gap:13px; flex-wrap:wrap; margin:22px 0; }}
 .kpi {{ flex:1 1 175px; background:#fff; border:1px solid var(--line);
         border-left:4px solid var(--acc); border-radius:6px; padding:13px 15px; }}
 .kpi.good {{ border-left-color:var(--good); }}
 .kpi .v {{ font-size:23px; font-weight:700; }}
 .kpi .l {{ color:var(--grey); font-size:12.5px; margin-top:2px; }}
 .note {{ background:#fff7ed; border-left:4px solid var(--acc); padding:12px 16px;
          border-radius:0 6px 6px 0; margin:18px 0; font-size:14.5px; }}
 .note.ok {{ background:#f0fdf4; border-left-color:var(--good); }}
 blockquote {{ margin:18px 0; padding:2px 0 2px 16px; border-left:4px solid var(--ink); }}
 footer {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
           color:var(--grey); font-size:13px; }}
</style></head><body><div class="wrap">

<header>
  <h1>Where the project stands</h1>
  <div class="sub">Stereo cloud-base height from two fisheye cameras &middot; state after the
  measurement campaign &middot; 2025-05-23, one usable session</div>
</header>

<div class="note"><strong>The short version.</strong> The <em>instrument</em> is now
characterised to a few percent: its geometry, its residual misalignment and its lens
distortion are measured, cross-checked and in two cases corrected end-to-end. The
<em>retrieval</em> is not demonstrated. The estimate is nearly flat at ~1800 m while the
reference moves through 1450 m, and where it does respond the sign is inverted. Every
earlier correlation figure has been withdrawn after failing a null test, including two
of my own. So: we can say precisely what the camera system does, and we cannot yet say
that the height it reports is cloud base height.</div>

<div class="big">
  <div class="kpi good"><div class="v">2.4 %</div>
    <div class="l">residual radial inconsistency of the height field, after correction (was 19.1 %)</div></div>
  <div class="kpi good"><div class="v">&minus;1.6&deg;</div>
    <div class="l">residual roll between the cameras, measured three independent ways</div></div>
  <div class="kpi good"><div class="v">2.03&times;</div>
    <div class="l">faster inference, with byte-identical output</div></div>
  <div class="kpi"><div class="v">~130 m</div>
    <div class="l">how far the estimate moves while the reference moves 1450 m</div></div>
</div>

<h2 id="s1"><span class="n">1</span>What we can describe</h2>
<p>These are established. Each was measured, and most were measured more than one way
because a single method had already proved unreliable once.</p>

<h3>The optical geometry</h3>
<p>The rig is two 180&deg; fisheye cameras on a 45 m baseline, 1920&times;1920, ten-second
cadence. Three things about it are now known rather than assumed:</p>
<ul>
<li><strong>The focal constant is ~654 px/rad, not the 611 the pipeline assumes</strong>
(ratio 1.070 &plusmn; 0.018). The assumed value is &xi;<sub>0</sub>/&pi;, which presumes
the 180&deg; field maps exactly onto the inscribed circle; the circle actually measures
1078&ndash;1156 px, not 960. Every height is therefore currently ~7 % low, purely
multiplicatively.</li>
<li><strong>A residual roll of about &minus;1.6&deg; between the rectified images.</strong>
Removing it collapses the within-frame disparity spread 4&ndash;5&times;.</li>
<li><strong>A cubic lens distortion, <code>a = &minus;0.30</code>,</strong> in
<code>&theta; = (r/f)(1 + a(r/f)&sup2;)</code>. The pipeline ignores it entirely. It is
large: about 16 % angular compression at the crop corner.</li>
</ul>
<figure><img src="{imgs['r2']}" alt="radial profile before and after the distortion correction"></figure>
<figcaption>The distortion correction, run end-to-end through the pipeline over
162,000 clusters. A flat cloud layer must give a flat profile; the correction takes the
profile's interquartile range from 19.1 % to 2.4 %, moving the median only &minus;2.0 %.</figcaption>

<h3>The reference's structure</h3>
<p>The ceilometer's lowest reported layer is <strong>bi-modal</strong>: 40 % of samples
between 2250 and 2750 m, 53 % between 3750 and 4250 m, and almost nothing between. Only
3.8 % of consecutive steps change by more than 500 m, so this is two clouds reported in
long stretches, not noise. <strong>For about half the window the two instruments are not
measuring the same object</strong>, which matters for any comparison regardless of how
good the algorithm is.</p>

<h3>The matched population</h3>
<p>Correspondences are cloud-driven: the clear day yields single-digit matches per hour
against 1,100&ndash;3,700 on the cloudy day. And the selectable rules differ enormously in
level &mdash; <code>lowest</code> 1156 m, <code>largest</code> 1744 m,
median-of-candidates 1878 m &mdash; which says the height field is broad and the reported
number is a choice about which part of it to quote.</p>

<h2 id="s2"><span class="n">2</span>What we cannot describe</h2>
<div class="note"><strong>The core result, and it is now diagnosed.</strong> There is no
demonstrated agreement with the reference in level or dynamics, and the reason is no
longer a mystery: <strong>the estimated height is dominated by a static, image-position
dependent pattern rather than by cloud.</strong> The test is below, and it is the single
most important measurement in this document.</div>

<h3>The decisive test: does the height follow the cloud, or the pixel?</h3>
<p>If the retrieval assigns a height to <em>cloud</em>, the height map must move as the
cloud moves. If it assigns a height to a <em>place in the image</em>, the map is the same
all session. The measured cloud motion moves air through the field many times over in five
hours, so a static map cannot mean the cloud stood still.</p>
<p>Correlating the height map of the session's <strong>first half</strong> with its
<strong>second half</strong> &mdash; disjoint frames, 2.5 hours apart, through which
entirely different air has passed:</p>
<table>
<tr><th>configuration</th><th class="num">r between halves</th>
<th class="num">shuffled null</th><th class="num">spatial spread</th></tr>
<tr><td>roll correction only</td><td class="num"><strong>+0.981</strong></td>
<td class="num">&minus;0.002</td><td class="num">173 m</td></tr>
<tr><td>roll + lens distortion</td><td class="num"><strong>+0.912</strong></td>
<td class="num">&minus;0.002</td><td class="num">92 m</td></tr>
</table>
<figure><img src="{imgs['r11']}" alt="height by image cell, first half vs second half"></figure>
<figcaption>A large part of the estimated height is fixed to the pixel, not to the cloud.
The correction halves the pattern's amplitude and the rest is something else.</figcaption>
<p><strong>One mechanism explains every failure found earlier.</strong> A static map has a
constant average, so the delivered series is nearly flat &mdash; the 130 m it moves against
the reference's 1450 m. Every selection rule samples the same static map, so none of them
helps. The residual cloud signal rides on top of a pattern that does not move, so its sign
carries no information. And adding matches widens the pattern rather than adding cloud
signal.</p>
<p><strong>A static height pattern is a calibration signature</strong>, and it has now
been characterised directly. The static part of the disparity field <em>is</em> the error
field, because the cloud moves and the error does not. Fitting it to candidate geometric
forms, as a fraction of the field's own 0.84 px standard deviation:</p>
<table>
<tr><th>model</th><th class="num">rms residual</th><th class="num">explains</th></tr>
<tr><td>constant</td><td class="num">0.837 px</td><td class="num">0 %</td></tr>
<tr><td><strong>linear in u</strong></td><td class="num"><strong>0.486 px</strong></td>
<td class="num"><strong>41.9 %</strong></td></tr>
<tr><td>radial (r)</td><td class="num">0.726 px</td><td class="num">13.3 %</td></tr>
<tr><td>radial (r&sup2;, r&sup3;, combined)</td><td class="num">0.725 px</td>
<td class="num">13.4 %</td></tr>
<tr><td>roll</td><td class="num">0.817 px</td><td class="num">2.4 %</td></tr>
<tr><td><strong>radial + linear + roll</strong></td><td class="num"><strong>0.168 px</strong></td>
<td class="num"><strong>79.9 %</strong></td></tr>
</table>
<figure><img src="{imgs['r8']}" alt="static disparity field, per-cell temporal spread, and radial structure"></figure>
<figcaption>The static disparity field; its per-cell temporal spread; and the test of whether
it is radial &mdash; it is not.</figcaption>
<p><strong>The dominant error is a horizontal SCALE, not the lens distortion.</strong> The
linear-in-u coefficient is +0.00394 px per px, a <strong>+16.6 % change in disparity across
the 640 px crop</strong>. At 1.5 km, where the entire disparity is 18.3 px, that moves the
height by more than 15 % from one side of the image to the other. The distortion is real
and separate: it is radial, and applying it removed about half the static amplitude.</p>
<p>Subtracting the exact geometry of a horizontal plane leaves a residual <em>larger</em>
than the field it was subtracted from, so the disagreement is systematic rather than
marginal. Three explanations remain live and are distinguishable: <strong>a wrong principal
point, differing focal lengths between the cameras, or a confound in which clouds occupy
which image regions.</strong> Fitting a correction before separating them would encode the
confound, so that separation is the next measurement and the gating one for the project.</p>
<p>Finally, the <strong>temporal filter is unwired</strong> while the retrieval is diagnosed
&mdash; it was removing 48 % of the raw variation of a signal already ~4&times;
under-responsive, and confounding the two makes both unreadable. It is a config switch
(<code>temporal.filter_enabled</code>), not a deletion.</p>

<figure><img src="{imgs['flat']}" alt="estimate flat against a swinging reference"></figure>
<figure><img src="{imgs['inversion']}" alt="inverted response by regime"></figure>
<p>Two facts sit behind that:</p>
<ul>
<li><strong>The estimate barely moves.</strong> Hourly medians range over ~130 m while
the reference moves through ~1450 m.</li>
<li><strong>Where it does respond, the sign is inverted.</strong> Stratified by the
reference's own regime, the estimate sits at 1944 m when the cloud is <em>low</em> and
1839 m when it is <em>high</em> &mdash; a shift of &minus;105 m (KS
p&nbsp;=&nbsp;3&times;10<sup>&minus;24</sup>), strengthening to &minus;154 m restricted
to the inner field. The match-radius distribution is identical across regimes (227.9 vs
227.7 px), so this is not a radial-geometry artefact.</li>
</ul>
<p><strong>Corrected 2026-10-13, in the favourable direction.</strong> An earlier version
of this report named a "1000&ndash;1400 m within-frame scatter" as the dominant error.
That figure came from the <em>uncorrected</em> configuration. Isolated on identical
matches, the roll correction halves the height spread (689 &rarr; 341 m) and the distortion
alone slightly worsens it (792 m) — the same compensation seen above — while
<strong>together they reduce it 2.9&times; to ~235 m and raise the median 31 %</strong>,
from 1441 m to 1886 m against a lower reference layer near 2400 m. The gap to the
reference therefore halves, to ~514 m, before the ~7 % focal correction. The flatness
remains in the height <em>field</em> rather than the selection rule, but the field is
2.9&times; narrower than it appeared.</p>

<h3>The limits on what can be tested here at all</h3>
<table>
<tr><th>limit</th><th>consequence</th></tr>
<tr><td><strong>One usable session</strong> &mdash; 2025-05-22 is cloud-free</td>
<td>no session hold-out, no independent period; every check is a time block of the same day</td></tr>
<tr><td><strong>No meteorology and no site coordinates for the session</strong></td>
<td>the advection lag can be bounded but not computed; the instrument separation is unknown</td></tr>
<tr><td><strong>Footprint ~1956 m against a ~10 m ceilometer column</strong></td>
<td>a point column cannot equal a spatial average, and no time shift repairs a spatial mismatch</td></tr>
<tr><td><strong>The filtered output has 18.8 effective samples in five hours</strong></td>
<td>the smallest detectable correlation is ~0.49; the smoothing destroys the variability validation needs</td></tr>
</table>

<h2 id="s3"><span class="n">3</span>How it was verified &mdash; and what was withdrawn</h2>
<p>The verification method matters as much as the results, because the first version of
this project's evaluation was wrong in a way that produced plausible numbers.</p>

<h3>Every claim against a null</h3>
<p>The windowed correlation metric &mdash; the project's original validation &mdash;
searched a time shift per 30-minute window and reported the largest correlation found.
A null that <em>rolls the estimate</em> against a fixed reference and performs the
<em>same search</em> scores higher than the observation (0.346 against 0.189 raw). The
cause is statistical: a window holds ~152 paired samples, but the reference's
autocorrelation leaves ~3.4 independent ones, and maximising over 121 shifts of three
points is selection, not measurement. <strong>Every windowed correlation figure is
withdrawn.</strong></p>

<h3>Synthetic ground truth gets an independent recovery check</h3>
<p>The sub-pixel and differential matcher tests build ground truth by translating a real
image by a known amount. Recovery of those shifts was confirmed with phase correlation
&mdash; an exact estimator with no learned prior &mdash; at ratio 0.88&ndash;1.36, so the
failures the matchers show are theirs and not the test's.</p>

<h3>A proxy metric was caught rewarding the defect it was meant to detect</h3>
<p>The first matcher evaluation ranked candidates by a plane-fit residual. It put
MASt3R first. A direct test of differential sensitivity at the working disparity &mdash;
can the matcher resolve a 0.5 px change on a 20 px disparity &mdash; gives it a slope of
<strong>0.000</strong>: no response at all. A consistently <em>quantised</em> field
satisfies a plane fit as well as an accurate one. That is why the test, not the proxy,
now decides.</p>

<h3>Reproducibility is enforced, not assumed</h3>
<p>The default matcher path reproduces the frozen reference correspondences
<strong>38 of 38 frames byte-for-byte</strong>, re-verified after every source edit. The
optimisation that made inference 2.03&times; faster was required to produce
<code>cmp</code>-identical <code>series.json</code>, <code>clusters.jsonl</code> and
<code>metrics.json</code>.</p>

<h3>Withdrawals, including my own</h3>
<table>
<tr><th>claim</th><th>what happened</th></tr>
<tr><td>The estimate's <em>dynamics</em> are anti-correlated with the reference (r = &minus;0.248)</td>
<td><strong>withdrawn.</strong> 30 of 469 differences spanned a missing epoch in the resampling grid. Gap-safe it is r = &minus;0.045, CI [&minus;0.100, &minus;0.007]. The failure is in <em>level</em>, not dynamics</td></tr>
<tr><td>Separating the reference's layers triples the correlation</td>
<td><strong>withdrawn.</strong> The null rises as much as the observation; p = 0.758 for the lower layer</td></tr>
<tr><td>best_shift saturates the &plusmn;300 s search in ~45 % of windows</td>
<td><strong>corrected</strong> to 15&ndash;25 % (max 30 % across stored runs)</td></tr>
<tr><td>A relative camera rotation explains the radial trend</td>
<td><strong>withdrawn.</strong> The fit predicted 13.7 px of vertical disparity against 2.24 measured &mdash; a height/rotation degeneracy</td></tr>
<tr><td>The spatial filter improves the spatial coherence of the product</td>
<td><strong>removed.</strong> It rejected 37&ndash;43 % of clusters while leaving the raw series bit-identical and moving the filtered series 0.3 %</td></tr>
</table>

<h2 id="s4"><span class="n">4</span>Claim-by-claim status</h2>
<table>
<tr><th>claim</th><th>status</th><th>how verified</th><th>artefact</th></tr>
{rows}
</table>

<h2 id="s5"><span class="n">5</span>What would change the picture</h2>
<p>In order, and each is falsifiable:</p>
<ol>
<li><strong>Establish what the matched population is.</strong> The estimate is a low-tail
order statistic of a broad, noisy field whose level is set by the field's spread and the
match count. That, not the correlation, is the dominant unexplained error.</li>
<li><strong>Apply the geometry corrections together.</strong> Exact triangulation
<em>without</em> the distortion is worse than the paraxial formula (19.1 % against
4.1 %) because the two errors partially cancel. They must ship as a pair.</li>
<li><strong>Re-test the multi-layer question under the corrected geometry</strong>, and
<strong>decompose the level offset</strong> only after that.</li>
<li><strong>Build the spatial-sampling model</strong> and decide whether a level
comparison between a 1956 m footprint and a 10 m column is defensible at all.</li>
</ol>
<blockquote>The one thing the project does not yet know: <strong>what the matched
population actually is.</strong> Until that is answered, neither the level offset nor the
single-versus-bi-modal-layer question can be interpreted.</blockquote>

<footer>Companion documents: <code>docs/report/findings.html</code> (the earlier, longer
narrative), <code>docs/results/</code> (citable index, incl. the matcher-selection
principle and every negative result), <code>docs/plan/evaluation.md</code> (how a result
is allowed to be claimed), <code>docs/plan/research-workplan.md</code> (R1&ndash;R8).
Every figure above is generated from artefacts in the repository.</footer>
</div></body></html>"""


# (name, status, how verified, artefact)
STATE = [
    ("Filenames are UTC; ceilometer is MSK; +3 h",
     "established",
     "two independent methods: the diurnal brightness envelope, and a solar ephemeris matching the Sun's image azimuth to &lt;0.5&deg; over 14 hours",
     "experiments/w1/sun_reference.py"),
    ("Focal constant f &asymp; 654 &plusmn; 11 px/rad",
     "established",
     "Sun radius against known zenith angle, four independent streams (2 cameras &times; 2 days); consistent to 3.5 %",
     "experiments/w1/f_identifiability.py"),
    ("Residual camera roll &asymp; &minus;1.6&deg;",
     "established",
     "three independent methods agreeing in magnitude: disparity-field fit, spread minimisation, infinity-test residuals",
     "experiments/w1/epipolar_check.py"),
    ("Lens distortion a = &minus;0.30",
     "established",
     "fitted from the sky's own flatness requirement; stable against f (a = &minus;0.30 at both 611 and 654); then confirmed end-to-end through the pipeline",
     "experiments/layers/distortion_fit.py, experiments/layers/r2_radial_profile.py"),
    ("The ceilometer's lowest layer is bi-modal",
     "established",
     "direct inspection of the H1 record: 40 % / 53 % in two populations, 3.8 % of steps &gt; 500 m",
     "experiments/report/ceilometer_layers.py"),
    ("RoMa is the most accurate available matcher here",
     "established",
     "differential sensitivity at the working disparity (slope 1.008, scatter 0.016 px), &gt;20&times; better than LoFTR; plus 0.68 px interior accuracy against a 7.02 px null",
     "experiments/matchers/differential_sensitivity.py"),
    ("Matcher path reproduces the frozen reference exactly",
     "established",
     "38 of 38 frames byte-identical at batch_size 1, re-verified after all source edits",
     "experiments/matchers/verify_loftr_baseline.py"),
    ("Inference is 2.03&times; faster with identical output",
     "established",
     "500-frame same-data A/B; outputs verified cmp-identical",
     "experiments/w1/opt/ab_evaluate.py"),
    ("The estimate tracks the reference in LEVEL",
     "not established",
     "level correlation is negative and significant (r = &minus;0.376); the estimate sits ~600&ndash;1000 m below the lower layer with a flat profile",
     "experiments/report/metric_power.py"),
    ("The estimate tracks the reference in DYNAMICS",
     "not established",
     "gap-safe first-difference correlation is nil (&minus;0.045, CI [&minus;0.100, &minus;0.007]); the estimate moves ~130 m against 1450 m",
     "experiments/planning/anti_correlation_probe.py"),
    ("The response has the correct SIGN",
     "not established",
     "stratified by reference regime the shift is &minus;105 m (p = 3e&minus;24), strengthening to &minus;154 m in the inner field",
     "experiments/planning/anti_correlation_probe.py"),
    ("The estimate is single-modal while the reference is bi-modal",
     "not established",
     "1.20 significant height modes per frame over 167 frames; single mode in 80 % of frames &mdash; but measured under the uncorrected geometry",
     "experiments/layers/layer_analysis.py"),
    ("The estimate follows the CLOUD rather than the image position",
     "not established",
     "the height map correlates r = 0.98 (roll only) / 0.91 (roll + distortion) between the session's two halves, against a shuffled null of -0.002; the cloud moves and the map does not",
     "experiments/layers/r11_identity.py"),
    ("The static calibration error is radial (lens distortion)",
     "withdrawn",
     "the static disparity field is dominated by a LINEAR gradient in u (41.9 %) not by radial terms (13.5 %); the coefficient implies a +16.6 % horizontal scale error across the crop",
     "experiments/layers/r8_static_field.py"),
    ("Any windowed correlation figure",
     "withdrawn",
     "fails a circular-roll null that performs the same shift search (0.346 null vs 0.189 observed)",
     "docs/plan/temporal-alignment.md"),
    ("Advection explains the observed time shifts",
     "withdrawn",
     "a fixed-baseline advection fit gives R&sup2; = 0.10 (p = 0.18); adjacent windows jump 200&ndash;250 s; changing only the roll moves the best shift by 168 s",
     "docs/plan/temporal-alignment.md"),
]


def main():
    imgs = {
        "r8": b64(os.path.join(OUT, "r8_static_field.png")),
        "r11": b64(os.path.join(OUT, "r11_identity.png")),
        "r2": b64(os.path.join(OUT, "r2_radial_profile.png")),
        "flat": b64(os.path.join(OUT, chart_flat_vs_swing())),
        "inversion": b64(os.path.join(OUT, chart_inversion())),
    }
    html = build(imgs, STATE)
    path = os.path.join(OUT, "state.html")
    open(path, "w").write(html)
    print(f"wrote {path}  ({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
