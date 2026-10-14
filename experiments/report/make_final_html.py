# PURPOSE: Generate the final project report as one self-contained HTML document, for a
#          reader who has been away from the work and needs the whole of it.
#
#          The narrative follows docs/report/final-report.md, which is the verified
#          source. Every number in this file is traced to an artefact under
#          experiments/, and every figure is one of the generated PNGs under
#          docs/report/ - nothing is drawn by hand and nothing is recalled.
#
#          The three architectural standards the report is required to carry:
#            - an explicit LIMITATION block for every claim of capability;
#            - a named pre-registration for every trigger, so no metric reads as post-hoc;
#            - the harness post-mortem, as a first-class finding about the process.
# INPUTS: docs/report/*.png and the figures' generating scripts.
# OUTPUTS: docs/report/final.html (self-contained; ~1-3 MB with embedded images).
# KEYWORDS: report, html, self_contained, final, findings, null_result
from __future__ import annotations

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(REPO, "docs", "report")
OUT = os.path.join(FIGDIR, "final.html")


def b64(name: str) -> str:
    with open(os.path.join(FIGDIR, name), "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def fig(name, caption, key):
    """A figure with a caption, embedded so the document stands alone."""
    return (f'<figure><img src="{b64(name)}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


# --- figures, loaded once so the base64 payload appears once each -----------------
FIG = {
    "w7_control": fig("w7_control.png",
        "The validated synthetic control. Left: RoMa's recovered disparity against the "
        "exact per-pixel ground truth across a 4× height range. Right: the selection "
        "rules against the truth — the shipped <code>lowest</code> rule tracks nothing, "
        "while <code>lowest_mode</code> follows it.", None),
    "w8_relief": fig("w8_relief.png",
        "Injection: 3-D relief. Relief up to 20 % of the cloud height — twice the "
        "physically expected roughness — leaves the recovery monotonic with "
        "ρ = +1.000.", None),
    "r2_radial": fig("r2_radial_profile.png",
        "The measured lens distortion, applied end-to-end through the pipeline. The "
        "radial inconsistency of the height field falls from 19.1 % to 2.4 % with the "
        "median moving only −2.0 %.", None),
    "r11": fig("r11_identity.png",
        "The decisive anomaly test: the height map per image cell, first half against "
        "second half. The pattern is fixed to the pixel, not the cloud — r = 0.98 "
        "against a shuffled null of 0.", None),
    "r8": fig("r8_static_field.png",
        "The static disparity field, its per-cell temporal spread, and the test of "
        "whether it is radial. It is not: the static error is dominated by a linear "
        "gradient in u.", None),
    "w4_cert": fig("w4_certainty.png",
        "Gate 1: certainty. Within every one of ten certainty deciles the high regime "
        "carries the higher disparity, so certainty cannot be the cause.", None),
    "w4_smooth": fig("w4_smoothness.png",
        "Gate 2: coherence. The high regime is, if anything, marginally more coherent "
        "than the low regime (ratio 0.865) while its disparity differs by 1.785 px.", None),
    "w4_desync": fig("w4_desync.png",
        "Gate 8: desynchronisation. The epipolar violation tracks the cloud's height "
        "regime (right), not its speed (left) — ρ = +0.447 against +0.002.", None),
    "w6": fig("w6_photometric.png",
        "Gate 10: photometry. The within-frame brightness coupling is real but runs the "
        "wrong way, and the gradient-direction signature a photometric pull requires is "
        "absent (ρ = −0.000).", None),
    "w8_super": fig("w8_superposition.png",
        "Gate 11: photometric superposition. It widens the core where reality widens "
        "the tails, and only at 80 % clear sky against the real 2.6–3.6 %.", None),
    "flat_swing": fig("state_flat_vs_swing.png",
        "The anomaly in its simplest form: the estimate is nearly flat while the "
        "reference moves through 1450 m.", None),
    "comparison": fig("comparison_timeseries.png",
        "Estimate against reference over the session.", None),
    "ceilometer": fig("ceilometer_layers.png",
        "The reference's own structure: its lowest reported layer is bi-modal, 40 % at "
        "2250–2750 m and 53 % at 3750–4250 m, so for about half the window the two "
        "instruments are not measuring the same object.", None),
}


def limitation(text):
    return f'<div class="lim"><span class="limtag">LIMITATION</span> {text}</div>'


def refrow(n, hyp, meas, why, record):
    return (f"<tr><td class='num'>{n}</td><td><strong>{hyp}</strong></td>"
            f"<td>{meas}</td><td>{why}</td><td><code>{record}</code></td></tr>")


def build():
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Stereo cloud-base height — project report</title><style>
 :root {{ --ink:#161b22; --dim:#5c6672; --acc:#c1440e; --good:#1a7f37; --blue:#1f6feb;
          --line:#e3e6ea; --bg:#fcfcfd; --warn:#9a6700; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   -webkit-font-smoothing:antialiased; }}
 .wrap {{ max-width:1080px; margin:0 auto; padding:0 30px 110px; }}
 header {{ padding:58px 0 26px; border-bottom:3px solid var(--ink); }}
 h1 {{ font-size:34px; line-height:1.18; margin:0 0 12px; letter-spacing:-.6px; }}
 .sub {{ color:var(--dim); font-size:15.5px; max-width:75ch; }}
 h2 {{ font-size:23px; margin:56px 0 4px; letter-spacing:-.3px; }}
 h2 .n {{ color:var(--acc); margin-right:10px; font-variant-numeric:tabular-nums; }}
 h3 {{ font-size:16px; margin:30px 0 6px; }}
 p, li, td, th {{ font-size:15px; }}
 figure {{ margin:22px 0; }}
 figure img {{ width:100%; border:1px solid var(--line); border-radius:7px; background:#fff; }}
 figcaption {{ color:var(--dim); font-size:13.5px; margin-top:8px; max-width:92ch; }}
 table {{ border-collapse:collapse; width:100%; margin:18px 0; font-size:14px; }}
 th, td {{ text-align:left; padding:9px 11px; border-bottom:1px solid var(--line);
   vertical-align:top; }}
 th {{ background:#f3f5f8; font-weight:600; font-size:13px; letter-spacing:.02em;
   text-transform:uppercase; color:#404850; }}
 td.num {{ font-variant-numeric:tabular-nums; color:var(--dim); }}
 code {{ background:#f3f5f8; padding:1.5px 5px; border-radius:4px; font-size:12.5px; }}
 .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin:24px 0; }}
 .kpi {{ flex:1 1 190px; background:#fff; border:1px solid var(--line);
   border-left:4px solid var(--acc); border-radius:7px; padding:14px 16px; }}
 .kpi.good {{ border-left-color:var(--good); }}
 .kpi .v {{ font-size:24px; font-weight:700; font-variant-numeric:tabular-nums;
   letter-spacing:-.5px; }}
 .kpi .l {{ color:var(--dim); font-size:12.5px; margin-top:3px; line-height:1.45; }}
 .note {{ background:#fff7ed; border-left:4px solid var(--acc); padding:14px 18px;
   border-radius:0 7px 7px 0; margin:20px 0; }}
 .note.ok {{ background:#f0fdf4; border-left-color:var(--good); }}
 .lim {{ background:#fbfaf7; border:1px solid #e8e2d6; border-left:4px solid var(--warn);
   padding:13px 17px; border-radius:0 7px 7px 0; margin:16px 0; font-size:14.5px; }}
 .limtag {{ display:inline-block; background:var(--warn); color:#fff; font-size:10.5px;
   font-weight:700; letter-spacing:.09em; padding:2px 7px; border-radius:3px;
   margin-right:9px; vertical-align:1px; }}
 .pre {{ background:#f5f8ff; border-left:4px solid var(--blue); padding:12px 16px;
   border-radius:0 7px 7px 0; margin:16px 0; font-size:14.5px; }}
 .pre code {{ background:#e8eeff; }}
 blockquote {{ margin:20px 0; padding:4px 0 4px 18px; border-left:4px solid var(--ink);
   color:#2a3138; }}
 footer {{ margin-top:64px; padding-top:20px; border-top:1px solid var(--line);
   color:var(--dim); font-size:13.5px; }}
 .toc {{ background:#fff; border:1px solid var(--line); border-radius:8px;
   padding:16px 22px; margin:26px 0; }}
 .toc ol {{ margin:6px 0 0; padding-left:22px; }}
 .toc a {{ color:var(--ink); text-decoration:none; border-bottom:1px solid var(--line); }}
 .toc a:hover {{ border-bottom-color:var(--acc); }}
</style></head><body><div class="wrap">

<header>
  <h1>Stereo cloud-base height from a 45&nbsp;m fisheye baseline</h1>
  <div class="sub">A validated instrument, a measured anomaly, and twelve pre-registered
  refutations. The instrument works; the cause of the real-world failure is
  <strong>unattributed</strong>, and this report says so rather than inferring one.</div>
</header>

<div class="toc"><strong>Contents</strong>
<ol>
<li><a href="#s1">The validated instrument</a></li>
<li><a href="#s2">The measured anomaly</a></li>
<li><a href="#s3">Twelve refutations</a></li>
<li><a href="#s4">The boundary — what may and may not be claimed</a></li>
<li><a href="#s5">What would settle it</a></li>
<li><a href="#s6">Methodology, and the harness post-mortem</a></li>
</ol></div>

<div class="note"><strong>The one-paragraph version.</strong> We built a passive stereo
cloud-base-height estimator from two zenith-pointing fisheye cameras on a 45&nbsp;m baseline
and tested it against a co-located laser ceilometer. We first proved the instrument works
<em>by construction</em>: a synthetic renderer, whose geometry is derived from the shipped
calibration and gated against the analytic parallax, recovers a cloud layer of known height
with <strong>ρ&nbsp;=&nbsp;+1.000 and 0.07&nbsp;px median bias</strong>. Through that same
renderer we found the shipped <code>lowest</code> selection rule errs by
<strong>325–1471&nbsp;m</strong> on ground truth, and replaced it with
<code>lowest_mode</code>, which errs by ≤64&nbsp;m. On real data the estimator does
<strong>not</strong> recover cloud base height: higher clouds produce lower reported heights,
significantly and robustly. We tested and refuted <strong>twelve</strong> explanations for
that inversion with pre-registered criteria. <strong>None accounts for it.</strong></div>

<div class="kpis">
  <div class="kpi good"><div class="v">+1.000</div>
    <div class="l">ρ(true, recovered) on a synthetic cloud of known height, median |bias|
    0.072 px</div></div>
  <div class="kpi good"><div class="v">325–1471 m</div>
    <div class="l">error of the shipped <code>lowest</code> selection rule on ground truth
    (replaced: ≤64 m)</div></div>
  <div class="kpi"><div class="v">12</div>
    <div class="l">hypotheses refuted, each against a criterion stated before the
    measurement</div></div>
  <div class="kpi"><div class="v">unattributed</div>
    <div class="l">the cause of the real-world inversion</div></div>
</div>

<h2 id="s1"><span class="n">1</span>The validated instrument</h2>

<p>Two 180° fisheye cameras, 1920×1920, baseline 45&nbsp;m, 10&nbsp;s cadence, on
2025-05-23. The reference is a DOL-2 laser ceilometer whose log is stamped MSK; the image
filenames are UTC; the offset is exactly +3&nbsp;h, established twice independently — by the
diurnal brightness envelope and by a solar ephemeris matching the Sun's image azimuth to
better than 0.5° across fourteen hours.</p>

<p>Three calibration terms were <em>measured</em> rather than assumed, each by more than one
method:</p>

<table>
<tr><th>term</th><th>measured value</th><th>how</th></tr>
<tr><td>focal constant</td><td><strong>654 ± 11 px/rad</strong> (assumed: 611)</td>
<td>five independent estimates — four-stream Sun-radius fits on both cameras and both days,
plus the clear-day Sun metric</td></tr>
<tr><td>residual roll</td><td><strong>−1.57 ± 0.10°</strong></td>
<td>three — disparity-field fit, spread minimisation, infinity-test residuals</td></tr>
<tr><td>cubic lens distortion</td><td><strong>a = −0.30</strong> in θ = (r/f)(1+a(r/f)²)</td>
<td>fitted from the sky's own flatness requirement; stable against f; confirmed end-to-end</td></tr>
</table>

{FIG['r2_radial']}

<h3>The synthetic control, and why it took two attempts</h3>
<p>A renderer casts rays through the measured fisheye geometry onto a cloud plane at a known
height, samples a procedural texture in <em>metres</em>, and warps camera 1 by the shipped
affine exactly as production does. Its rig geometry was <strong>derived from the
calibration</strong>, and one term took a withdrawn attempt to find: the affine maps camera
1's principal point onto camera 2's, so <code>c2 = R·c1 + t = (1014.10, 965.03)</code>,
<strong>not the image centre</strong>. With both principal points assumed central, the
rendered parallax comes out negative and grows with height.</p>

<div class="pre"><strong>Renderer gate</strong> (named in <code>w7_renderer.validate()</code>):
<code>du(zenith)</code> must equal <code>f·atan(B/H)</code> and fall as 1/H.
Measured 29.428 / 14.715 / 7.357&nbsp;px at 1000 / 2000 / 4000&nbsp;m against analytic
29.410 / 14.713 / 7.357&nbsp;px. <strong>Passes exactly.</strong></div>

{FIG['w7_control']}

<h3>The shipped selection rule was a catastrophic defect</h3>
<p>The production chain clustered the 2-D disparity vectors with HDBSCAN
(<code>min_cluster_size=3</code>) and reported the <em>lowest</em> cluster of size ≥&nbsp;8 —
an extreme low-tail order statistic of the measurement noise, and one whose value depends on
how many points there are.</p>

<table>
<tr><th>selector</th><th>ρ(true, reported)</th><th>p</th><th>error range</th></tr>
<tr><td>field median</td><td>+1.000</td><td>0.000</td><td>−56 to +24 m</td></tr>
<tr><td><strong><code>lowest</code></strong> (shipped)</td>
    <td><strong>+0.700</strong></td><td><strong>0.188 (n.s.)</strong></td>
    <td><strong>−1471 to −325 m</strong></td></tr>
<tr><td><strong><code>lowest_mode</code></strong> (replacement)</td>
    <td><strong>+1.000</strong></td><td>0.000</td><td>−64 to +25 m</td></tr>
</table>

<p>It reports <strong>675&nbsp;m for a 1000&nbsp;m cloud</strong>, 722&nbsp;m for 1500&nbsp;m
and 529&nbsp;m for 2000&nbsp;m. The replacement works on the per-match height distribution and
reports the lowest <em>prominent</em> mode — one carrying ≥10&nbsp;% of the weight and
≥50&nbsp;matches — so it depends on where the data concentrate rather than how many points
exist. It is now the project default.</p>

<div class="lim"><span class="limtag">SCOPE</span> The control establishes capability on
<strong>synthetic procedural texture at a known plane</strong>. It models geometry and
occlusion. It applies <em>no phase function</em>, and the surface is smooth. Nothing here
establishes performance on real cloud radiance.</div>

<h2 id="s2"><span class="n">2</span>The measured anomaly</h2>

<p>On the real session the estimator does not recover cloud base height. Stratified by the
ceilometer's own reported regime, it places higher clouds at <strong>lower</strong>
heights:</p>

<table>
<tr><th>regime</th><th>n</th><th>reported height</th><th>reference H1</th></tr>
<tr><td>low (H1 &lt; 2750 m)</td><td>215</td><td>2361 m</td><td>2250 m</td></tr>
<tr><td>high (H1 &gt; 3500 m)</td><td>120</td><td>2064 m</td><td>3790 m</td></tr>
</table>

<p>The shift is <strong>−297&nbsp;m</strong> in the reported height while the reference rises
<strong>+1540&nbsp;m</strong>. Rank correlation with the reference is
<strong>−0.547 (p = 4e−14)</strong>; a correct estimator requires a positive sign.</p>

{FIG['flat_swing']}

<h3>The signature: a reshape, not a translation</h3>

<table>
<tr><th>percentile</th><th>low regime</th><th>high regime</th><th>shift</th></tr>
<tr><td>1 %</td><td>−1.88 px</td><td>−9.07 px</td><td><strong>−7.19</strong></td></tr>
<tr><td>10 %</td><td>8.33</td><td>7.85</td><td>−0.48</td></tr>
<tr><td>25 %</td><td>10.22</td><td>11.09</td><td>+0.87</td></tr>
<tr><td><strong>50 %</strong></td><td>12.27</td><td><strong>14.07</strong></td>
    <td><strong>+1.80</strong></td></tr>
<tr><td>75 %</td><td>14.33</td><td>15.65</td><td>+1.32</td></tr>
<tr><td>90 %</td><td>15.96</td><td>17.77</td><td>+1.82</td></tr>
<tr><td>99 %</td><td>35.16</td><td>72.67</td><td><strong>+37.51</strong></td></tr>
<tr><td><strong>1–99 span</strong></td><td><strong>37.0 px</strong></td>
    <td><strong>81.7 px</strong></td><td><strong>×2.2</strong></td></tr>
<tr><td>IQR (25–75)</td><td>4.11 px</td><td>4.56 px</td><td><strong>×1.11</strong></td></tr>
</table>

<p><strong>Both tails extend while the core shifts by about +1.5&nbsp;px.</strong> The
perpendicular epipolar residual <code>dv</code> doubles in the same way
(0.866 → 1.885&nbsp;px, p90 3.35 → 6.33, KS p ≈ 0) — two statistics on two different
components agreeing that the high regime's matching is more dispersed.</p>

{FIG['r11']}

<h3>It survives the obvious controls</h3>
<table>
<tr><th>control</th><th>result</th></tr>
<tr><td>spatial scale</td><td>present at every radial cut: ρ = +0.540 at r ≤ 100 px and
+0.547 at r ≤ 452 px</td></tr>
<tr><td>diurnal confound</td><td>ρ(hour, reference) = +0.114 (n.s.); partial
ρ(reference, disparity | hour removed) = <strong>+0.498 (p = 1.6e−11)</strong></td></tr>
<tr><td>within a single layer</td><td>low-regime-only ρ = <strong>−0.473 (p = 2.3e−13)</strong>,
gridded <strong>−0.525 (p = 0.030)</strong></td></tr>
<tr><td>removing the instrumental field</td><td><strong>strengthens</strong> the inversion
(−129 → −164 m)</td></tr>
<tr><td>choice of matcher</td><td>LoFTR sees the same sign, 6.5× smaller</td></tr>
</table>

<h2 id="s3"><span class="n">3</span>Twelve refutations</h2>

<p>Each hypothesis was tested against a criterion stated <strong>before</strong> the
measurement, recorded in <code>docs/plan/research-workplan.md</code> and in the individual
records. The gate is named for each, so no metric can be read as post-hoc.</p>

<table>
<tr><th>#</th><th>hypothesis</th><th>discriminating measurement</th><th>refuted because</th>
<th>record</th></tr>
{refrow(1, "certainty / low-confidence hallucination", "disparity at matched certainty",
        "the high regime is higher at <em>every</em> of ten certainty deciles (+0.82 to "
        "+2.50 px); the surplus population's median certainty is <strong>0.782</strong> — "
        "high, not low", "w4_certainty.py")}
{refrow(2, "incoherent matching noise", "residual against the 25 nearest neighbours",
        "0.047 px (low) vs 0.041 px (high), ratio <strong>0.865</strong> — the high "
        "regime is <em>more</em> coherent", "w4_smoothness.py")}
{refrow(3, "calibration / geometric error", "remove the instrumental component, re-measure",
        "the inversion <strong>strengthens</strong>: −129 → −164 m, ρ −0.547 → −0.581",
        "w3_unmask.py")}
{refrow(4, "matched-population size", "raw per-match rate",
        "<strong>identical</strong>: 11,655.1 vs 11,655.8 per frame — RoMa emits a fixed "
        "grid", "dump_matches.py")}
{refrow(5, "a pure along-epipolar depth change", "predicted vs observed shift",
        "the reference predicts <strong>−5.315 px</strong>, observation "
        "<strong>+1.798 px</strong> — opposite sign, 15 % of magnitude; the perpendicular "
        "residual also doubles", "w4_gate3.py")}
{refrow(6, "a RoMa-specific artefact", "an independent matcher",
        "LoFTR <strong>+0.275 px, CI [+0.112, +0.401]</strong> — same sign, 6.5× smaller; "
        "matcher-independent in direction, representation-dependent in size",
        "w4_gate4.py")}
{refrow(7, "target-identity layering", "low-regime-only correlation",
        "<strong>−0.473 (p = 2.3e−13)</strong> — the inversion persists with both "
        "instruments on one layer; layering explains the <em>level</em> (+111 m), not the "
        "<em>sign</em>", "w4_lowregime.json")}
{refrow(8, "camera desync / cloud motion", "does <code>dv</code> scale with cloud speed?",
        "ρ(speed, dv) = <strong>+0.002 (p = 0.97)</strong> against ρ(regime, dv) = +0.447; "
        "motion is identical across regimes while dv doubles; the required exposure offset "
        "is <strong>18.14 s</strong> against an archive maximum of 5 s", "w4_desync.py")}
{refrow(9, "a fixed pipeline artefact", "the cloudless-day null",
        "the clear day gives <strong>655 m</strong>, not the cloudy day's 2242 m "
        "(KS p = 0.000) — the level is content-driven", "matches_roma_20250522.npz")}
{refrow(10, "non-Lambertian photometry", "sign of the within-frame brightness coupling",
        "ρ(brightness, du) = <strong>−0.176</strong> — brighter means lower disparity, "
        "while the high regime is brighter and has higher disparity; the gradient-direction "
        "signature is absent (ρ = −0.000)", "w6_photometric.py")}
{refrow(11, "photometric superposition", "tails grow at a fixed core",
        "it widens the <strong>core</strong> (IQR ×9.19) where reality widens the "
        "<strong>tails</strong> (IQR ×1.11); and it needs <strong>80 % clear sky</strong> "
        "against the real 2.6–3.6 %", "w8_superposition.py")}
{refrow(12, "phase aliasing", "periodic structure in the disparity histogram",
        "<strong>no period peaks</strong> above the 0.05 autocorrelation floor in either "
        "regime; a +1.8 px jump needs a ~1.8 px wavelength against 100–1000 px cloud "
        "features", "W8_PHASE_ALIASING.md")}
</table>

<div class="note ok"><strong>Two of the project's own results were retracted first.</strong>
The original validation metric — a windowed correlation that searched ±300 s per window and
reported the maximum — is <strong>withdrawn</strong>: a circular-roll null performing the
same search scores <em>higher</em> than the observation (0.346 against 0.189), because a
30-minute window holds ~152 paired samples but only <strong>~3.4 independent ones</strong>.
And two of the authors' own claims — an anti-correlation in the estimator's dynamics, and a
claimed improvement from separating the reference's layers — were separately retracted after
the same treatment.</div>

{FIG['w4_cert']}
{FIG['w4_smooth']}
{FIG['w4_desync']}
{FIG['w6']}
{FIG['w8_super']}

<div class="pre"><strong>Pre-registered triggers, named</strong> (guardrail G2 — a metric
chosen after seeing the result is not evidence):<br>
<code>w7_renderer.validate()</code> — <code>du(zenith) = f·atan(B/H)</code>, falling as 1/H<br>
<code>w8_relief.py</code> — ρ(true, reported) &lt; 0.9 across ≥3 heights<br>
<code>w8_superposition.py</code> — 1–99 span ×≥1.8 <em>while</em> IQR ×&lt;1.4<br>
<code>experiments/w2/PREREGISTRATION.md</code> — ≥50 inliers, epipolar RMS &lt; 2.0 px, ≥5 frames<br>
<code>w4_smoothness.py</code> — coherence ratio &lt; 1.10 means a real displacement</div>

<h2 id="s4"><span class="n">4</span>The boundary — what may and may not be claimed</h2>

<h3>What is established</h3>
<p>The instrument recovers a geometrically known height on synthetic procedural texture
through the measured fisheye geometry: <strong>ρ = +1.000, median |bias| = 0.072 px</strong>,
robust to a 4× apparent-scale range and to 20&nbsp;% relief. The shipped
<code>lowest</code> rule was a defect erring by up to 1471&nbsp;m on ground truth;
<code>lowest_mode</code> errs by ≤64&nbsp;m. The real-data inversion is robust and specific
in its distributional signature.</p>

<h3>What is claimed as product capability</h3>
<p>A <strong>level</strong> consistent with the ceilometer's lower cloud population, in the
low regime only:</p>

<table>
<tr><th>regime</th><th>camera</th><th>ceilometer H1</th><th>difference</th></tr>
<tr><td>low (H1 &lt; 2750 m)</td><td>2361 m</td><td>2250 m</td><td><strong>+111 m</strong></td></tr>
<tr><td>high (H1 &gt; 3500 m)</td><td>2064 m</td><td>3790 m</td><td>−1726 m</td></tr>
</table>

{limitation('''<strong>level agreement.</strong> The +111 m figure is a <em>level</em>
statistic from 215 frames of one session, not a validated calibration. To elevate it to a
demonstrated result requires (a) an independently calibrated level reference — the baseline
is <em>assumed</em> at 45 m and never measured, and the focal constant carries ±11 px/rad,
about ±1.7 % in level; (b) a second session, to exclude a coincidence at this site; and
(c) demonstration that the agreement is not produced by the same unattributed process that
produces the inversion at other heights.''')}

<p><strong>No dynamic tracking is claimed.</strong> Within the low regime the estimator's
variation is <em>anti-correlated</em> with the reference.</p>

{limitation('''<strong>dynamics.</strong> ρ = −0.473 (p = 2.3e−13) per frame; −0.525
(p = 0.030) on a grid at the slower series' decorrelation time. The effective sample size is
<strong>7</strong> (nominal 215, lag-1 ρ ≈ 0.94), so the study is underpowered and no finer
structure should be claimed. A dynamics claim in <em>either</em> direction requires a session
yielding ≥30 independent samples of both series, and an independent motion reference — no
wind data exists for this session.''')}

{limitation('''<strong>absolute cloud-base height.</strong> No absolute claim is supported
for high, thin or multi-layer fields. The reference is itself bi-modal (40 % at
2250–2750 m, 53 % at 3750–4250 m) and reports up to 7010 m against a stated 10–3000 m range.
Supporting such a claim requires a reference with a footprint comparable to the camera's
1956 m at 1837 m — a radar or scanning lidar — rather than the ~10 m column used here, a
~200:1 mismatch spanning 8.5 texture decorrelation lengths.''')}

{limitation('''<strong>instrument identity.</strong> That the stereo measures "a valid but
different physical quantity" than the ceilometer is a <em>motivating hypothesis, not a
result</em>. No measurement here identifies which quantity. Establishing it requires a third
instrument with an overlapping footprint and a known retrieval.''')}

{limitation('''<strong>external validity.</strong> One usable session exists. 2025-05-22 is
cloudless and yields single-digit correspondences per hour, so there is no session-level
hold-out, no unseen-session validation and no drift measurement between sessions. Every
result here is a time block of one day.''')}

{limitation('''<strong>the anomaly's cause.</strong> The cause of the inversion is
<strong>unattributed</strong>. Twelve mechanisms were refuted with pre-registered criteria;
none accounts for the observed combination of a +1.5 px core shift with ×2.2 tail growth at
a fixed IQR. It is <em>not</em> a claim that the cause is atmospheric, volumetric or
photometric — two of those were tested and refuted, and the photometric coupling that does
exist runs the <em>wrong way</em>.''')}

<h2 id="s5"><span class="n">5</span>What would settle it</h2>

<p>The analysis side of this question is exhausted: twelve mechanisms tested, twelve refuted,
on a validated instrument. What remains is not another hypothesis but a
<strong>controlled input</strong>.</p>

<ol>
<li><strong>Bisect the renderer against the real signature.</strong> The validated renderer
is now the project's most valuable asset. Any proposed mechanism must reproduce the
<em>specific</em> signature — core +1.5 px, span ×2.2, IQR ×1.11, dv ×2.2 — and not merely a
bias or a spread. Two mechanisms have already failed by producing the wrong part of the
distribution.</li>
<li><strong>A physical target, or an instrumented session.</strong> The foreground control
failed for a quantified geometric reason: an 87–247 px disparity at a 45 m baseline against a
close, oblique building, unmatchable by either SIFT or RoMa. The limit is the rig, not the
algorithm.</li>
<li><strong>A footprint-matched reference.</strong> A radar or scanning lidar co-located with
the cameras would remove the 200:1 sampling mismatch that makes a level comparison between a
point column and a 2 km area difficult to interpret at all.</li>
</ol>

<h2 id="s6"><span class="n">6</span>Methodology, and the harness post-mortem</h2>

<h3>What a claim required</h3>
<p>No value produced by maximising over alternatives without a null that maximises
identically. Effective sample size reported rather than nominal. A p-value, or an explicit
statement that none exists, and the <em>direction</em> of any effect. Synthetic ground truth
validated by an independent exact estimator before any method was judged by it.</p>

<h3>The three harness bugs, reported as findings about the process</h3>
<p>Each changed a scientific conclusion, and none was caught by inspection. They are given
here as first-class results, because in our experience this is where the time actually
goes.</p>

<table>
<tr><th></th><th>what happened</th><th>what it changed</th></tr>
<tr><td class="num">1</td>
<td><strong>A metric that scored below its own null.</strong> The original validation
searched a time shift per window and reported the maximum; a null performing the same search
scored higher (0.346 vs 0.189).</td>
<td>the project's original validation metric was <strong>withdrawn</strong></td></tr>
<tr><td class="num">2</td>
<td><strong>A ground truth verified in the wrong frame.</strong> The first synthetic
renderer was gated in the <em>native</em> frame; the matcher consumes the <em>warped</em>
frame, which was never validated. It also modelled both cameras with the same orientation —
a rig that does not exist.</td>
<td>a <strong>positive control was withdrawn</strong></td></tr>
<tr><td class="num">3</td>
<td><strong>A default argument that cost 551 m.</strong> Two scripts rendered the "same"
synthetic cloud 1.5° apart, because one call passed <code>roll_deg</code> implicitly and the
other did not. The control refused to reproduce — 1984 m against 1433 m on inputs believed
identical.</td>
<td>three hypotheses were proposed <strong>before the call sites were compared</strong></td></tr>
</table>

<blockquote><strong>What they have in common.</strong> Each is a case of a quantity being
verified in a proxy that was not the thing consumed. The null is a proxy for the search; the
native frame is a proxy for the warped frame; one conversion path is a proxy for the other.
<strong>In every case the verification passed and the conclusion was still wrong.</strong>
We would recommend that any pipeline claiming a synthetic control state explicitly, for each
gate, <em>which artefact the consumer actually reads</em>.</blockquote>

<h3>A note on the shape of this result</h3>
<p>Twelve refutations with pre-registered triggers, on an instrument whose capability was
demonstrated independently, is a bounded null result: it does not say the measurement is
impossible, it says this instrument, on this data, under every mechanism we could formulate
and test, does not measure what the reference measures. The instrument is sound. The
anomaly is real and specific. Its cause is open — and it is now stated as open, rather than
filled with the most attractive available mechanism.</p>

<footer>
<strong>Artefacts.</strong>
<a href="../plan/research-workplan.md">docs/plan/research-workplan.md</a> — the full record,
all twelve refutations ·
<a href="../results/README.md">docs/results/</a> — citable index, every negative result, and
the matcher-selection principle ·
<a href="final-report.md">final-report.md</a> — this document in markdown ·
<a href="../../../experiments/w7/w7_renderer.py">experiments/w7/w7_renderer.py</a> — the
validated renderer.
<br><br>
<strong>Data.</strong> One usable session, 2025-05-23, 17,265 frames at 10 s. 2025-05-22 is
cloudless (8,632 frames) and serves as a control and calibration reference. No meteorology,
no wind data, and no site coordinates exist for either session.
</footer>
</div></body></html>"""
    return html


def main():
    html = build()
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
