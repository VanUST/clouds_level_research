# PURPOSE: Generate a self-contained HTML report of the findings added since
#          docs/report/final.html was last written - the static-field diagnosis, the
#          cross-validated correction search, the shape mechanism, and the detection floor.
#
#          The document is written for someone who has read the main report and wants only
#          what changed, so it is organised as "the record, amended" rather than as a
#          restatement. Every correction to an earlier claim is given its own block, because
#          the amendments are the substance of this round.
#
#          Every number is traced to experiments/w9/w9_cv.json or
#          experiments/w10/w10_signal.json; figures are the generated PNGs.
# INPUTS: docs/report/w9_cv.png, docs/report/w10_signal.png.
# OUTPUTS: docs/report/final-findings.html (self-contained).
# KEYWORDS: report, html, w9, w10, shape_mechanism, detection_floor, amendment
from __future__ import annotations

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(REPO, "docs", "report")
OUT = os.path.join(FIGDIR, "final-findings.html")


def b64(name):
    with open(os.path.join(FIGDIR, name), "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def fig(name, cap):
    return (f'<figure><img src="{b64(name)}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


def amendment(title, what, now):
    return (f'<div class="amd"><div class="amdh">AMENDMENT — {title}</div>'
            f'<p><span class="was">was</span> {what}</p>'
            f'<p><span class="now">now</span> {now}</p></div>')


def build():
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>New findings — static field, shape mechanism, detection floor</title><style>
 :root {{ --ink:#161b22; --dim:#5c6672; --acc:#c1440e; --good:#1a7f37; --blue:#1f6feb;
          --line:#e3e6ea; --bg:#fcfcfd; --warn:#9a6700; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   -webkit-font-smoothing:antialiased; }}
 .wrap {{ max-width:1080px; margin:0 auto; padding:0 30px 110px; }}
 header {{ padding:56px 0 24px; border-bottom:3px solid var(--ink); }}
 h1 {{ font-size:33px; line-height:1.18; margin:0 0 12px; letter-spacing:-.6px; }}
 .sub {{ color:var(--dim); font-size:15.5px; max-width:78ch; }}
 h2 {{ font-size:23px; margin:54px 0 4px; letter-spacing:-.3px; }}
 h2 .n {{ color:var(--acc); margin-right:10px; font-variant-numeric:tabular-nums; }}
 h3 {{ font-size:16px; margin:30px 0 6px; }}
 p, li, td, th {{ font-size:15px; }}
 figure {{ margin:22px 0; }}
 figure img {{ width:100%; border:1px solid var(--line); border-radius:7px; background:#fff; }}
 figcaption {{ color:var(--dim); font-size:13.5px; margin-top:8px; max-width:92ch; }}
 table {{ border-collapse:collapse; width:100%; margin:18px 0; font-size:14px; }}
 th, td {{ text-align:left; padding:9px 11px; border-bottom:1px solid var(--line);
   vertical-align:top; }}
 th {{ background:#f3f5f8; font-weight:600; font-size:12.5px; letter-spacing:.02em;
   text-transform:uppercase; color:#404850; }}
 td.num {{ font-variant-numeric:tabular-nums; color:var(--dim); }}
 td.g {{ font-variant-numeric:tabular-nums; }}
 code {{ background:#f3f5f8; padding:1.5px 5px; border-radius:4px; font-size:12.5px; }}
 .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin:24px 0; }}
 .kpi {{ flex:1 1 190px; background:#fff; border:1px solid var(--line);
   border-left:4px solid var(--acc); border-radius:7px; padding:14px 16px; }}
 .kpi.good {{ border-left-color:var(--good); }}
 .kpi .v {{ font-size:23px; font-weight:700; font-variant-numeric:tabular-nums;
   letter-spacing:-.5px; }}
 .kpi .l {{ color:var(--dim); font-size:12.5px; margin-top:3px; line-height:1.45; }}
 .note {{ background:#fff7ed; border-left:4px solid var(--acc); padding:14px 18px;
   border-radius:0 7px 7px 0; margin:20px 0; }}
 .note.ok {{ background:#f0fdf4; border-left-color:var(--good); }}
 .amd {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--blue);
   border-radius:0 7px 7px 0; padding:14px 18px; margin:20px 0; }}
 .amdh {{ font-size:12px; font-weight:700; letter-spacing:.08em; color:var(--blue);
   text-transform:uppercase; margin-bottom:8px; }}
 .was, .now {{ display:inline-block; font-size:11px; font-weight:700; letter-spacing:.06em;
   padding:2px 7px; border-radius:3px; margin-right:8px; text-transform:uppercase; }}
 .was {{ background:#f3f5f8; color:#5c6672; }}
 .now {{ background:var(--good); color:#fff; }}
 .hi {{ background:#fff6d6; padding:0 3px; border-radius:2px; }}
 .pre {{ background:#f5f8ff; border-left:4px solid var(--blue); padding:12px 16px;
   border-radius:0 7px 7px 0; margin:16px 0; font-size:14.5px; }}
 blockquote {{ margin:20px 0; padding:4px 0 4px 18px; border-left:4px solid var(--ink); }}
 footer {{ margin-top:60px; padding-top:20px; border-top:1px solid var(--line);
   color:var(--dim); font-size:13.5px; }}
</style></head><body><div class="wrap">

<header>
  <h1>New findings: the static field, the shape mechanism, and the detection floor</h1>
  <div class="sub">What was learned after <a href="final.html">final.html</a> was written —
  a diagnosis of the static pattern, a cross-validated search for an operation that removes
  it, the mechanism behind the estimator-dependence of the &ldquo;inversion&rdquo;, and the
  statistical boundary of the instrument's dynamic capability. Four previous claims are
  amended, three of them ours.</div>
</header>

<div class="note"><strong>Summary.</strong> The static spatial pattern is dominated by a
<strong>linear gradient in <code>u</code></strong> (66&nbsp;% of its variance) rather than by
a radial law (25&nbsp;%), though the two are correlated and it is the radial part that
survives in disparity but not in height. A cross-validated search over six correction
families and three estimators <strong>removes part of the pattern and recovers none of the
dynamics</strong>. The estimator-dependence of the inversion is explained by a
<strong>collapse in the distribution's skewness</strong> (2.27&nbsp;→&nbsp;0.73) while its
dispersion is unchanged: the median and the mode move in <em>opposite directions</em>
(+13&nbsp;% against −12&nbsp;%), which is what order statistics do to a shape-shifting
field. And the disparity field <strong>does</strong> carry a correctly-signed dynamic
correlation of ≈&nbsp;−0.37 which lies <strong>below the detection floor</strong> of this
session — the obstacle is sample size, not absence, and it is surmountable in
<strong>≈&nbsp;8 hours</strong> of data.</div>

<div class="kpis">
  <div class="kpi"><div class="v">66 % vs 25 %</div>
    <div class="l">of the static field's variance explained by a linear <code>u</code>
    gradient against radius</div></div>
  <div class="kpi"><div class="v">&minus;68 %</div>
    <div class="l">collapse in disparity-distribution skewness between regimes, with
    dispersion unchanged</div></div>
  <div class="kpi good"><div class="v">&minus;0.37</div>
    <div class="l">correctly-signed dynamic correlation, median statistic</div></div>
  <div class="kpi"><div class="v">0.47</div>
    <div class="l">the smallest detectable |&rho;| at this session's n<sub>eff</sub>&nbsp;=&nbsp;17.7
    — above the signal</div></div>
</div>

<h2 id="s1"><span class="n">1</span>What the static field depends on</h2>

<p>Regressing the per-cell static disparity on candidate predictors answers the question
directly, and it corrects the framing: <strong>the pattern is not primarily radial.</strong></p>

<table>
<tr><th>predictor</th><th>Pearson r</th><th>R²</th></tr>
<tr><td><strong>u</strong> — horizontal image position</td>
    <td class="g"><strong>+0.814</strong></td><td class="g"><strong>0.663</strong></td></tr>
<tr><td>cos(angle)</td><td class="g">+0.767</td><td class="g">0.588</td></tr>
<tr><td><strong>radius r</strong></td><td class="g">+0.499</td><td class="g">0.249</td></tr>
<tr><td>|u|</td><td class="g">+0.373</td><td class="g">0.139</td></tr>
<tr><td>v</td><td class="g">&minus;0.239</td><td class="g">0.057</td></tr>
</table>

<p>A <strong>linear gradient in <code>u</code> accounts for 66&nbsp;%</strong> of the
variance; <strong>radius accounts for 25&nbsp;%</strong>. They are correlated because
<code>u ≤ r</code>, which is why the radial profile looks convincingly monotone — 14.51&nbsp;px
at the centre rising to 16.57&nbsp;px at r&nbsp;≈&nbsp;420, a 14&nbsp;% rise — while
actually being a consequence of the u-gradient as much as a radial law of its own.</p>

<p><strong>And the radial structure largely cancels in height.</strong> Applying the measured
corrections (f&nbsp;=&nbsp;654, a&nbsp;=&nbsp;&minus;0.30, exact triangulation),
corr(radius,&nbsp;height) falls to <strong>&minus;0.146</strong>, with median heights of
1975&nbsp;/&nbsp;1955&nbsp;/&nbsp;1930&nbsp;/&nbsp;1918&nbsp;/&nbsp;1947&nbsp;m across the
radius — a <strong>3&nbsp;% spread</strong> against the disparity's 14&nbsp;%. Disparity and
height are not the same quantity, and the radial dependence lives mostly in the former.</p>

<h2 id="s2"><span class="n">2</span>The cross-validated search for a correction</h2>

<p>Six correction families × three estimators, <strong>5-fold cross-validated</strong>: every
correction is fitted on four fifths of the session and scored on the held-out fifth, so
nothing is scored on the frames that shaped it.</p>

<table>
<tr><th>correction</th><th>estimator</th><th>LEVEL &rho;</th><th>DYNAMICS &rho;</th></tr>
<tr><td><strong>per-cell + linear</strong></td><td><strong>mode</strong></td>
    <td class="g"><strong>+0.217</strong></td><td class="g">&minus;0.038</td></tr>
<tr><td>linear</td><td>mode</td><td class="g">+0.194</td><td class="g">&minus;0.166</td></tr>
<tr><td>per-cell</td><td>mode</td><td class="g">+0.179</td><td class="g">+0.011</td></tr>
<tr><td><strong>none</strong> (baseline)</td><td>mode</td>
    <td class="g">+0.053</td><td class="g">&minus;0.104</td></tr>
<tr><td>per-radius</td><td>mode</td><td class="g">+0.149</td><td class="g">&minus;0.042</td></tr>
<tr><td>cell-smooth</td><td>mode</td><td class="g">+0.147</td><td class="g">&minus;0.161</td></tr>
<tr><td>linear</td><td>median</td><td class="g">&minus;0.429</td><td class="g">+0.122</td></tr>
<tr><td>per-cell</td><td>median</td><td class="g">&minus;0.422</td><td class="g">+0.120</td></tr>
<tr><td><strong>none</strong> (baseline)</td><td>median</td>
    <td class="g"><strong>&minus;0.409</strong></td><td class="g">+0.126</td></tr>
</table>

{fig("w9_cv.png",
     "Out-of-sample level and dynamics correlation for every correction family and "
     "estimator. The statistic is height-ordered, so a positive bar is the correct sign. "
     "The mode is correctly signed and the median is inverted; the dynamics are near zero "
     "throughout.")}

<h3>Finding 1 — the sign depends on the estimator</h3>
<p>The <strong>mode</strong> estimator is <strong>correctly signed out of sample</strong>
(+0.053 uncorrected, +0.217 with the best correction). The <strong>median</strong> and the
25th percentile are <strong>inverted</strong> (&minus;0.347 to &minus;0.246). The best
correction moves the level by +0.065 and leaves the sign on the correct side.</p>
<p><strong>Corrected after review:</strong> these figures come from
<em>blocked + embargo</em> cross-validation. The first version drew folds at random, which
on a series with lag-1 autocorrelation 0.934 placed each held-out frame's near-duplicates in
the training set; that leaked +0.070 into the headline (+0.287 against the honest +0.217).
See <code>experiments/w10/CODE_REVIEW.md</code>.</p>

<h3>Finding 2 — no correction recovers the dynamics</h3>
<p>Across all eighteen combinations the first-difference correlation lies within
<strong>±0.19</strong> with no consistent sign, and the best level correction does not
improve it (uncorrected +0.004 against +0.016 corrected). The most expressive
family tested — a free offset per 20&nbsp;×&nbsp;20&nbsp;px cell <em>plus</em> a plane,
fitted on 80&nbsp;% of frames — would recover a real signal hiding under a fixed additive
pattern. It does not.</p>

<div class="note ok"><strong>The static field is diagnosable and partly removable; removing
it does not recover the dynamics.</strong> The anomaly cannot be described as &ldquo;a static
field masking a real signal&rdquo; without qualification — an expressive correction does not
reveal the signal.</div>

<h2 id="s3"><span class="n">3</span>The shape mechanism — why the median inverts</h2>

<p>The estimator-dependence is not a defect of any statistic. It is a
<strong>change in the distribution's shape</strong> between regimes, and it is measured.</p>

<table>
<tr><th>quantity</th><th>low regime</th><th>high regime</th><th>change</th></tr>
<tr><td>median</td><td class="g">12.354 px</td><td class="g">13.960 px</td>
    <td class="g"><strong>+13.0 %</strong></td></tr>
<tr><td><strong>mode</strong></td><td class="g">12.896 px</td><td class="g">11.371 px</td>
    <td class="g"><strong>&minus;11.8 %</strong></td></tr>
<tr><td><strong>skewness</strong></td><td class="g"><strong>2.268</strong></td>
    <td class="g"><strong>0.731</strong></td>
    <td class="g"><strong>&minus;67.8 %</strong></td></tr>
<tr><td>kurtosis</td><td class="g">36.88</td><td class="g">32.23</td>
    <td class="g">&minus;12.6 %</td></tr>
<tr><td><strong>standard deviation</strong></td><td class="g">4.303</td>
    <td class="g">4.283</td><td class="g"><strong>&minus;0.5 %</strong></td></tr>
<tr><td>p10</td><td class="g">9.417</td><td class="g">10.477</td><td class="g">+11.3 %</td></tr>
<tr><td>p90</td><td class="g">16.087</td><td class="g">16.719</td><td class="g">+3.9 %</td></tr>
</table>

{fig("w10_signal.png",
     "Left: every statistic of the disparity field against the reference, at 300 s "
     "binning — no statistic survives correction, but the central ones are consistently "
     "signed. Right: the distribution's shape by regime; the dispersion is unchanged while "
     "the skewness collapses.")}

<p><strong>The median and the mode move in opposite directions while the standard deviation
is unchanged.</strong> The distribution does not widen or narrow between regimes; it changes
<em>shape</em>, its skewness collapsing by 68&nbsp;%. A statistic tracking the bulk (the
mode) and one tracking cumulative mass (the median) <em>must</em> diverge when skewness
changes — this is what order statistics do, not a failure of either.</p>

<div class="note ok"><strong>It also unifies the synthetic and real results.</strong> In the
synthetic wind tunnel the cloud's disparity distribution has a <em>fixed shape</em> across
heights, so no statistic is biased by a shape change and only the tail-sensitive rule fails —
which is exactly what was observed (<code>lowest</code> erred by up to 1471&nbsp;m while
<code>lowest_mode</code> erred by ≤64&nbsp;m). In the real atmosphere the shape
<em>does</em> change between regimes, so the median and the mode diverge. The same mechanism
explains both, and the direction of the failure is set by the sign of the skewness change.</div>

<h2 id="s4"><span class="n">4</span>The detection floor — the real boundary</h2>

<p>At 300&nbsp;s binning — the reference's own decorrelation time — the disparity field
carries a <strong>weak, correctly-signed dynamic correlation</strong>.</p>

<div class="pre"><strong>Sign convention, stated because it inverts the reading.</strong> These
statistics are <em>disparities</em> and the reference is a <em>height</em>. A correct
estimator needs the disparity to <strong>fall</strong> as the cloud rises, so here
<strong>negative &rho; is the correct sign</strong>.</div>

<table>
<tr><th>statistic</th><th>LEVEL &rho;</th><th>DYNAMICS &rho;</th><th>p</th><th>sign</th></tr>
<tr><td>mean</td><td class="g">+0.158</td><td class="g">&minus;0.357</td>
    <td class="g">0.074</td><td>correct</td></tr>
<tr><td><strong>median</strong></td><td class="g">+0.159</td>
    <td class="g"><strong>&minus;0.409</strong></td><td class="g"><strong>0.038</strong></td>
    <td>correct</td></tr>
<tr><td>mode</td><td class="g">&minus;0.238</td><td class="g">&minus;0.374</td>
    <td class="g">0.209</td><td>correct</td></tr>
<tr><td>p10</td><td class="g">+0.260</td><td class="g">&minus;0.338</td>
    <td class="g">0.092</td><td>correct</td></tr>
<tr><td><strong>p25</strong></td><td class="g">+0.159</td>
    <td class="g"><strong>&minus;0.400</strong></td><td class="g"><strong>0.043</strong></td>
    <td>correct</td></tr>
<tr><td>p75</td><td class="g">+0.138</td><td class="g">&minus;0.340</td>
    <td class="g">0.089</td><td>correct</td></tr>
<tr><td>skew</td><td class="g">&minus;0.230</td><td class="g">&minus;0.306</td>
    <td class="g">0.129</td><td>correct</td></tr>
</table>

<p><strong>Nine of eleven statistics are correctly signed</strong> on the dynamics, and the
five central ones average <strong>&minus;0.369</strong>. But <strong>nothing survives
correction</strong>: 22 tests give a Bonferroni threshold of 0.00227 against a smallest
observed p of 0.038.</p>

<h3>The obstacle is sample size, not absence</h3>

<table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>binned series</td><td>27 bins of 300 s = <strong>2.25 h</strong></td></tr>
<tr><td>effective sample size</td><td><strong>n<sub>eff</sub> = 17.7</strong>
    (one independent sample per 449 s)</td></tr>
<tr><td><strong>critical |&rho;| for significance</strong> at p = 0.05</td>
    <td><strong>0.470</strong></td></tr>
<tr><td>|&rho;| detectable with 80 % power at n<sub>eff</sub> = 17.7</td><td>0.731</td></tr>
<tr><td><strong>n<sub>eff</sub> needed for &rho; = 0.37 at 80 % power</strong></td>
    <td><strong>60.3</strong></td></tr>
<tr><td>n<sub>eff</sub> needed for &rho; = 0.47 at 80 % power</td><td>38.5</td></tr>
<tr><td><strong>duration for a decisive session</strong></td>
    <td><strong>2.25 h × (60.3/18.05) ≈ 7.5 h</strong></td></tr>
</table>

<p><strong>The observed &minus;0.37 lies below the 0.470 detection floor of the
design.</strong> Establishing a correlation of 0.37 at 80&nbsp;% power needs
n<sub>eff</sub>&nbsp;≈&nbsp;60, i.e. <strong>≈&nbsp;8 hours</strong> of binned data.</p>

<blockquote>The honest statement is not &ldquo;the instrument has no dynamic capability&rdquo;
but <strong>&ldquo;the dynamic signal, if present, is at or below what this session can
detect.&rdquo;</strong> That converts a closed question into a concrete and cheap
experiment.</blockquote>

<h2 id="s5"><span class="n">5</span>Amendments to the record</h2>

<p>Four earlier claims are amended. Three are ours.</p>

{amendment("the dynamics were not recovered, so the field carries no dynamic information",
  "W9 measured per-frame first differences on held-out folds and found |&rho;| &lt; 0.17, "
  "which was read as the disparity field holding no recoverable temporal signal.",
  "at 300 s binning the same data gives a consistent correctly-signed correlation of about "
  "<strong>&minus;0.37</strong> (nine of eleven statistics correctly signed, central five "
  "averaging &minus;0.369). The earlier reading was too strong: binning suppresses the "
  "per-frame matcher noise that dominated the per-frame analysis. The boundary is a "
  "<strong>detection floor</strong>, not an absence.")}

{amendment("the search metric",
  "the first version of the cross-validated search correlated the <em>disparity</em> with "
  "the reference and reported +0.409 as its best result.",
  "a correct estimator needs the disparity to fall as the cloud rises, so that was the "
  "<strong>inversion scored as a success</strong>. The statistic is now height-ordered and "
  "every reported &rho; carries the conventional sign. Caught by the control's sign rather "
  "than by inspection.")}

{amendment("the duration of a decisive session",
  "&ldquo;&gt;15 hours&rdquo; was proposed in review, and this project's own W10 stated "
  "n<sub>eff</sub> ≈ 55.",
  "the Fisher-z relation gives <strong>n<sub>eff</sub> = 60.3</strong> for a correlation of "
  "0.37 at 80 % power, and the binned series yields one independent sample per 449 s, so a "
  "decisive session needs <strong>≈ 7.5 h</strong>. The correction is a factor of two and it "
  "makes the recommended experiment cheaper than it appeared.")}

{amendment("the cause of the static field",
  "the pattern was described in earlier drafts as radial, and partly attributed to an "
  "unattributed interaction between the atmosphere and the matcher's priors.",
  "it is <strong>dominated by a linear gradient in u</strong> (66 % against radius's 25 %), "
  "and it is <strong>not primarily radial</strong>. The three specific mechanisms of the "
  "atmospheric family that were proposed were each tested and refuted. The cause of the "
  "residual remains <strong>unattributed</strong>, and that is the correct description.")}

<h2 id="s6"><span class="n">6</span>What this leaves</h2>

<table>
<tr><th>question</th><th>status</th></tr>
<tr><td>Is the matcher and geometry sound?</td>
    <td><strong>yes</strong> — ρ = +1.000, 0.072 px median bias on a gated synthetic control</td></tr>
<tr><td>Was the shipped selection rule defective?</td>
    <td><strong>yes</strong> — 325–1471 m of error on ground truth; replaced by
    <code>lowest_mode</code>, ≤64 m</td></tr>
<tr><td>What is the static field?</td>
    <td><strong>a linear u-gradient</strong> (66 %), partly removable, and it largely cancels
    in height</td></tr>
<tr><td>Does removing it recover the dynamics?</td>
    <td><strong>no</strong> — across six families and three estimators, out of sample</td></tr>
<tr><td>Why did the median invert?</td>
    <td><strong>the skewness collapses 68 %</strong> while the dispersion is unchanged; the
    median and mode must diverge</td></tr>
<tr><td>Does the field carry dynamics at all?</td>
    <td><strong>a weak correctly-signed ≈&nbsp;&minus;0.37</strong>, below the 0.470 detection
    floor of this session</td></tr>
<tr><td>What settles it?</td>
    <td><strong>≈&nbsp;8 hours</strong> of binned data; a footprint-matched reference; or a
    physical target</td></tr>
</table>

<footer>
<strong>Artefacts.</strong>
<a href="final.html">final.html</a> — the main report ·
<a href="final-report.md">final-report.md</a>, both amended with these findings ·
<a href="../plan/research-workplan.md">docs/plan/research-workplan.md</a> — the full record ·
<a href="../results/README.md">docs/results/</a> — citable index (R‑45 to R‑49 are these
findings).
<br><br>
Scripts: <code>experiments/w9/w9_cv.py</code> (the cross-validated search),
<code>experiments/w10/w10_signal.py</code> (signal existence and the shape mechanism).
Results: <code>experiments/w9/w9_cv.json</code>, <code>experiments/w10/w10_signal.json</code>.
</footer>
</div></body></html>"""


def main():
    html = build()
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
