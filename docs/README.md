# Documentation

Four kinds of document, kept separate because they answer different questions.

| Directory | Question it answers | Author |
|---|---|---|
| [`results/`](results/README.md) | **Paper-usable results index** — each result with the artefact and command that reproduces it. |
| **[`../paper/main.pdf`](../paper/main.pdf)** | **The paper** — self-contained 15-page account of the whole programme, written for a reader with no prior context. LaTeX source at `paper/main.tex`. |
| **[`report/final-findings.html`](report/final-findings.html)** | **New findings** — the static-field diagnosis, the cross-validated correction search, the shape mechanism, and the detection floor, with four amendments to the record. |
| **[`report/final.html`](report/final.html)** | **The final report, rendered** — figures embedded, self-contained, opens in a browser. |
| [`report/final-report.md`](report/final-report.md) | **The final report** — a validated instrument, twelve pre-registered refutations, and a bounded null result. Start here for the science. |
| [`briefing.md`](briefing.md) | **Self-contained briefing for an external consultation** — task, data, instruments, algorithm, calibration, processing, validation, findings, and the specific questions asked of a supervisor. |
| [`report/state.html`](report/state.html) | **Where the project stands** — start here. What we can describe, what we cannot, and how each was verified. |
| [`report/`](report/findings.html) | **The findings report** — the longer narrative account with charts and photographic evidence. |
| [`plan/`](plan/) | Plan, guardrails, stage audit, W1 calibration, **evaluation design**, performance, pipeline steps, temporal alignment, research workplan, literature. What are we building, what is wrong with the current approach, what does prior art already solve, and what rules govern the work? | whoever directs it |
| [`runbooks/`](runbooks/) | How do I run this, and what should I expect to see? | whoever operates it |
| [`data/`](data/) | What are these instruments, what is the calibration history, what does the data actually contain? | whoever collects it |
| [`repro/`](repro/) | Which published numbers are reproducible, on what configuration, and which are not? | whoever checks them |

The [deprecation map](deprecation-map.md) lists every module with its status.

Inline documentation next to the code remains the authority on what each
function does. These documents record decisions, evidence and operating
procedure — they are not a prose copy of the implementation, and they should be
updated when a decision changes, not when a line of code changes.

**Handing this project to someone else?** Start with
[`briefing.md`](briefing.md): it is written to be read cold, with no prior context.
