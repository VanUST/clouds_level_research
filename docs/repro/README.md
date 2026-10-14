# Reproduction records

One file per published result set, stating what was checked, on which
configuration, and what held. These are records of measurement, not summaries of
the code: a claim is either reproduced with the command and output that show it,
or reported as not reproduced with the reason.

| Record | Result set |
|---|---|
| [dlcp2026.md](dlcp2026.md) | DLCP2026 submission (`papers/dlcp2026/37-Ustimov/tex/Ustimov.tex`) |

## How to add one

1. Write a config under `configs/` that pins every parameter to what the result
   claims, and record any place the written work and the code disagree in its
   header.
2. Run `tools/evaluate.py` (or `experiments/reproduce_paper.py` to get the claim
   table directly), keeping the run directory.
3. Record, for each claim: the claimed value, the measured value, the command,
   and the manifest path. Mark anything you could not measure as unverified
   rather than leaving it out.

A record whose numbers are not backed by a run manifest is a recollection.
