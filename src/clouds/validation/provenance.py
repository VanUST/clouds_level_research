# PURPOSE: Guards that assert WHERE a value came from, rather than what it is worth.
#
#          Every one of the six harness defects found in this project shared a single
#          mechanism: a quantity was read from somewhere other than where it was claimed to
#          come from. A validation metric that maximised over a time shift was scored
#          against a null that did not. A synthetic ground truth was verified in the native
#          frame while the matcher consumed the warped one. Two scripts rendered "the same"
#          cloud 1.5 degrees apart because one passed a parameter and the other took a
#          default. A script asserted partway through three chained replacements and never
#          reached its write, while the commit message described edits that were not there.
#          A cross-validation drew random folds from a series whose lag-1 autocorrelation is
#          0.93, so a held-out frame had its own near-duplicates in training. A binned
#          reference was taken from the single sample nearest the bin's edge instead of the
#          median over the bin, which manufactured a dynamic signal that did not exist.
#
#          None of these raised an exception. Two of them made the results look MORE
#          significant than the truth. That is the class of defect a provenance assertion
#          exists to catch: not "is this number right" - which no guard can know - but "did
#          this number come from where it is claimed to".
#
#          Each guard raises ProvenanceError with the quantities that disagree, so a
#          failure names the mismatch rather than only signalling a bad state.
# INPUTS: as documented per function.
# OUTPUTS: the inputs are returned unchanged on success; ProvenanceError otherwise.
# KEYWORDS: provenance, guardrail, cross_validation, embargo, binning, leakage
from __future__ import annotations

import numpy as np


class ProvenanceError(AssertionError):
    """Raised when a value does not come from where it is claimed to come from."""


# PURPOSE: Assert that a binned reference value is an aggregate of the samples INSIDE its
#          bin, not of samples near it.
# INPUTS:
#   t_bin   (tuple) the half-open interval [t0, t1) of the bin, as comparable scalars
#           (seconds, or any monotone numeric time).
#   t_samples (array) the times of every reference sample available.
#   v_samples (array) the values of those samples.
#   v_used  (float) the value actually used for the bin.
#   agg     (callable) the aggregate that was CLAIMED to be used; default the median.
#   min_n   (int) the smallest number of in-bin samples that may support a value.
# OUTPUTS: v_used on success. Raises ProvenanceError when the value is not the claimed
#          aggregate of the in-bin samples, or when the bin holds too few samples to
#          support any value at all.
# KEYWORDS: binning, reference, provenance, alignment
def assert_bin_reference(t_bin, t_samples, v_samples, v_used, agg=np.median, min_n=3):
    t0, t1 = t_bin
    t_samples = np.asarray(t_samples, float)
    v_samples = np.asarray(v_samples, float)
    inside = (t_samples >= t0) & (t_samples < t1)
    n_in = int(inside.sum())
    if n_in < min_n:
        raise ProvenanceError(
            f"bin [{t0}, {t1}) holds {n_in} reference samples, fewer than the {min_n} "
            f"required to support a value; it must be dropped, not filled")
    claimed = float(agg(v_samples[inside]))
    if not np.isclose(float(v_used), claimed, rtol=1e-9, atol=1e-9):
        # a value taken from the nearest sample to an Edge produces exactly this failure:
        # it is a real sample, just not one inside the bin
        nearest_i = int(np.argmin(np.abs(t_samples - t0)))
        raise ProvenanceError(
            f"bin reference {v_used!r} is not the {agg.__name__} of the {n_in} samples "
            f"inside [{t0}, {t1}), which is {claimed!r}; the nearest sample to the bin "
            f"START is {v_samples[nearest_i]!r} at t={t_samples[nearest_i]!r}")
    return v_used


# PURPOSE: Assert that no training index lies within `embargo` seconds of a held-out block.
# INPUTS:
#   train_i, test_i (arrays) index arrays into t_frames.
#   t_frames (array) the time of every frame, as comparable scalars.
#   embargo_s (float) the minimum separation required.
#   series_t (array) optional; the times of the reference series, so the lag-1
#       autocorrelation can be reported with the failure.
# OUTPUTS: None on success. Raises ProvenanceError naming the violating pair, on the
#          commonest failure - a training frame adjacent in time to a held-out one, which on
#          an autocorrelated series makes the "held-out" observation a near-duplicate of
#          something the model was fitted on.
# KEYWORDS: cross_validation, embargo, leakage, provenance
def assert_disjoint_with_embargo(train_i, test_i, t_frames, embargo_s, series_t=None):
    train_i = np.asarray(train_i); test_i = np.asarray(test_i)
    t_frames = np.asarray(t_frames, float)
    shared = np.intersect1d(train_i, test_i)
    if shared.size:
        raise ProvenanceError(
            f"{shared.size} indices appear in BOTH train and test, e.g. {shared[:5].tolist()}")
    if not len(train_i) or not len(test_i):
        return
    # EVERY held-out frame must be checked, not only the block's start: a training frame
    # can sit just outside the block's leading edge and still be a near-duplicate of a
    # frame inside it. Both index sets are sorted in time and resolved with a binary
    # search, so the check is O((n_train + n_test) log n_train) rather than a full
    # outer product.
    tr = train_i[np.argsort(t_frames[train_i])]
    te = test_i[np.argsort(t_frames[test_i])]
    t_tr = t_frames[tr]
    pos = np.searchsorted(t_tr, t_frames[te])
    worst_d, worst_pair = np.inf, None
    for k, p_ in enumerate(pos):
        for j in (p_ - 1, p_):
            if 0 <= j < len(tr):
                d = abs(float(t_frames[te[k]] - t_tr[j]))
                if d < worst_d:
                    worst_d, worst_pair = d, (int(tr[j]), int(te[k]))
    if worst_d <= embargo_s:
        j, k = worst_pair
        raise ProvenanceError(
            f"training index {j} at t={t_frames[j]!r} lies {worst_d:.1f} s from held-out "
            f"index {k} at t={t_frames[k]!r}, within the {embargo_s:.0f} s embargo; on an "
            f"autocorrelated series that frame is a near-duplicate of a held-out "
            f"observation, so the fold is not out of sample")


# PURPOSE: Report the lag-1 autocorrelation and the implied decorrelation time of a series,
#          so a cross-validation design can be checked against it before being trusted.
# INPUTS: x (array) the series; dt_s (float) the spacing between samples.
# OUTPUTS: dict with lag1, decorrelation_s and n_eff - the effective sample size, which is
#          what any significance claim must be based on rather than the nominal length.
# KEYWORDS: autocorrelation, n_eff, design
def autocorrelation_report(x, dt_s=1.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return {"lag1": float("nan"), "decorrelation_s": float("nan"), "n_eff": 1.0}
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1 = max(min(r1, 0.999), -0.999)
    n_eff = max(1.0, len(x) * (1 - r1) / (1 + r1))
    return {"lag1": r1, "decorrelation_s": dt_s * len(x) / n_eff, "n_eff": n_eff}
