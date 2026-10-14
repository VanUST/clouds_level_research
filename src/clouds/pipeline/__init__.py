"""PURPOSE: Pipeline — per-pair stereo analysis, the honest evaluation path and
the task-level entry points that operate on a whole acquisition.

KEYWORDS: package, pipeline, analysis, tasks, evaluation
"""
from .analyze import StereoAnalyzer  # noqa: F401
from .evaluate import evaluate, write_artefacts  # noqa: F401
