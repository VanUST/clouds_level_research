# PURPOSE: Give every run a self-describing record. A run creates a dedicated
#          output directory containing run_manifest.json (what code, what config,
#          what calibration, what dataset, what counts, what results) and
#          run.log (structured JSON-lines events). This is the artefact that
#          makes a result reproducible later without reverse-engineering the
#          code that produced it.
# INPUTS: an entry-point name, argv, a loaded clouds.config.Config, and the
#         identity of the data actually read (paths plus content hashes).
# OUTPUTS: a Run object; on disk, <run_dir>/run_manifest.json and run.log.
# KEYWORDS: provenance, reproducibility, manifest, logging, jsonl, hashing
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Packages recorded in every manifest. Missing ones are recorded as null rather
# than omitted, so an absent dependency is visible instead of implied.
_TRACKED_PACKAGES = (
    "numpy", "scipy", "torch", "torchvision", "kornia", "opencv-contrib-python",
    "scikit-learn", "pandas", "matplotlib", "pillow", "dtw-python", "PyYAML",
)


# PURPOSE: Compute the sha256 of a file's bytes, for content-addressing the data
#          a run consumed.
# INPUTS: path (str|Path).
# OUTPUTS: lowercase hex digest, or None when the path is absent or unreadable
#          (e.g. an optional log). Reads in chunks so a large file does not have
#          to fit in memory.
# KEYWORDS: hashing, sha256, data_identity
def file_sha256(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# PURPOSE: Describe the git state of the code that is about to run.
# INPUTS: repo_root (str|Path).
# OUTPUTS: dict {commit, branch, dirty, describe}. All values are None when the
#          directory is not a git repository, so a non-git run is still recorded.
# KEYWORDS: git, revision, dirty, provenance
def git_info(repo_root: str | Path) -> dict:
    root = Path(repo_root)

    def _git(*args):
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=10, check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else len(status.splitlines()),
        "describe": _git("describe", "--always", "--dirty"),
    }


# PURPOSE: Collect installed versions of the packages a run depends on.
# INPUTS: none.
# OUTPUTS: dict mapping package name to version string or None if not installed.
# KEYWORDS: environment, versions, dependencies
def package_versions() -> dict:
    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


# PURPOSE: Build a filesystem-safe, sortable, unique run directory name.
# INPUTS: base (str|Path) — parent directory; label (str) — short description.
# OUTPUTS: Path of the created directory, named
#          <base>/<UTC timestamp>_<sanitised label>_<6 hex>.
# KEYWORDS: run_dir, naming, timestamp
def new_run_dir(base: str | Path, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
    path = Path(base) / f"{stamp}_{safe}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


# PURPOSE: Represent one execution: its manifest, its JSONL log and its output
#          directory. Fields may be added as the run progresses (counts, timings,
#          results) and are flushed to run_manifest.json on close, so a crashed
#          run still leaves whatever the log recorded up to that point.
# INPUTS: run_dir (Path), entry_point (str), argv (list[str]), context (dict).
# OUTPUTS: Run instance; finalises into run_manifest.json plus run.log.
# KEYWORDS: run, manifest, journal, lifecycle
@dataclass
class Run:
    run_dir: Path
    entry_point: str
    argv: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    _log_fh: Any = field(default=None, repr=False)

    # PURPOSE: Initialise the manifest skeleton and open the JSONL log.
    # INPUTS: repo_root (str|Path), config (clouds.config.Config or None).
    # OUTPUTS: None (mutates self.manifest, opens self._log_fh).
    # KEYWORDS: manifest, initialise, logging
    def open(self, repo_root: str | Path, config=None) -> "Run":
        self.manifest.update({
            "run_id": self.run_dir.name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "entry_point": self.entry_point,
            "argv": list(self.argv),
            "run_dir": str(self.run_dir),
            "git": git_info(repo_root),
            "python": {
                "version": sys.version.split()[0],
                "executable": sys.executable,
                "platform": platform.platform(),
                "cwd": os.getcwd(),
            },
            "packages": package_versions(),
            "counts": {},
            "timings": {},
            "results": {},
        })
        if config is not None:
            cal = config.calibration_record
            self.manifest["config"] = {
                "path": config.source_path,
                "sha256": config.sha256,
                "resolved": config.resolved,
            }
            self.manifest["calibration"] = {
                "id": cal.id,
                "status": cal.status,
                "matrix_2x3": [list(r) for r in cal.matrix_2x3],
                "matrix_3x3": [list(r) for r in cal.matrix_3x3()],
                "calibrated_on": cal.calibrated_on,
                "calibrated_on_source": cal.calibrated_on_source,
                "calibrated_on_verified": cal.calibrated_on_verified,
                "method": cal.method,
                "applies_to_sessions": list(cal.applies_to_sessions),
                "applies_to_sessions_verified": cal.applies_to_sessions_verified,
                "source_artifact": cal.source_artifact,
                "source_artifact_sha256": cal.source_artifact_sha256,
                "notes": list(cal.notes),
            }
            self.manifest["dataset"] = {
                "date": config.dataset.date,
                "images_dir": config.dataset.images_dir,
                "window_start": config.dataset.window_start,
                "window_end": config.dataset.window_end,
                "image_clock": config.dataset.image_clock,
                "ceilometer_clock": config.dataset.ceilometer_clock,
                "clock_offset_h": config.dataset.clock_offset_h,
                "ceilometer_log": config.dataset.ceilometer_log,
                "ceilometer_log_sha256": file_sha256(config.dataset.ceilometer_log),
                "pair_tolerance_s": config.dataset.pair_tolerance_s,
            }
        self._log_fh = (self.run_dir / "run.log").open("a")
        self.event("run_opened", entry_point=self.entry_point, run_id=self.manifest["run_id"])
        return self

    # PURPOSE: Append one structured event to the JSONL log.
    # INPUTS: event (str), **fields (JSON-serialisable; non-serialisable values
    #         are stringified).
    # OUTPUTS: None (side effect: one line appended and flushed, so a crash does
    #          not lose the events that preceded it).
    # KEYWORDS: logging, jsonl, event
    def event(self, event: str, **fields) -> None:
        record = {"ts_utc": datetime.now(timezone.utc).isoformat(), "event": event}
        record.update(fields)
        if self._log_fh is not None:
            self._log_fh.write(json.dumps(record, default=str) + "\n")
            self._log_fh.flush()

    # PURPOSE: Merge values into a manifest sub-section.
    # INPUTS: section (str, e.g. "counts"), **fields.
    # OUTPUTS: None (mutates the manifest).
    # KEYWORDS: manifest, update
    def record(self, section: str, **fields) -> None:
        self.manifest.setdefault(section, {}).update(fields)

    # PURPOSE: Store provenance for one input file (path plus content hash).
    # INPUTS: key (str), path (str|Path).
    # OUTPUTS: None (mutates manifest["inputs"]). Missing files are recorded with
    #          a null hash and exists=False rather than raising.
    # KEYWORDS: inputs, hashing, provenance
    def add_input(self, key: str, path: str | Path) -> None:
        p = Path(path)
        self.manifest.setdefault("inputs", {})[key] = {
            "path": str(p),
            "exists": p.is_file(),
            "sha256": file_sha256(p) if p.is_file() else None,
            "size_bytes": p.stat().st_size if p.is_file() else None,
        }

    # PURPOSE: Close the run: stamp the end time and write run_manifest.json.
    # INPUTS: status (str, default "ok"); error (str|None).
    # OUTPUTS: Path of the written manifest.
    # KEYWORDS: finalise, manifest, write
    def close(self, status: str = "ok", error: Optional[str] = None) -> Path:
        self.manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["status"] = status
        if error:
            self.manifest["error"] = error
        self.event("run_closed", status=status, error=error)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        out = self.run_dir / "run_manifest.json"
        out.write_text(json.dumps(self.manifest, indent=2, default=str))
        return out


# PURPOSE: Convenience factory that creates a run directory and opens a Run.
# INPUTS: base (str|Path), label (str), entry_point (str), argv (list[str]),
#         repo_root (str|Path), config (Config|None).
# OUTPUTS: an opened Run.
# KEYWORDS: factory, run, provenance
def start_run(base: str | Path, label: str, entry_point: str, argv,
              repo_root: str | Path, config=None) -> Run:
    run = Run(run_dir=new_run_dir(base, label), entry_point=entry_point, argv=list(argv))
    return run.open(repo_root=repo_root, config=config)
