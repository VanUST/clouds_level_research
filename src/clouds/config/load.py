# PURPOSE: Load, validate and freeze a YAML configuration, and render a human
#          readable summary of the result.
# INPUTS: path to a YAML config file.
# OUTPUTS: a frozen Config whose sha256 covers the raw file bytes; raises
#          ConfigError on any schema violation.
# KEYWORDS: load, validate, freeze, config, summary
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from .calibration import load_calibration
from .schema import (
    AblationCfg, CameraCfg, ClusteringCfg, Config, ConfigError, DatasetCfg,
    MatchingCfg, MetricsCfg, OutputCfg, ProcessingCfg, ProjectCfg,
    TemporalCfg,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


# PURPOSE: Substitute ${var} references from the top-level `vars` mapping.
# INPUTS: node (any YAML value), variables (dict[str, str]).
# OUTPUTS: the same structure with string leaves expanded. Raises ConfigError on
#          an unknown variable name, so a typo cannot silently become a literal.
# KEYWORDS: substitution, variables, yaml
def _substitute(node: Any, variables: dict) -> Any:
    if isinstance(node, str):
        def repl(m):
            key = m.group(1)
            if key not in variables:
                raise ConfigError(f"unknown variable '${{{key}}}' referenced in config")
            return str(variables[key])
        return _VAR_RE.sub(repl, node)
    if isinstance(node, dict):
        return {k: _substitute(v, variables) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, variables) for v in node]
    return node


# PURPOSE: Require a section/key and coerce its type.
# INPUTS: section (str), data (dict), key (str), types (type or tuple of types).
# OUTPUTS: the value. Raises ConfigError naming the exact missing/wrong field, so
#          a config problem is diagnosable without reading the loader.
# KEYWORDS: validation, schema, required
def _req(section: str, data: dict, key: str, types):
    if key not in data:
        raise ConfigError(f"config section '{section}' is missing required key '{key}'")
    value = data[key]
    if not isinstance(value, types) or (isinstance(value, bool) and types is not bool):
        raise ConfigError(
            f"config '{section}.{key}' has type {type(value).__name__}, expected {types}"
        )
    return value


# PURPOSE: Load, validate and freeze a YAML configuration.
# INPUTS: path (str|Path) to a config file.
# OUTPUTS: Config whose `sha256` covers the raw file bytes and `resolved` holds
#          the substituted mapping. Raises ConfigError on any schema violation or
#          when the named calibration is not in configs/calib/.
# KEYWORDS: load, validate, freeze, config
def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")

    variables = _substitute(data.get("vars", {}), {})
    if not isinstance(variables, dict):
        raise ConfigError("top-level 'vars' must be a mapping")
    data = {k: _substitute(v, variables) for k, v in data.items() if k != "vars"}

    for section in ("project", "dataset", "camera", "matching", "clustering",
                    "temporal", "processing", "ablation",
                    "metrics", "output"):
        if section not in data:
            raise ConfigError(f"config {path} is missing section '{section}'")

    p = data["project"]; d = data["dataset"]; c = data["camera"]
    m = data["matching"]; cl = data["clustering"]
    tp = data["temporal"]; pr = data["processing"]; ab = data["ablation"]
    mt = data["metrics"]; o = data["output"]

    images_dir = _req("dataset", d, "images_dir", str)
    if not Path(images_dir).is_dir():
        raise ConfigError(f"dataset.images_dir does not exist: {images_dir}")

    model = _req("camera", c, "model", str)
    if model not in ("fisheye", "pinhole"):
        raise ConfigError(f"camera.model must be 'fisheye' or 'pinhole', got {model!r}")

    backend = _req("matching", m, "backend", str)
    # Validate against the matcher registry rather than a hard-coded list, so a
    # newly registered backend is selectable through the config with no change
    # here. The import is deferred and tolerated: configuration loading must not
    # fail merely because the (torch-dependent) matching package cannot be
    # imported in some minimal environment, and in that case the check is skipped
    # and the factory raises its own error later.
    try:
        from ..matching.matchers import get_available_backends
        known = set(get_available_backends())
    except Exception:
        known = None
    if known is not None and backend not in known:
        raise ConfigError(f"matching.backend unknown: {backend!r}; available: {sorted(known)}")

    strategy = _req("temporal", tp, "cbh_strategy", str)
    if strategy not in ("largest", "lowest", "lowest_reliable", "lowest_mode"):
        raise ConfigError(f"temporal.cbh_strategy unknown: {strategy!r}")

    mode = _req("temporal", tp, "mode", str)
    if mode not in ("simple", "multi_track"):
        raise ConfigError(f"temporal.mode unknown: {mode!r}")

    calib_id = _req("camera", c, "calibration", str)
    # The calibration registry lives next to the config by default. A config kept
    # elsewhere can name it explicitly with a top-level `calibration_dir`, so a
    # scratch config does not silently resolve an empty registry.
    raw_calib_dir = data.get("calibration_dir")
    if raw_calib_dir:
        calib_dir = Path(raw_calib_dir)
        if not calib_dir.is_absolute():
            # A relative registry path is relative to the config file, never to
            # the process working directory.
            calib_dir = path.parent / calib_dir
    else:
        calib_dir = path.parent / "calib"
    calibration = load_calibration(calib_dir, calib_id)
    date = _req("dataset", d, "date", str)
    applies = calibration.applies_to(date)
    if not applies["applies"]:
        raise ConfigError(
            f"calibration '{calib_id}' is not documented as applying to session "
            f"{date} (applies_to_sessions={calibration.applies_to_sessions}). "
            f"Add the session to the registry entry if that is intentional."
        )

    cfg = Config(
        project=ProjectCfg(name=_req("project", p, "name", str),
                           purpose=str(p.get("purpose", ""))),
        dataset=DatasetCfg(
            images_dir=images_dir,
            date=date,
            window_start=_req("dataset", d, "window_start", str),
            window_end=_req("dataset", d, "window_end", str),
            image_clock=str(d.get("image_clock", "unknown")),
            ceilometer_clock=str(d.get("ceilometer_clock", "unknown")),
            clock_offset_h=float(_req("dataset", d, "clock_offset_h", (int, float))),
            ceilometer_log=_req("dataset", d, "ceilometer_log", str),
            pair_tolerance_s=float(d.get("pair_tolerance_s", 2.0)),
            manual_points=str(d.get("manual_points", "")),
        ),
        camera=CameraCfg(
            model=model,
            stereo_base_m=float(_req("camera", c, "stereo_base_m", (int, float))),
            angle_of_view_deg=float(_req("camera", c, "angle_of_view_deg", (int, float))),
            image_width_px=int(_req("camera", c, "image_width_px", int)),
            original_size=tuple(_req("camera", c, "original_size", list)),
            target_size=tuple(_req("camera", c, "target_size", list)),
            calibration=calib_id,
            roll_correction_deg=float(c.get("roll_correction_deg", 0.0)),
            distortion_a=float(c.get("distortion_a", 0.0)),
            exact_triangulation=bool(c.get("exact_triangulation", False)),
            f_px_per_rad=float(c.get("f_px_per_rad", 0.0)),
            field_radius_px=float(c.get("field_radius_px", 0.0)),
            mask_foreground=bool(c.get("mask_foreground", False)),
            foreground_radius_px=float(c.get("foreground_radius_px", 0.0)),
        ),
        matching=MatchingCfg(backend=backend,
                             match_threshold=float(m.get("match_threshold", 0.5)),
                             filter=m.get("filter"),
                             batch_size=int(m.get("batch_size", 1)),
                             prefetch=int(m.get("prefetch", 1)),
                             overlap_matching=bool(m.get("overlap_matching", False)),
                             params=m.get("params")),
        clustering=ClusteringCfg(
            min_cluster_size=int(cl.get("min_cluster_size", 3)),
            min_samples=int(cl.get("min_samples", 3)),
            allow_single_cluster=bool(cl.get("allow_single_cluster", True)),
        ),
        temporal=TemporalCfg(
            mode=mode,
            cbh_strategy=strategy,
            sigma_h_accel=float(tp.get("sigma_h_accel", 5.0)),
            measurement_noise_scale=float(tp.get("measurement_noise_scale", 50.0)),
            cluster_merge_height_tol_m=float(tp.get("cluster_merge_height_tol_m", 0.0)),
            min_cluster_size_for_tracking=int(tp.get("min_cluster_size_for_tracking", 8)),
            filter_enabled=bool(tp.get("filter_enabled", True)),
        ),
        processing=ProcessingCfg(
            task=str(pr.get("task", "evaluate")),
            stride=int(pr.get("stride", 1)),
            visualize=bool(pr.get("visualize", False)),
            log_results=bool(pr.get("log_results", True)),
        ),
        ablation=AblationCfg(
            max_noise_theta=float(ab.get("max_noise_theta", 15.0)),
            steps=int(ab.get("steps", 10)),
        ),
        metrics=MetricsCfg(
            window_min=int(mt.get("window_min", 30)),
            hop_min=int(mt.get("hop_min", 15)),
            shift_search_s=int(mt.get("shift_search_s", 300)),
            tolerance_s=float(mt.get("tolerance_s", 10.0)),
        ),
        output=OutputCfg(
            logs_dir=str(_req("output", o, "logs_dir", str)),
            assets_dir=str(o.get("assets_dir", "")),
        ),
        calibration_dir=str(calib_dir.resolve()),
        source_path=str(path.resolve()),
        sha256=sha,
        resolved=data,
        calibration_record=calibration,
    )
    return cfg


# PURPOSE: Produce a compact, human-readable summary of the resolved config for
#          console output and log headers.
# INPUTS: cfg (Config).
# OUTPUTS: multi-line string.
# KEYWORDS: summary, report, config
def describe(cfg: Config) -> str:
    cal = cfg.calibration_record
    lines = [
        f"config          : {cfg.source_path}",
        f"config sha256   : {cfg.sha256}",
        f"project         : {cfg.project.name} ({cfg.project.purpose})",
        f"dataset         : {cfg.dataset.date} {cfg.dataset.window_start}-{cfg.dataset.window_end} "
        f"[{cfg.dataset.image_clock} filenames, {cfg.dataset.ceilometer_clock} ceilometer, "
        f"+{cfg.dataset.clock_offset_h}h]",
        f"images_dir      : {cfg.dataset.images_dir}",
        f"ceilometer_log  : {cfg.dataset.ceilometer_log}",
        f"camera          : {cfg.camera.model} B={cfg.camera.stereo_base_m}m "
        f"fov={cfg.camera.angle_of_view_deg}deg crop={cfg.camera.target_size}",
        f"calibration     : {cal.id} status={cal.status} "
        f"done={cal.calibrated_on} ({cal.calibrated_on_source}, verified={cal.calibrated_on_verified})",
        f"matching        : {cfg.matching.backend} thr={cfg.matching.match_threshold}",
        f"temporal        : mode={cfg.temporal.mode} strategy={cfg.temporal.cbh_strategy} "
        f"R_scale={cfg.temporal.measurement_noise_scale} sigma_h={cfg.temporal.sigma_h_accel}",
        f"metrics         : window={cfg.metrics.window_min}min hop={cfg.metrics.hop_min}min "
        f"shift=+-{cfg.metrics.shift_search_s}s",
    ]
    return "\n".join(lines)
