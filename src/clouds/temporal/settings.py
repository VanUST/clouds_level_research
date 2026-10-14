# PURPOSE: Translate the validated, frozen project configuration into the plain
#          settings dictionary that TemporalFilter consumes. This is the single
#          place where the configured cloud-base-height strategy, process-noise
#          and measurement-noise scales reach
#          the tracker, so the tracker can never fall back to an undocumented
#          default and no experiment script needs to hard-code them.
# INPUTS: cfg (clouds.config.Config).
# OUTPUTS: a plain dict of scalars, plus a list of {key, code_default, configured}
#          records naming every value that differs from the historical library
#          default, so a divergence is visible in the run log rather than silent.
# KEYWORDS: temporal, kalman, settings, defaults, provenance
from __future__ import annotations

# Historical library defaults, kept only so deviations can be reported.
_LIBRARY_DEFAULTS = {
    "mode": "multi_track",
    "cbh_strategy": "largest",
    "measurement_noise_scale": 10.0,
    "sigma_h_accel_simple": 3.0,
    "cluster_merge_height_tol_m": 0.0,
    "min_cluster_size_for_tracking": 8,
}


# PURPOSE: Build the TemporalFilter settings from the project configuration.
# INPUTS: cfg (clouds.config.Config).
# OUTPUTS: (settings dict, deviations list). The settings dict is JSON-safe.
# KEYWORDS: temporal, kalman, settings, ablation
def build_temporal_settings(cfg) -> tuple:
    t = cfg.temporal

    settings = {
        "mode": t.mode,
        "cbh_strategy": t.cbh_strategy,
        "process_noise": {"sigma_h_accel": float(t.sigma_h_accel)},
        "measurement_noise_scale": float(t.measurement_noise_scale),
        "min_cluster_size_for_tracking": int(t.min_cluster_size_for_tracking),
        "cluster_merge_height_tol_m": float(t.cluster_merge_height_tol_m),
        "image_width": int(cfg.camera.image_width_px),
    }

    deviations = []
    for key, default in _LIBRARY_DEFAULTS.items():
        if key == "sigma_h_accel_simple":
            configured = float(t.sigma_h_accel)
            label = "process_noise.sigma_h_accel (simple mode)"

        else:
            configured = settings.get(key)
            label = key
        if configured is not None and configured != default:
            deviations.append({
                "key": label,
                "code_default": default,
                "configured": configured,
            })

    return settings, deviations
