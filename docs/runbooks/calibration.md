# Runbook — calibration

A 1-pixel disparity error moves the height estimate by roughly 30 m at cloud
heights of 1–2 km, and the written work reports that repeated calibrations on
different days produced disparity shifts of up to 30 px for the same object
(≈900 m). Calibration identity is therefore a first-class input, not a constant.

## Where it lives

Each matrix is a JSON entry under `configs/calib/`. The registry entry is the
only place a matrix is defined; nothing else may hard-code one.

```json
{
  "id": "2025-05-22_affine_old",
  "matrix_2x3": [[...], [...]],
  "calibrated_on": "2025-05-22",
  "calibrated_on_source": "inferred_from_artifact_filename",
  "calibrated_on_verified": false,
  "method": "sun position; cv2.estimateAffine2D (RANSAC initialisation)",
  "applies_to_sessions": ["2025-05-22", "2025-05-23"],
  "applies_to_sessions_verified": false,
  "notes": ["..."]
}
```

A config names one entry (`camera.calibration`). The loader refuses to start if
that entry is not documented as applying to the configured session, so a
cross-session calibration cannot be applied by accident.

## Adding a calibration

1. Produce the matrix with `cv2.estimateAffine2D` over sun-position pairs from one
   session's stereo pair (the Sun is the only object at infinity visible to both
   zenith-pointing cameras).
2. Add `configs/calib/<session>_<label>.json`. Fill in `calibrated_on` **and**
   `calibrated_on_source`. Set `calibrated_on_verified: true` only when a
   calibration log or the operator confirms the date — an inferred date stays
   `false`.
3. Record which sessions it may be applied to, and whether that is verified.
4. Reference it from a config and confirm with
   `python tools/run_info.py --config <config>`.

## Comparing calibrations

```bash
python tools/compare_calibrations.py --config configs/paper_dlcp2026.yaml \
    --calibrations 2025-05-22_affine_old 2025-05-22_affine_new
```

Each variant gets its own run directory and manifest; the summary is written to
`<logs>/calibration_comparison.json`.

## Known gaps

- **There is no calibration for the 2025-05-23 validation session.** The session
  is processed with the 2025-05-22 matrix, which is a documented, unquantified
  error source.
- `2025-05-22_affine_new` has lost its provenance: no date, no source images and
  only four significant figures.
- `2023-05-29_rig_2023` appears in the written work with a *near-identity*
  rotation (2.4°). It belongs to a different rig and must not be applied to the
  2025 data, whose matrices rotate by ≈90°. It is registered as
  `historical_not_applicable` so the two cannot be confused.
