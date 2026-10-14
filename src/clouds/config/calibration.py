# PURPOSE: The calibration registry — one record per affine matrix, carrying the
#          provenance needed to answer "which calibration, done when, from what,
#          and does it actually apply to this session" rather than only its values.
# INPUTS: parsed JSON of a file under configs/calib/.
# OUTPUTS: Calibration records looked up by id.
# KEYWORDS: calibration, affine, provenance, registry
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .schema import ConfigError


@dataclass(frozen=True)
class Calibration:
    id: str
    matrix_2x3: tuple
    status: str
    calibrated_on: Optional[str]
    calibrated_on_source: str
    calibrated_on_verified: bool
    method: str
    applies_to_sessions: tuple
    applies_to_sessions_verified: bool
    source_artifact: str
    source_artifact_sha256: Optional[str]
    notes: tuple
    raw: dict = field(repr=False, default_factory=dict)

    # PURPOSE: Return the 3x3 homogeneous form of the affine transform.
    # INPUTS: none.
    # OUTPUTS: tuple of three 3-element rows; the bottom row is always [0,0,1].
    # KEYWORDS: calibration, matrix, homogeneous
    def matrix_3x3(self) -> tuple:
        (a, b, tx), (c, d, ty) = self.matrix_2x3
        return ((a, b, tx), (c, d, ty), (0.0, 0.0, 1.0))

    # PURPOSE: Report whether this calibration is documented as applying to a
    #          given acquisition date, and how strongly that is known.
    # INPUTS: session (str, "YYYY-MM-DD").
    # OUTPUTS: dict {applies: bool, verified: bool} — applies is False when the
    #          date is absent from applies_to_sessions.
    # KEYWORDS: calibration, applicability, session
    def applies_to(self, session: str) -> dict:
        return {
            "applies": session in self.applies_to_sessions,
            "verified": bool(self.applies_to_sessions_verified),
        }


# PURPOSE: Read a calibration registry entry by id.
# INPUTS: calib_dir (Path) — the configs/calib directory; calib_id (str) — the
#         entry id, which is also its filename stem.
# OUTPUTS: Calibration. Raises ConfigError when the entry or the matrix is absent.
# KEYWORDS: calibration, registry, load
def load_calibration(calib_dir: Path, calib_id: str) -> Calibration:
    path = Path(calib_dir) / f"{calib_id}.json"
    if not path.is_file():
        raise ConfigError(
            f"calibration '{calib_id}' not found at {path}. "
            f"Known ids: {sorted(p.stem for p in Path(calib_dir).glob('*.json'))}"
        )
    with path.open() as fh:
        data = json.load(fh)
    matrix = data.get("matrix_2x3")
    if not matrix or len(matrix) != 2 or any(len(r) != 3 for r in matrix):
        raise ConfigError(f"calibration '{calib_id}': matrix_2x3 must be a 2x3 array")
    return Calibration(
        id=data.get("id", calib_id),
        matrix_2x3=tuple(tuple(float(v) for v in row) for row in matrix),
        status=data.get("status", "unknown"),
        calibrated_on=data.get("calibrated_on"),
        calibrated_on_source=data.get("calibrated_on_source", "unknown"),
        calibrated_on_verified=bool(data.get("calibrated_on_verified", False)),
        method=data.get("method", "unknown"),
        applies_to_sessions=tuple(data.get("applies_to_sessions", [])),
        applies_to_sessions_verified=bool(data.get("applies_to_sessions_verified", False)),
        source_artifact=data.get("source_artifact", "unknown"),
        source_artifact_sha256=data.get("source_artifact_sha256"),
        notes=tuple(data.get("notes", [])),
        raw=data,
    )
