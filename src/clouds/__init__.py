# PURPOSE: Mark src/clouds as the project's importable package and declare the
#          stable public surface other modules are expected to use.
# INPUTS: none.
# OUTPUTS: package metadata (`__version__`) and re-exports of the configuration
#          and provenance entry points.
# KEYWORDS: package, clouds, version, public_api
"""Cloud base height estimation from stereo fisheye imagery.

Subpackages are introduced in phase P2 of the reorganization; modules currently
live under the flat ``src/`` tree and are migrated here incrementally. The
configuration and provenance layers are the first to move, because every other
component depends on being able to state which parameters and which data a
result came from.
"""

__version__ = "0.1.0"

from .config import Config, ConfigError, load_config, describe  # noqa: F401
from .provenance import Run, start_run, file_sha256, git_info  # noqa: F401

__all__ = [
    "__version__",
    "Config",
    "ConfigError",
    "load_config",
    "describe",
    "Run",
    "start_run",
    "file_sha256",
    "git_info",
]
