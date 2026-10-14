"""PURPOSE: Configuration as versioned data — schema, calibration registry and loader.

KEYWORDS: package, config, calibration
"""
from .calibration import Calibration, load_calibration
from .load import describe, load_config
from .schema import Config, ConfigError

__all__ = ["Calibration", "load_calibration", "Config", "ConfigError",
           "load_config", "describe"]
