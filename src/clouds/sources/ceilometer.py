# PURPOSE: Read DOL-2 laser-ceilometer log files into a validated, time-stamped
#          series of cloud-base heights. The instrument writes one line per
#          acquisition slot (15 s) and reports up to two cloud layers plus a
#          quality code; 9999 means "no detection".
#          Two on-disk dialects exist and BOTH are handled here:
#            (a) raw serial repr  : 250122000007 MSK H1 = b'1510' H2 = b'1940' L = b'9999' Q = b'99'
#            (b) exported plain   : 250122000007 MSK H1 = 1510 H2 = 1940 L = 9999 Q = 99
#          The shipped parser understood only (a), which silently returned zero
#          measurements for every file in dialect (b).
# INPUTS: path to a .txt log.
# OUTPUTS: list[CeilometerSample], sorted by time. Raises CeilometerFormatError
#          when the file exists but no line matches either dialect, so a format
#          change cannot masquerade as "clear sky all day".
# KEYWORDS: ceilometer, DOL-2, parsing, cloud_base_height, ground_truth, quality
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Sentinel written by the instrument when no layer was detected at that slot.
NO_DETECTION = 9999

# Timestamp (12 digits, YYMMDDHHMMSS) then the MSK label, then the field list.
# Each value may be wrapped in a Python bytes repr (b'NNNN') or bare.
_LINE_RE = re.compile(
    r"^(?P<ts>\d{12})\s+MSK\s+"
    r"H1\s*=\s*(?:b')?(?P<h1>\d+)(?:')?\s+"
    r"H2\s*=\s*(?:b')?(?P<h2>\d+)(?:')?\s+"
    r"L\s*=\s*(?:b')?(?P<l>\d+)(?:')?\s+"
    r"Q\s*=\s*(?:b')?(?P<q>\d+)(?:')?"
)


class CeilometerFormatError(Exception):
    """Raised when a ceilometer log matches neither known dialect."""


# PURPOSE: Represent one acquisition slot of the DOL-2 ceilometer.
# INPUTS: parsed timestamp and the four raw fields.
# OUTPUTS: immutable sample; `lowest` is the lowest detected layer (None when no
#          layer was detected), which is what the instrument reports as cloud base.
# KEYWORDS: ceilometer, sample, layers, quality
@dataclass(frozen=True)
class CeilometerSample:
    time: datetime
    h1: Optional[float]
    h2: Optional[float]
    backscatter_l: Optional[float]
    quality: Optional[int]

    # PURPOSE: Return the lowest detected cloud layer, the quantity comparable to
    #          a stereo cloud-base estimate.
    # INPUTS: none.
    # OUTPUTS: float height in metres, or None when neither layer was detected.
    # KEYWORDS: cloud_base, lowest_layer
    @property
    def lowest(self) -> Optional[float]:
        layers = [h for h in (self.h1, self.h2) if h is not None]
        return min(layers) if layers else None


# PURPOSE: Parse a DOL-2 log file, accepting both known on-disk dialects.
# INPUTS: path (str|Path), quality_min (int) — samples whose quality code is
#         below this are dropped; pass 0 to keep everything that has a layer.
# OUTPUTS: list[CeilometerSample] sorted by time, containing only slots with at
#          least one detected layer and sufficient quality. Returns [] for a
#          missing file (with no exception) so an optional input does not abort a
#          run; raises CeilometerFormatError if the file exists but is unparsable.
# KEYWORDS: parse, DOL-2, dialect, quality_gate
def parse_ceilometer_log(path: str | Path, quality_min: int = 1) -> list[CeilometerSample]:
    p = Path(path)
    if not p.is_file():
        return []

    samples: list[CeilometerSample] = []
    total_lines = 0
    matched = 0
    for line in p.read_text(errors="replace").splitlines():
        total_lines += 1
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        matched += 1
        try:
            ts = datetime.strptime(m.group("ts"), "%y%m%d%H%M%S")
        except ValueError:
            continue
        h1 = int(m.group("h1"))
        h2 = int(m.group("h2"))
        l = int(m.group("l"))
        q = int(m.group("q"))
        if q < quality_min:
            continue
        sample = CeilometerSample(
            time=ts,
            h1=None if h1 == NO_DETECTION else float(h1),
            h2=None if h2 == NO_DETECTION else float(h2),
            backscatter_l=None if l == NO_DETECTION else float(l),
            quality=q,
        )
        if sample.lowest is not None:
            samples.append(sample)

    if total_lines and matched == 0:
        raise CeilometerFormatError(
            f"{p}: {total_lines} lines but none matched a known DOL-2 dialect. "
            "The instrument export format may have changed."
        )
    samples.sort(key=lambda s: s.time)
    return samples


# PURPOSE: Flatten a parsed log into the (time, height) series the comparison
#          metrics consume.
# INPUTS: samples (list[CeilometerSample]) or a path (str|Path); offset_h (float)
#         — hours to add when the consumer works in another clock.
# OUTPUTS: list of (datetime, float) tuples sorted by time.
# KEYWORDS: series, adapter, time_base
def load_series(source, offset_h: float = 0.0) -> list:
    from datetime import timedelta
    samples = parse_ceilometer_log(source) if isinstance(source, (str, Path)) else source
    delta = timedelta(hours=offset_h)
    return [(s.time + delta, s.lowest) for s in samples]
