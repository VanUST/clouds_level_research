# PURPOSE: Locate and pair the two camera streams on disk. Two modes are needed:
#          an unconstrained sweep over a directory, and a calendar-day window in
#          which each devID1 frame is matched to the temporally nearest devID2
#          frame. The cameras do not stamp identical seconds, so exact-name
#          pairing silently discards roughly a tenth of the data.
# INPUTS: a directory; optionally a wall-clock window and a calendar date.
# OUTPUTS: sorted list of (devID1_path, devID2_path) tuples.
# KEYWORDS: pairing, stereo, images, nearest_timestamp, time_window
from __future__ import annotations

import bisect
import datetime
import os
import re

_MODERN_PATTERN = re.compile(r"img-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})devID([12])")


# PURPOSE: Find and pair images whose names end in _1 / _2 in a directory.
# INPUTS: directory (str).
# OUTPUTS: sorted list of (path1, path2) tuples; incomplete pairs are dropped.
# KEYWORDS: pairing, legacy, directory
def find_image_pairs(directory):
    """Pair images whose names end in _1 / _2 within one directory."""
    image_pairs = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg")):
            base_name, ext = os.path.splitext(filename)
            if base_name.endswith(('_1', '_2')):
                pair_id = base_name[:-2]
                img_num = int(base_name[-1]) - 1 # 0 for _1, 1 for _2
                
                if pair_id not in image_pairs:
                    image_pairs[pair_id] = [None, None]
                
                image_pairs[pair_id][img_num] = os.path.join(directory, filename)
    
    # Filter out incomplete pairs and return a sorted list
    valid_pairs = [tuple(paths) for paths in image_pairs.values() if all(paths)]
    return sorted(valid_pairs)


# PURPOSE: Pair the two camera streams inside a wall-clock window, matching each
#          devID1 frame to the temporally nearest devID2 frame.
# INPUTS: directory (str); start_str, end_str ("HH-MM-SS", inclusive, in the
#         FILENAME clock); date_str ("YYYY-MM-DD" or None) — required when the
#         directory holds more than one acquisition day; pair_tolerance_s (float).
# OUTPUTS: sorted list of (devID1_path, devID2_path); each devID2 frame is used at
#          most once; [] on a malformed window.
# KEYWORDS: pairing, nearest_timestamp, date_filter, window
def find_pairs_in_window(directory, start_str, end_str, date_str=None,
                         pair_tolerance_s=2.0):
    """
    Pair images inside a time window by nearest filename timestamp.

    The two cameras do not stamp identical seconds: in the 2025-05-22/23
    archives the devID1 filename is typically 1 s ahead of its devID2
    partner. Grouping by an exact 'img-<timestamp>' base name (the previous
    behaviour) therefore silently discarded roughly 10% of all frames.
    Nearest-timestamp matching within `pair_tolerance_s` recovers them.
    """
    try:
        start_time = datetime.datetime.strptime(start_str, '%H-%M-%S').time()
        end_time = datetime.datetime.strptime(end_str, '%H-%M-%S').time()
    except ValueError:
        print("Error: invalid window format. Expected HH-MM-SS.")
        return []

    # Single-day selection. Without it a consolidated image directory holding
    # several acquisition days yields pairs from every day whose wall-clock
    # time falls in the window, silently mixing days into one "time series".
    want_date = None
    if date_str:
        try:
            want_date = datetime.datetime.strptime(str(date_str), '%Y-%m-%d').date()
        except ValueError:
            print(f"Error: invalid date '{date_str}'. Expected YYYY-MM-DD.")
            return []

    pattern = _MODERN_PATTERN

    per_device = {1: {}, 2: {}}
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if not match:
            continue
        timestamp_str, dev_id = match.groups()
        try:
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H-%M-%S')
        except ValueError:
            continue
        if want_date is not None and timestamp.date() != want_date:
            continue
        if start_time <= timestamp.time() <= end_time:
            per_device[int(dev_id)][timestamp] = os.path.join(directory, filename)

    dev2_times = sorted(per_device[2].keys())
    tolerance = datetime.timedelta(seconds=pair_tolerance_s)
    used_dev2 = set()
    pairs = []
    exact, offset = 0, 0
    for t1 in sorted(per_device[1].keys()):
        idx = bisect.bisect_left(dev2_times, t1)
        candidates = [dev2_times[i] for i in (idx - 1, idx) if 0 <= i < len(dev2_times)]
        candidates = [c for c in candidates if c not in used_dev2]
        if not candidates:
            continue
        best = min(candidates, key=lambda c: abs(c - t1))
        if abs(best - t1) <= tolerance:
            pairs.append((per_device[1][t1], per_device[2][best]))
            used_dev2.add(best)
            if best == t1:
                exact += 1
            else:
                offset += 1

    print(f"[PAIR] date={date_str or 'ANY'} window {start_str}--{end_str}: "
          f"devID1={len(per_device[1])} "
          f"devID2={len(per_device[2])} -> {len(pairs)} pairs "
          f"({exact} exact-timestamp, {offset} recovered by <= {pair_tolerance_s}s offset, "
          f"{len(per_device[1]) - len(pairs)} devID1 unmatched)")
    return sorted(pairs)

# PURPOSE: Express a configuration's analysis window in the image filename clock.
# INPUTS: cfg (clouds.config.Config).
# OUTPUTS: (start, end) as "HH-MM-SS" strings for find_pairs_in_window.
# KEYWORDS: window, clock, conversion
def image_clock_window(cfg):
    offset = datetime.timedelta(hours=float(cfg.dataset.clock_offset_h))
    start = datetime.datetime.strptime(cfg.dataset.window_start, "%H:%M:%S") - offset
    end = datetime.datetime.strptime(cfg.dataset.window_end, "%H:%M:%S") - offset
    return start.strftime("%H-%M-%S"), end.strftime("%H-%M-%S")
