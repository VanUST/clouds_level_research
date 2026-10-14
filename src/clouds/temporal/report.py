# PURPOSE: Human-readable and machine-readable summaries of a finished tracking
#          run: the per-frame cloud-base series, per-track quality summaries and a
#          formatted report. Pure presentation — it reads track state and never
#          changes it.
# INPUTS: the track list and tracker settings, or a tracker instance.
# OUTPUTS: lists of dicts, or a report string.
# KEYWORDS: report, summary, export, cloud_base_series, tracks
from __future__ import annotations

import numpy as np


def cbh_series(tracks: list, cfg: dict) -> list[dict]:
    frames = {}
    for track in tracks:
        for entry in track.history:
            ts = entry["timestamp"]
            key = round(ts, 1)
            if key not in frames:
                frames[key] = {
                    "timestamp": ts,
                    "cbh": None,
                    "tracks": [],
                }
    for track in tracks:
        for entry in track.history:
            if entry["status"] != "CONFIRMED":
                continue
            ts = entry["timestamp"]
            key = round(ts, 1)
            if key in frames:
                frames[key]["tracks"].append({
                    "track_id": track.track_id,
                    "h": entry["h"],
                    "size": entry["size"],
                    "quality": track.quality_score(cfg),
                })

    sorted_keys = sorted(frames.keys())
    result = []
    for key in sorted_keys:
        f = frames[key]
        if f["tracks"]:
            f["tracks"].sort(key=lambda x: x["h"])
            best = f["tracks"][0]
            f["cbh"] = best["h"]
        result.append(f)
    return result


def track_summaries(tracks: list, cfg: dict) -> list[dict]:
    summaries = []
    for track in tracks:
        if track.status == TrackStatus.DEAD:
            continue
        summaries.append({
            "track_id": track.track_id,
            "status": track.status.name,
            "frames": len(track.history),
            "lifetime_s": track.lifetime_seconds(),
            "quality": track.quality_score(cfg),
            "avg_h": float(np.mean([e["h"] for e in track.history])),
            "final_h": float(track.x[0]),
            "final_size": float(track.x[6]),
            "outlier_rate": (track.outlier_count /
                             max(len(track.history), 1)),
        })
    return summaries


# PURPOSE: Render a tracking run's outcome as a printable report.
# INPUTS: tracker — an object exposing tracks, dead_tracks, cfg, frame_count,
#         total_clusters_seen (a TemporalFilter).
# OUTPUTS: the report as a string. Never mutates the tracker.
# KEYWORDS: report, format, summary
def format_report(tracker) -> str:
    summaries = track_summaries(tracker.tracks, tracker.cfg)
    confirmed = [s for s in summaries if s["status"] == "CONFIRMED"]
    prob = [s for s in summaries if s["status"] == "PROBATION"]

    out = [f"\n{'='*70}",
           "  TEMPORAL FILTER REPORT",
           f"{'='*70}",
           f"  Frames processed:            {tracker.frame_count}",
           f"  Total clusters seen:         {tracker.total_clusters_seen}",
           f"  Confirmed tracks:            {len(confirmed)}",
           f"  Probation tracks:            {len(prob)}",
           f"  Dead tracks:                 {len(tracker.dead_tracks)}",
           f"  Total tracks created:        {len(tracker.tracks)}",
           f"{'='*70}"]
    if confirmed:
        out.append(f"\n  {'ID':>4} {'Quality':>8} {'Frames':>7} {'Life(s)':>8} "
                   f"{'Avg H(m)':>9} {'Final H(m)':>10} {'Outlier%':>9}")
        out.append(f"  {'-'*62}")
        for s_ in sorted(confirmed, key=lambda x: x["quality"], reverse=True):
            out.append(f"  {s_['track_id']:4d} {s_['quality']:8.3f} {s_['frames']:7d} "
                       f"{s_['lifetime_s']:8.1f} {s_['avg_h']:9.1f} {s_['final_h']:10.1f} "
                       f"{s_['outlier_rate']:9.1%}")
    return "\n".join(out)
