# PURPOSE: Merge disparity clusters that sit at nearly the same height. HDBSCAN
#          fragments a single physical cloud layer into several clusters; merging
#          by height reconstitutes the layer before it reaches the tracker.
# INPUTS: a list of measurements and a height tolerance in metres.
# OUTPUTS: a merged list of measurements, size-weighted where groups combine.
# KEYWORDS: merge, cluster, fragmentation, cloud_layer
from __future__ import annotations

import numpy as np

from .measurement import ClusterMeasurement


# PURPOSE: Merge clusters with similar heights into composite "super-clusters"
#          to counter HDBSCAN fragmentation of a single cloud layer into
#          multiple small clusters.
# INPUTS: clusters (list[ClusterMeasurement]), height_tolerance_m (float)
# OUTPUTS: merged list[ClusterMeasurement]
# KEYWORDS: merge, height, cluster, fragmentation, consolidation
def merge_clusters_by_height(clusters: list, tol_m: float) -> list:
    if len(clusters) <= 1:
        return clusters
    sorted_c = sorted(clusters, key=lambda c: c.h)
    merged = []
    current_group = [sorted_c[0]]
    for c in sorted_c[1:]:
        group_avg_h = np.mean([m.h for m in current_group])
        if abs(c.h - group_avg_h) < tol_m:
            current_group.append(c)
        else:
            merged.append(merge_group(current_group))
            current_group = [c]
    if current_group:
        merged.append(merge_group(current_group))
    return merged


def merge_group(group: list) -> ClusterMeasurement:
    total_size = sum(c.size for c in group)
    weights = np.array([c.size for c in group], dtype=float)
    weights /= weights.sum()
    return ClusterMeasurement(
        h=float(np.average([c.h for c in group], weights=weights)),
        h_err=float(np.average([c.h_err for c in group], weights=weights)),
        u=float(np.average([c.u for c in group], weights=weights)),
        v=float(np.average([c.v for c in group], weights=weights)),
        size=total_size,
        disp_x=float(np.average([c.disp_x for c in group], weights=weights)),
        disp_std=float(np.average([c.disp_std for c in group], weights=weights)),
        cluster_id=group[0].cluster_id,
    )

