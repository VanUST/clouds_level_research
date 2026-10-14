#!/usr/bin/env bash
set -u
cd /home/iustimov/clouds
PY="python experiments/matchers/evaluate.py"
CFG_DENSE='{"grid_stride": 8, "certainty_threshold": 0.5}'
DIS_CFG='{"grid_stride": 8}'
run() {
  echo "=== $1 $2 ==="
  $PY --backend "$1" --date "$2" --cfg-json "$3" 2>&1 | grep -viE "warning|warn|Downloading|it/s|MB/s|INFO" | tail -40
}
for d in 2025-05-23 2025-05-22; do
  run kornia_loftr   $d '{}'
  run efficient_loftr $d '{}'
  run roma           $d "$CFG_DENSE"
  run dkm            $d "$CFG_DENSE"
  run cv2_dis_flow   $d "$DIS_CFG"
  run cv2_sgbm       $d "$DIS_CFG"
  run disk_lightglue $d '{}'
done
echo ALLDONE
