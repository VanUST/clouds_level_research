#!/usr/bin/env bash
set -u
cd /home/iustimov/clouds
PY="python experiments/matchers/evaluate.py"
CFG='{"grid_stride": 8, "certainty_threshold": 0.5}'
C05='{"grid_stride": 8, "certainty_threshold": 0.05}'
DIS='{"grid_stride": 8}'
run(){ echo "=== $1 $2 ==="; $PY --backend "$1" --date 2025-05-23 ${2:+--tag "$2"} --cfg-json "$3" 2>&1 | grep -viE "warning|warn|INFO|it/s|MB/s|autocast|resolver|register_resolver|hydra|weights from|Downloading" | tail -40; }
run kornia_loftr   ""   '{}'
run efficient_loftr ""  '{}'
run roma           ""   "$CFG"
run roma           "_c05" "$C05"
run dkm            ""   "$CFG"
run dkm            "_c05" "$C05"
run da3            ""   '{}'
run cv2_dis_flow   ""   "$DIS"
run cv2_sgbm       ""   "$DIS"
run disk_lightglue ""   '{}'
run mast3r         ""   '{}'
echo RERUN23DONE
