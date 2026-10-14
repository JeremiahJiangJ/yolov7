#!/usr/bin/env bash
# Evaluates every run of scripts/train_lse_experiments.sh: clean validation set, and the same images rescaled by random
# local warps (--warp-test: identical warps for every model), with per-size AP. Results: runs/lse_test/<run>_<test>/
# and the printed tables (also saved to runs/lse_test/summary.txt).
#   DATA=data/your_data.yaml bash scripts/test_lse_experiments.sh
set -euo pipefail
DATA=${DATA:-data/your_data.yaml}
IMG=${IMG:-640}
LABELS=${LABELS:-labels}
AREA=${AREA:-"1024 9216 36864"}   # per-size AP cut points in px^2 (32^2, 96^2, 192^2): adapt to your object sizes
mkdir -p runs/lse_test

for W in runs/lse/*/weights/best.pt; do
  RUN=$(basename "$(dirname "$(dirname "$W")")")
  for WT in 0 0.5 1.0; do
    echo "=== $RUN warp-test $WT" | tee -a runs/lse_test/summary.txt
    python test.py --weights "$W" --data "$DATA" --img-size $IMG --batch-size 32 --device 0 \
      --label-folder-name $LABELS --area-int $AREA --warp-test $WT \
      --project runs/lse_test --name ${RUN}_warp$WT --exist-ok 2>&1 | tee -a runs/lse_test/summary.txt
  done
done
