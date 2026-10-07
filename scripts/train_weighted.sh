#!/usr/bin/env bash
# Weighted multi-source, native-scale fine-tuning of yolov7-tiny. Usage: bash scripts/train_weighted.sh [extra args]
# --img-size 640 is the training crop size: native sources validate on full frames regardless of the 2nd value.
set -euo pipefail
DATA=${DATA:-data/mixed/5_weighted_native.yaml}
NAME=${NAME:-weighted_native}

python train.py \
  --weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml \
  --data "$DATA" --hyp data/hyp.finetune.sgd.yaml \
  --epochs 100 --batch-size 32 --img-size 640 640 \
  --seed 42 --device 0 --workers 8 \
  --label-folder-name labels \
  --project runs/mixed --name "$NAME" "$@"
