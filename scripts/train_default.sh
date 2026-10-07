#!/usr/bin/env bash
# Stock YOLOv7 fine-tuning of yolov7-tiny (no train_sources). Usage: bash scripts/train_default.sh [extra train.py args]
set -euo pipefail
DATA=${DATA:-data/mixed/2_default_pooled.yaml}   # or data/mixed/1_default_target.yaml
NAME=${NAME:-default_pooled}

python train.py \
  --weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml \
  --data "$DATA" --hyp data/hyp.finetune.sgd.yaml \
  --epochs 100 --batch-size 32 --img-size 640 640 \
  --seed 42 --device 0 --workers 8 \
  --label-folder-name labels \
  --project runs/mixed --name "$NAME" "$@"
