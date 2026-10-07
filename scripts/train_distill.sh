#!/usr/bin/env bash
# Distil a larger YOLOv7 (teacher) into yolov7-tiny (student). Usage: bash scripts/train_distill.sh [extra args]
#
# 1. Train the teacher first, on the same data / classes and at the same scale the student will use
#    (e.g. the same train_sources yaml): yolov7 / yolov7x with train.py, W6 / E6 / D6 / E6E with train_aux.py.
# 2. Distil: the student trains on the ground truth as usual, plus a loss pulling its detection outputs towards the
#    teacher's on the same augmented batches (utils/distill.py). The student adopts the teacher's anchors (autoanchor
#    is skipped). --distill-weight scales the distillation loss (1 ~ as strong as the ground-truth loss; try 0.5-2).
#
# Notes:
# - teacher and student must predict the same classes in the same order
# - a P6 teacher (W6 / E6 / D6 / E6E, stride 64) needs --img-size multiples of 64; the tiny student's P3-P5 outputs are
#   paired with the teacher's levels of the same stride
# - the teacher runs on every batch (no gradients): expect training to be several times slower than tiny alone
# - logged per epoch: "distill loss (box, obj, cls)" and TensorBoard train/distill_*; obj / cls are KL divergences
#   (0 = student matches teacher)
set -euo pipefail
TEACHER=${TEACHER:-runs/teacher/yolov7_camC/weights/best.pt}
DATA=${DATA:-data/mixed/template.yaml}
NAME=${NAME:-tiny_distilled}

python train.py \
  --weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml \
  --teacher "$TEACHER" --distill-weight 1.0 \
  --data "$DATA" --hyp data/hyp.finetune.sgd.yaml \
  --epochs 150 --batch-size 16 --img-size 1280 1280 \
  --seed 42 --device 0 --workers 8 \
  --fitness-metric-weights 0 0 1 0 --close-mosaic 10 --patience 30 \
  --project runs/distill --name "$NAME" "$@"
