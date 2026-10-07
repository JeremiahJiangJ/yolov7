#!/usr/bin/env bash
# Train the data/mixed experiment ladder with identical settings, to measure each change's effect
# (see data/mixed/README.md). Every config except 1 has the same images per epoch (total images over all sources),
# so --epochs gives the same training budget; for 1 (target only) EPOCHS_TARGET is scaled to match.
# Usage: bash scripts/train_mixed_ladder.sh [extra train.py args, e.g. --area-int 300 650 1250]
set -euo pipefail
EPOCHS=${EPOCHS:-100}
EPOCHS_TARGET=${EPOCHS_TARGET:-$EPOCHS}   # e.g. EPOCHS * (all images / target images) for an equal budget
COMMON=(--weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.finetune.sgd.yaml
        --batch-size 32 --img-size 640 640 --seed 42 --device 0 --workers 8 --label-folder-name labels
        --fitness-metric-weights 0 0 1 0 --project runs/ladder --exist-ok)

python train.py "${COMMON[@]}" --epochs "$EPOCHS_TARGET" --data data/mixed/1_default_target.yaml --name 1_default_target "$@"
for cfg in 2_default_pooled 3_proportional_fit 4_proportional_native 5_weighted_native; do
  python train.py "${COMMON[@]}" --epochs "$EPOCHS" --data "data/mixed/$cfg.yaml" --name "$cfg" "$@"
done

# Fair comparison: evaluate every best.pt on the same target val set at the same scale. Native-scale models must be
# tested at native scale (--resize native), resized models at --img-size (--resize fit); add both if in doubt.
for run in 1_default_target 2_default_pooled 3_proportional_fit; do
  python test.py --weights "runs/ladder/$run/weights/best.pt" --data data/mixed/1_default_target.yaml \
    --img-size 640 --label-folder-name labels --resize fit --project runs/ladder_test --name "$run" --exist-ok
done
for run in 4_proportional_native 5_weighted_native; do
  python test.py --weights "runs/ladder/$run/weights/best.pt" --data data/mixed/1_default_target.yaml \
    --img-size 1280 --label-folder-name labels --resize native --project runs/ladder_test --name "$run" --exist-ok
done
