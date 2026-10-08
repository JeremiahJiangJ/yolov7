#!/usr/bin/env bash
# Train the data/mixed experiment ladder with identical settings, to measure what each step changes
# (see data/mixed/README.md):
#   1 stock, target only -> 2 stock, pooled -> 3 multi-source, unweighted -> 4 native scale -> 5 weighted -> 6 + rfs
# Steps 2-6 have the same samples per epoch (as many as frames in all sources), so the same --epochs is the same
# training budget; step 1 (target only) uses EPOCHS_TARGET, scaled to match by default if you set it.
# Usage: bash scripts/train_mixed_ladder.sh [extra train.py args, e.g. --area-int 300 650 1250]
#        STEPS="1 2 5 6" bash scripts/train_mixed_ladder.sh   # a subset
set -euo pipefail
EPOCHS=${EPOCHS:-100}
EPOCHS_TARGET=${EPOCHS_TARGET:-$EPOCHS}   # e.g. EPOCHS * (frames in all sources / target frames) for an equal budget
STEPS=${STEPS:-"1 2 3 4 5 6"}
COMMON=(--weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.finetune.sgd.yaml
        --batch-size 32 --img-size 640 640 --seed 42 --device 0 --workers 8 --label-folder-name labels
        --fitness-metric-weights 0 0 1 0 --project runs/ladder --exist-ok)
CONFIGS=(x 1_default_target 2_default_pooled 3_proportional_fit 4_proportional_native 5_weighted_native
         6_weighted_native_rfs)

for s in $STEPS; do
  cfg=${CONFIGS[$s]}
  epochs=$EPOCHS; [ "$s" = 1 ] && epochs=$EPOCHS_TARGET
  python train.py "${COMMON[@]}" --epochs "$epochs" --data "data/mixed/$cfg.yaml" --name "$cfg" "$@"
done

# Fair comparison: every best.pt on the same target val set, each at the scale it was trained for: resized models at
# --img-size (--resize fit), native-scale models on full frames (--resize native, --img-size = the target's long side).
# --freq-groups adds mAP per class-frequency group (rare / common / frequent in the target's training labels).
for s in $STEPS; do
  cfg=${CONFIGS[$s]}
  if [ "$s" -le 3 ]; then scale=(--img-size 640 --resize fit); else scale=(--img-size 1280 --resize native); fi
  python test.py --weights "runs/ladder/$cfg/weights/best.pt" --data data/mixed/1_default_target.yaml \
    "${scale[@]}" --label-folder-name labels --freq-groups --project runs/ladder_test --name "$cfg" --exist-ok
done
