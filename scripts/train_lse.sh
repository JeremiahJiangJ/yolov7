#!/usr/bin/env bash
# Local scale canonicalization experiment (README.md, section 14): is a deep equilibrium canonicalizer worth its cost
# for yolov7-tiny, compared with simply augmenting with the same local scaling? Usage: bash scripts/train_lse.sh
# [extra train.py args]. STEPS="aug input" runs a subset.
#
# Steps, identical except for the change named:
#   base    stock yolov7-tiny
#   aug     + random local-scale augmentation (lse_aug 0.5): the baseline the canonicalizer has to beat
#   input   + input canonicalization (cfg yolov7-tiny-lse-input.yaml) and its consistency loss, with the augmentation
#   latent  + latent canonicalization of the P3 / P4 stages instead (cfg yolov7-tiny-lse-latent.yaml)
# Each is tested on the clean val set and under random local scaling (test.py --lse-warp, the same warps for every
# model); compare mAP, the drop from clean to warped, and the speed line of test.py.
#
# Notes:
# - start from the stock weights: the canonicalizers start at the identity warp, so the pretrained detector is
#   unchanged at the first step
# - lse_equi > 0 (objectness equivariance) doubles the training forward; try it once the cheaper steps are in
# - not supported with lse models: export.py, train_aux.py, DataParallel (DDP is), --teacher with input
#   canonicalization
set -euo pipefail
DATA=${DATA:-data/mixed/template.yaml}
STEPS=${STEPS:-base aug input latent}
IMG=${IMG:-1280}
PROJECT=${PROJECT:-runs/lse}
WARP=${WARP:-0.25}  # test-time local scaling strength

hyp_with() {  # hyp.finetune.sgd.lse.yaml with key=value overrides -> a file in $PROJECT
  local out="$PROJECT/hyp_$1.yaml"; shift
  mkdir -p "$PROJECT"
  cp data/hyp.finetune.sgd.lse.yaml "$out"
  for kv in "$@"; do sed -i "s/^${kv%%=*}:.*/${kv%%=*}: ${kv#*=}/" "$out"; done
  echo "$out"
}

for step in $STEPS; do
  case $step in
    base)   CFG=cfg/training/yolov7-tiny.yaml;            HYP=data/hyp.finetune.sgd.yaml ;;
    aug)    CFG=cfg/training/yolov7-tiny.yaml;            HYP=$(hyp_with aug lse_aug=0.5) ;;
    input)  CFG=cfg/training/yolov7-tiny-lse-input.yaml;  HYP=$(hyp_with input lse_aug=0.5) ;;
    latent) CFG=cfg/training/yolov7-tiny-lse-latent.yaml; HYP=$(hyp_with latent lse_aug=0.5) ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
  python train.py \
    --weights yolov7-tiny.pt --cfg "$CFG" --data "$DATA" --hyp "$HYP" \
    --epochs 150 --batch-size 16 --img-size "$IMG" "$IMG" \
    --seed 42 --device 0 --workers 8 \
    --fitness-metric-weights 0 0 1 0 --close-mosaic 10 --patience 30 \
    --project "$PROJECT" --name "$step" --exist-ok "$@"
  W="$PROJECT/$step/weights/best.pt"
  python test.py --data "$DATA" --weights "$W" --img-size "$IMG" --project "$PROJECT" --name "${step}_test" --exist-ok
  python test.py --data "$DATA" --weights "$W" --img-size "$IMG" --lse-warp "$WARP" --project "$PROJECT" \
    --name "${step}_test_warped" --exist-ok
done
