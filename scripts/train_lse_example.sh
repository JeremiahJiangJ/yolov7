#!/usr/bin/env bash
# Example: yolov7-tiny with local scale equivariance (LSE, README.md section 14) for objects whose apparent size changes
# as they move towards or away from the camera. Every setting is exposed below; ★ marks the ones to look at first. The
# LSE settings themselves are in data/lse/example.yaml (also ★-marked).
#
# Stages (run them all, or one at a time: bash scripts/train_lse_example.sh <stage> ...):
#   stage1   fine-tune the plain detector on your data, full augmentation (mosaic): the starting point
#   probe    one LSE epoch from stage 1, to read obj_px (median object size the detector sees) for target_size
#   stage2   LSE fine-tuning from stage 1 on single frames (no mosaic), with --lse-debug
#   base2    the same fine-tuning without LSE: the baseline stage 2 must beat
#   control  the same as stage2 with the canonicalizer frozen at the identity (random warps only): is the gain
#            canonicalization, or just warp augmentation?
#   test     all three on the clean validation set and with random local warps (--warp-test), per object size
#   detect   stage 2 on a folder of images / a video
#   all      stage1 base2 stage2 control test (default)
#
#   DATA=data/your_data.yaml bash scripts/train_lse_example.sh
#   DATA=data/your_data.yaml EPOCHS2=10 bash scripts/train_lse_example.sh stage2 test
#
# Before stage 2: run `probe`, set target_size in data/lse/example.yaml to the obj_px it prints (or near your anchors),
# and check the grid with: python tools/grid_capacity.py --data $DATA --img-size $IMG --target-size <obj_px>
# During stage 2 watch $PROJECT/stage2/lse_debug/status.txt (verdict ok, not STUCK) and the triplets in
# $PROJECT/stage2/lse_debug/epoch_*/ (red = magnified where objects are small / far, blue = shrunk, smooth grid).
# Single GPU; multi-GPU (DDP) and export.py are not supported / tested with LSE.
set -euo pipefail

# --- Data and resolution ----------------------------------------------------------------------------------------------
DATA=${DATA:-data/your_data.yaml}       # ★ data yaml (train: / val:, nc, names)
LABELS=${LABELS:-labels}                # label folder next to images/ (the fork's default is labels_upright)
IMG=${IMG:-1280}                        # ★ training / test size: 1920 x 1080 frames -> 1280 x 720 content. Set
                                        #   canon_size in the LSE config to the frames' aspect ratio ([252, 448] 16:9)
AREA_INT=${AREA_INT:-"1024 9216"}       # ★ per-size AP cut points, px^2 of the original frames: <32^2 (far, small),
                                        #   32^2 - 96^2, >96^2 (near, large). Adapt to your near / mid / far sizes
WARP_TESTS=${WARP_TESTS:-"0.5 1.0"}     # --warp-test strengths: random local warps of the validation images (labels
                                        #   follow), the same for every model: robustness to local scale changes

# --- Model and LSE ----------------------------------------------------------------------------------------------------
WEIGHTS=${WEIGHTS:-yolov7-tiny.pt}      # pretrained detector (loaded exactly: its anchors, implicit layers)
CFG=${CFG:-cfg/training/yolov7-tiny.yaml}
LSE=${LSE:-data/lse/example.yaml}       # ★ every LSE setting (★-marked inside)
LSE_DEBUG=${LSE_DEBUG:-1}               # ★ identity-basin verdicts every epoch, warp triplets every N epochs (0: off)
LSE_DEBUG_IMAGES=${LSE_DEBUG_IMAGES:-4} # validation images in the triplets
FREEZE_WARMUP=${FREEZE_WARMUP:-0}       # stage 2: first N epochs train the canonicalizer only (detector frozen), then
                                        #   everything. 0: off. Layer indices below assume decs: [input] on yolov7-tiny
                                        #   (DEC layers 0-2, detector 3-80)

# --- Training ---------------------------------------------------------------------------------------------------------
DEVICE=${DEVICE:-0}                     # GPU id (or cpu)
BS=${BS:-16}                            # ★ batch size; the same for every run so comparisons are fair. LSE adds little
                                        #   memory with checkpoint: true; lower it if stage 2 does not fit
WORKERS=${WORKERS:-8}
EPOCHS1=${EPOCHS1:-100}                 # stage 1 epochs
EPOCHS2=${EPOCHS2:-30}                  # ★ stage 2 / base2 / control epochs
SEED=${SEED:-0}                         # repeat with 3 seeds before trusting a difference (README.md section 6)
HYP1=${HYP1:-data/hyp.finetune.lse_mosaic.yaml}  # stage 1: full augmentation (mosaic; mixup 0)
HYP2=${HYP2:-data/hyp.finetune.lse.yaml}         # ★ stage 2: single frames (mosaic / mixup / paste-in 0); keep degrees,
                                                 #   shear, perspective 0 with LSE; `scale` (+/- 0.5) is the main scale
                                                 #   augmentation the detector gets besides warp_aug
PROJECT=${PROJECT:-runs/lse_example}
EXTRA=${EXTRA:-}                        # extra train.py flags for every run, e.g. "--adam" or "--patience 10"

# --- Inference --------------------------------------------------------------------------------------------------------
SOURCE=${SOURCE:-inference/images}      # detect: images / video / folder
CONF=${CONF:-0.25}                      # detect: confidence threshold

COMMON="--cfg $CFG --data $DATA --img-size $IMG $IMG --batch-size $BS --device $DEVICE --workers $WORKERS
        --label-folder-name $LABELS --project $PROJECT --exist-ok --seed $SEED --area-int $AREA_INT $EXTRA"
S1=$PROJECT/stage1/weights/best.pt
STAGES=${*:-all}
[[ " $STAGES " == *" all "* ]] && STAGES="stage1 base2 stage2 control test"

control_cfg() {  # the LSE config with the canonicalizer frozen at the identity: random warps (warp_aug) only
  mkdir -p "$PROJECT"
  { cat "$LSE"; printf 'lr_scale: 0.0\neq_weight: 0.0\nsize_weight: 0.0\ndet_grad_epochs: 0\n'; } > "$PROJECT/lse_control.yaml"
  echo "$PROJECT/lse_control.yaml"  # later keys win: same warps, canonicalizer never trains
}

for STAGE in $STAGES; do
  case $STAGE in
  stage1)
    python train.py $COMMON --weights $WEIGHTS --hyp $HYP1 --epochs $EPOCHS1 --name stage1 ;;
  probe)
    python train.py $COMMON --weights $S1 --hyp $HYP2 --epochs 1 --name probe --lse $LSE
    echo "obj_px (set target_size to about this):"; grep -o 'obj_px [0-9.]*' $PROJECT/probe/lse.txt | head -1 ;;
  stage2)
    FREEZE=""
    if [ "$FREEZE_WARMUP" -gt 0 ]; then FREEZE="--freeze $(seq -s ' ' 3 80) --unfreeze-epoch $FREEZE_WARMUP"; fi
    python train.py $COMMON --weights $S1 --hyp $HYP2 --epochs $EPOCHS2 --name stage2 --lse $LSE \
      --lse-debug $LSE_DEBUG --lse-debug-images $LSE_DEBUG_IMAGES $FREEZE ;;
  base2)
    python train.py $COMMON --weights $S1 --hyp $HYP2 --epochs $EPOCHS2 --name base2 ;;
  control)
    python train.py $COMMON --weights $S1 --hyp $HYP2 --epochs $EPOCHS2 --name control --lse "$(control_cfg)" \
      --lse-debug $LSE_DEBUG --lse-debug-images $LSE_DEBUG_IMAGES ;;
  test)
    for RUN in base2 control stage2; do
      W=$PROJECT/$RUN/weights/best.pt
      [ -f "$W" ] || { echo "skip $RUN: $W not found"; continue; }
      T="--weights $W --data $DATA --img-size $IMG --batch-size $BS --device $DEVICE --label-folder-name $LABELS
         --task val --area-int $AREA_INT --project $PROJECT/test --exist-ok"
      python test.py $T --name ${RUN}_clean
      for WT in $WARP_TESTS; do python test.py $T --name ${RUN}_warp$WT --warp-test $WT; done
    done ;;
  detect)
    python detect.py --weights $PROJECT/stage2/weights/best.pt --source $SOURCE --img-size $IMG --conf-thres $CONF \
      --device $DEVICE --project $PROJECT/detect --name stage2 --exist-ok ;;
  *)
    echo "unknown stage $STAGE (stage1 probe stage2 base2 control test detect all)"; exit 1 ;;
  esac
done
