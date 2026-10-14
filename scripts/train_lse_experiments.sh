#!/usr/bin/env bash
# Local scale equivariance experiments on yolov7-tiny (README.md, section 14). Plain train.py commands: comment out what
# you do not need. Every run starts from yolov7-tiny.pt; LSE runs start as exactly that detector (identity warps).
#   DATA=data/your_data.yaml bash scripts/train_lse_experiments.sh
# Then evaluate: bash scripts/test_lse_experiments.sh
#
# Before the experiments, set target_size in data/lse/exp/e2-e5 (TARGET_SIZE) to your median object size: run
#   python train.py <the A1 command below with --epochs 1 --name size_probe>
# and read obj_px in runs/lse/size_probe/lse.txt (the size the detector sees, sqrt(w * h) in pixels).
set -euo pipefail
DATA=${DATA:-data/your_data.yaml}
IMG=${IMG:-640}
EPOCHS=${EPOCHS:-100}
BS=${BS:-32}               # LSE runs need more memory: if they do not fit, lower BS for all runs, baselines included
SEEDS=${SEEDS:-"0 1 2"}
LABELS=${LABELS:-labels}
COMMON="--weights yolov7-tiny.pt --cfg cfg/training/yolov7-tiny.yaml --data $DATA --epochs $EPOCHS --batch-size $BS
        --img-size $IMG $IMG --device 0 --workers 8 --label-folder-name $LABELS --project runs/lse"

for S in $SEEDS; do
  # --- A. No augmentation (data/hyp.finetune.noaug.yaml): where LSE has the most to add. Which placement works? ------
  H=data/hyp.finetune.noaug.yaml
  python train.py $COMMON --hyp $H --seed $S --name A0_base_s$S
  python train.py $COMMON --hyp $H --seed $S --name A1_input_eq_s$S --lse data/lse/exp/e1_input_eq.yaml
  python train.py $COMMON --hyp $H --seed $S --name A2_input_s$S    --lse data/lse/exp/e2_input.yaml
  python train.py $COMMON --hyp $H --seed $S --name A3_open_p3_s$S  --lse data/lse/exp/e3_open_p3.yaml
  python train.py $COMMON --hyp $H --seed $S --name A4_p345_s$S     --lse data/lse/exp/e4_stages_p345.yaml
  # python train.py $COMMON --hyp $H --seed $S --name A5_input_open_p3_s$S --lse data/lse/exp/e5_input_open_p3.yaml

  # --- B. Full augmentation with mosaic (data/hyp.finetune.lse_mosaic.yaml: mixup 0; per-tile warps). Does it still
  #        help when mosaic and scale augmentation already teach scale robustness? Use the best one or two of A. -------
  H=data/hyp.finetune.lse_mosaic.yaml
  python train.py $COMMON --hyp $H --seed $S --name B0_base_s$S
  python train.py $COMMON --hyp $H --seed $S --name B2_input_s$S    --lse data/lse/exp/e2_input.yaml
  python train.py $COMMON --hyp $H --seed $S --name B3_open_p3_s$S  --lse data/lse/exp/e3_open_p3.yaml

  # --- C. Two stages: B0 (full augmentation, no LSE), then fine-tuned for 30 epochs on single frames
  #        (data/hyp.finetune.lse.yaml: no mosaic / mixup / paste-in) with and without LSE. ----------------------------
  H=data/hyp.finetune.lse.yaml
  C2="--weights runs/lse/B0_base_s$S/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --data $DATA --epochs 30
      --batch-size $BS --img-size $IMG $IMG --device 0 --workers 8 --label-folder-name $LABELS --project runs/lse"
  python train.py $C2 --hyp $H --seed $S --name C0_base_s$S
  python train.py $C2 --hyp $H --seed $S --name C2_input_s$S   --lse data/lse/exp/e2_input.yaml
  python train.py $C2 --hyp $H --seed $S --name C3_open_p3_s$S --lse data/lse/exp/e3_open_p3.yaml
done
