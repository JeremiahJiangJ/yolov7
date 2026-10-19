#!/usr/bin/env bash
# Input canonicalization, the ablation of the default latent placement (data/lse/example_input.yaml, README.md section
# 14): one canonicalizer warps the image, the unchanged detector runs on it, boxes are mapped back. The same stages,
# options and comparisons as scripts/train_lse_example.sh (see it for every variable, ★ on the important ones), with
# the input config and its own run directory:
#   DATA=data/your_data.yaml bash scripts/train_lse_input_example.sh                # stage1 base2 stage2 control test
# To compare with the default (latent) placement, run scripts/train_lse_example.sh first and reuse its stage 1:
#   STAGE1=runs/lse_example/stage1/weights/best.pt DATA=... bash scripts/train_lse_input_example.sh stage2 control test stability
# Compare stage2 here with base2 and control here, and with stage2 of the latent example.
set -euo pipefail
export LSE=${LSE:-data/lse/example_input.yaml}
export PROJECT=${PROJECT:-runs/lse_input_example}
if [ -n "${STAGE1:-}" ]; then  # reuse another run's stage 1
  mkdir -p "$PROJECT/stage1/weights"
  ln -sf "$(realpath "$STAGE1")" "$PROJECT/stage1/weights/best.pt"
fi
exec bash "$(dirname "$0")/train_lse_example.sh" "$@"
