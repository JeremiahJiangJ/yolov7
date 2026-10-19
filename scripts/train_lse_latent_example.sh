#!/usr/bin/env bash
# Latent local scale equivariance example: a canonicalizer at every backbone stage P1 - P5 (data/lse/example_latent.yaml,
# README.md section 14). The same stages, options and comparisons as scripts/train_lse_example.sh (see it for every
# variable, ★ on the important ones), with the latent config and its own run directory:
#   DATA=data/your_data.yaml bash scripts/train_lse_latent_example.sh                # stage1 base2 stage2 control test
#   DATA=data/your_data.yaml bash scripts/train_lse_latent_example.sh stage2 test    # reuse a stage 1 (STAGE1 below)
# To compare with input canonicalization, run scripts/train_lse_example.sh too and reuse its stage 1:
#   STAGE1=runs/lse_example/stage1/weights/best.pt DATA=... bash scripts/train_lse_latent_example.sh stage2 control test
# Compare stage2 here with base2 and control here, and with stage2 of the input example.
set -euo pipefail
export LSE=${LSE:-data/lse/example_latent.yaml}
export PROJECT=${PROJECT:-runs/lse_latent_example}
if [ -n "${STAGE1:-}" ]; then  # reuse another run's stage 1
  mkdir -p "$PROJECT/stage1/weights"
  ln -sf "$(realpath "$STAGE1")" "$PROJECT/stage1/weights/best.pt"
fi
exec bash "$(dirname "$0")/train_lse_example.sh" "$@"
