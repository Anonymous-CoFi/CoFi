#!/usr/bin/env bash
set -euo pipefail

# MLVU-Dev: select visual units and run LLaVA-Video in one invocation.
cofi run \
  --dataset mlvu \
  --dataset-root /path/to/your/MLVU \
  --annotation /path/to/your/mlvu_dev.json \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/mlvu_siglip_1fps \
  --routes-output outputs/mlvu/cofi_routes.jsonl \
  --output outputs/mlvu/cofi_predictions.jsonl \
  --metrics-output outputs/mlvu/metrics.json \
  --device cuda \
  --llava-next-path /path/to/your/LLaVA-NeXT
