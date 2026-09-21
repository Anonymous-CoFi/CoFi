#!/usr/bin/env bash
set -euo pipefail

# MLVU-Dev: save selected context panels and focus frames without VLM inference.
cofi select \
  --dataset mlvu \
  --dataset-root /path/to/your/MLVU \
  --annotation /path/to/your/mlvu_dev.json \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/mlvu_siglip_1fps \
  --routes-output outputs/mlvu/cofi_routes.jsonl \
  --device cuda
