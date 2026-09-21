#!/usr/bin/env bash
set -euo pipefail

# Video-MME: save selected context panels and focus frames without VLM inference.
cofi select \
  --dataset videomme \
  --dataset-root /path/to/your/Video-MME \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/videomme_siglip_1fps \
  --routes-output outputs/videomme/cofi_routes.jsonl \
  --device cuda
