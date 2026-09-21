#!/usr/bin/env bash
set -euo pipefail

# Video-MME: select visual units and run LLaVA-Video in one invocation.
cofi run \
  --dataset videomme \
  --dataset-root /path/to/your/Video-MME \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/videomme_siglip_1fps \
  --routes-output outputs/videomme/cofi_routes.jsonl \
  --output outputs/videomme/cofi_predictions.jsonl \
  --metrics-output outputs/videomme/metrics.json \
  --device cuda \
  --llava-next-path /path/to/your/LLaVA-NeXT
