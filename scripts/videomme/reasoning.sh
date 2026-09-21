#!/usr/bin/env bash
set -euo pipefail

# Video-MME: run LLaVA-Video on previously selected visual units.
cofi reason \
  --config configs/cofi_c4_f28.json \
  --routes-input outputs/videomme/cofi_routes.jsonl \
  --output outputs/videomme/cofi_predictions.jsonl \
  --metrics-output outputs/videomme/metrics.json \
  --llava-next-path /path/to/your/LLaVA-NeXT
