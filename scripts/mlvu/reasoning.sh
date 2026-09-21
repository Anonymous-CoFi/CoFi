#!/usr/bin/env bash
set -euo pipefail

# MLVU-Dev: run LLaVA-Video on previously selected visual units.
cofi reason \
  --config configs/cofi_c4_f28.json \
  --routes-input outputs/mlvu/cofi_routes.jsonl \
  --output outputs/mlvu/cofi_predictions.jsonl \
  --metrics-output outputs/mlvu/metrics.json \
  --llava-next-path /path/to/your/LLaVA-NeXT
