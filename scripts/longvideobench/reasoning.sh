#!/usr/bin/env bash
set -euo pipefail

# LongVideoBench validation: run LLaVA-Video on selected visual units.
cofi reason \
  --config configs/cofi_c4_f28.json \
  --routes-input outputs/longvideobench/cofi_routes.jsonl \
  --output outputs/longvideobench/cofi_predictions.jsonl \
  --metrics-output outputs/longvideobench/metrics.json \
  --llava-next-path /path/to/your/LLaVA-NeXT
