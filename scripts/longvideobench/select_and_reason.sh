#!/usr/bin/env bash
set -euo pipefail

# LongVideoBench validation: select units and run LLaVA-Video in one invocation.
cofi run \
  --dataset longvideobench \
  --dataset-root /path/to/your/LongVideoBench \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/longvideobench_siglip_1fps \
  --routes-output outputs/longvideobench/cofi_routes.jsonl \
  --output outputs/longvideobench/cofi_predictions.jsonl \
  --metrics-output outputs/longvideobench/metrics.json \
  --device cuda \
  --llava-next-path /path/to/your/LLaVA-NeXT
