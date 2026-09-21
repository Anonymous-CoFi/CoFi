#!/usr/bin/env bash
set -euo pipefail

# LongVideoBench validation: save selected visual units without VLM inference.
cofi select \
  --dataset longvideobench \
  --dataset-root /path/to/your/LongVideoBench \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/longvideobench_siglip_1fps \
  --routes-output outputs/longvideobench/cofi_routes.jsonl \
  --device cuda
