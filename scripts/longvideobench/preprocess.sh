#!/usr/bin/env bash
set -euo pipefail

# LongVideoBench validation: sample 1-fps candidates and cache SigLIP embeddings.
cofi preprocess \
  --dataset longvideobench \
  --dataset-root /path/to/your/LongVideoBench \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/longvideobench_siglip_1fps \
  --device cuda
