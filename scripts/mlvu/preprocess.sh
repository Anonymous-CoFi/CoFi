#!/usr/bin/env bash
set -euo pipefail

# MLVU-Dev: sample 1-fps candidates and cache SigLIP embeddings.
cofi preprocess \
  --dataset mlvu \
  --dataset-root /path/to/your/MLVU \
  --annotation /path/to/your/mlvu_dev.json \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/mlvu_siglip_1fps \
  --device cuda
