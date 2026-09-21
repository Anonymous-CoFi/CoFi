#!/usr/bin/env bash
set -euo pipefail

# Video-MME: sample 1-fps candidates and cache SigLIP embeddings.
cofi preprocess \
  --dataset videomme \
  --dataset-root /path/to/your/Video-MME \
  --config configs/cofi_c4_f28.json \
  --cache-dir cache/videomme_siglip_1fps \
  --device cuda
