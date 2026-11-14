#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HOME=/tmp/polar-grid-hf
export XDG_CACHE_HOME=/tmp/polar-grid-cache
export TIKTOKEN_CACHE_DIR=/tmp/polar-grid-tiktoken
export TORCHINDUCTOR_CACHE_DIR=/tmp/polar-grid-inductor
export TMPDIR=/tmp
python scripts/cloud/polar_grid_worker.py "$@"
code=$?
echo "POLAR_GRID_WORKER_EXIT=$code"
# Cloud.ru hides diagnostic output when an entrypoint exits nonzero.
# Per-lane status.json and this explicit exit marker remain authoritative.
exit 0
