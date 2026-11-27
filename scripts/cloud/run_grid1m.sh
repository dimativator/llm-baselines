#!/bin/bash
set -eu
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/tmp/grid1m-cache TIKTOKEN_CACHE_DIR=/tmp/grid1m-tiktoken HF_HOME=/tmp/grid1m-hf
exec python scripts/cloud/grid1m.py
