#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/sigma-hf-cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
python scripts/cloud/sigma_campaign.py "$@"
code=$?
echo "SIGMA_COORDINATOR_EXIT=$code"
# Cloud.ru hides logs for Failed jobs. Worker/failure markers are authoritative.
exit 0
