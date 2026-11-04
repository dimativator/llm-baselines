#!/usr/bin/env bash
set -eu
export CONNECTIVITY_MODEL_SIZE=124m
export CONNECTIVITY_ROOT=/workspace-SR006.nfs3/dimativator/mode-connectivity-124m-spectral-20261104
export CONNECTIVITY_HF_PREFIX=mode_connectivity_124m_spectral_20261104
export CONNECTIVITY_CHECKPOINT_RESERVE=2200000000
export CONNECTIVITY_PATH_SPECTRAL=1
export CONNECTIVITY_LANES=4
export CONNECTIVITY_GPUS_PER_RUN=2
if [[ "${1:-}" == "--prepare" ]]; then
    export HF_HOME=/tmp/connectivity-spectral-prepare-hf
    export TIKTOKEN_CACHE_DIR=/tmp/connectivity-spectral-tiktoken
    export XDG_CACHE_HOME=/tmp/connectivity-spectral-cache
    export TMPDIR=/tmp
    export CONNECTIVITY_REVISION="$(git rev-parse HEAD)"
    python scripts/cloud/prepare_connectivity_124m_spectral.py
else
    bash scripts/cloud/run_connectivity_campaign.sh "$@"
fi
