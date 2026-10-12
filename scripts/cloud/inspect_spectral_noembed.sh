#!/usr/bin/env bash
set -euo pipefail
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/jovyan/.local-efficient}"
export HF_HOME=/tmp/spectral-noembed-inspector
export HF_HUB_DISABLE_XET=1
python scripts/cloud/inspect_spectral_noembed.py
