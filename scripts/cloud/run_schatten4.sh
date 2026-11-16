#!/usr/bin/env bash
export POLAR_GRID_VARIANT=schatten4
export POLAR_GRID_ROOT=/home/jovyan/shares/SR006.nfs3/dimativator-schatten4-124m-20261116
if [[ " $* " == *" --smoke "* ]]; then
    python scripts/cloud/test_schatten4.py || exit 0
fi
bash scripts/cloud/run_polar_grid.sh "$@"
