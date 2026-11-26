#!/bin/bash
set -eu
date -u
for path in / /tmp /home/jovyan /home/jovyan/mlsub-logs /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
    timeout 15 df -B1 --output=target,avail "$path"
done
