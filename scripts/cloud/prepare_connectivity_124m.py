"""Remove exactly three authorized dense checkpoints, then verify the 124M launch."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time

os.environ.update(CONNECTIVITY_MODEL_SIZE='124m',
    CONNECTIVITY_ROOT='/workspace-SR006.nfs3/dimativator/mode-connectivity-124m-20261103',
    CONNECTIVITY_HF_PREFIX='mode_connectivity_124m_20261103',
    CONNECTIVITY_CHECKPOINT_RESERVE='2200000000',
    HF_HOME='/tmp/connectivity-124m-preflight-hf',
    TIKTOKEN_CACHE_DIR='/tmp/connectivity-124m-preflight-tiktoken',
    XDG_CACHE_HOME='/tmp/connectivity-124m-preflight-cache', TMPDIR='/tmp')

from connectivity_campaign import ROOT, prepare, preflight, publish, write_json
from connectivity_experiment import Llama, config
from connectivity_guard import enforce
from prune_slim_checkpoint import INSPECT

SIZES = dict(GaLore=3286026585, Apollo=2666750433, AdamW=3060218301)


def remove_checkpoint(path: Path, expected_size: int) -> dict:
    mount = Path('/home/jovyan')
    relative = path.relative_to(mount)
    if relative.parts[0] != 'dimativator':
        raise RuntimeError('Checkpoint is outside the authorized user directory')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(mount, flags)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        before = os.statvfs(directory)
        try:
            info = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return dict(path=str(path), status='already_absent', timestamp=time.time())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != expected_size:
            raise RuntimeError('Checkpoint does not match the inventoried file: '+str(path))
        os.unlink(path.name, dir_fd=directory)
        after = os.statvfs(directory)
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('Checkpoint deletion verification failed')
        return dict(path=str(path), status='deleted', size_bytes=info.st_size,
            allocated_bytes=info.st_blocks*512, before_available_bytes=before.f_bavail*before.f_frsize,
            after_available_bytes=after.f_bavail*after.f_frsize, timestamp=time.time())
    finally:
        os.close(directory)


if __name__ == '__main__':
    enforce(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt = ROOT/'old_dense_cleanup.json'
    previous = json.loads(receipt.read_text()) if receipt.is_file() else dict(deleted=[])
    completed = {row['path'] for row in previous['deleted'] if row['status'] in ('deleted','already_absent')}
    for name, filename in INSPECT.items():
        if filename not in completed:
            row = remove_checkpoint(Path(filename), SIZES[name])
            row['name'] = name
            previous['deleted'].append(row)
            previous['storage_after'] = enforce(ROOT)
            write_json(receipt, previous)
    publish([receipt], verify=True)
    prepare(False)  # Verify exact training/validation token hashes.
    import torch
    torch.set_num_threads(4)
    model = Llama(config())
    count = sum(p.numel() for p in model.parameters())
    assert count == 123587328 and model.transformer.wte.weight is model.lm_head.weight
    write_json(ROOT/'model_verified.json', dict(parameters=count, config=vars(model.config),
        tied_embeddings=True, timestamp=time.time()))
    publish([ROOT/'model_verified.json'], verify=True)
    preflight()
