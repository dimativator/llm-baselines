"""Delete one explicitly authorized file and inspect three other checkpoints read-only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import time

from connectivity_guard import enforce

OUTPUT = Path('/workspace-SR006.nfs3/dimativator/mode-connectivity-20m-20261103')
TARGET = Path('/workspace-SR006.nfs3/dimativator/slim-normuon-500m-20261024/adaptive_muon_500M_2xC_20261024/slim_normuon_500m_bf16_lr1e-3_wd1e-4_2xC_2gpu/ckpts/67911/main.pt')
EXPECTED_SIZE = 3957701506
INSPECT = {
    'GaLore': '/home/jovyan/dimativator/projected-500m-precisions-20261013/2xChinchilla_500M_projected_wd1e-2_precisions_20261013/llama500M_galore_fp8_full_wd1e-2_2xC_warmup2000_2gpu_20261013/ckpts/67911/main.pt',
    'Apollo': '/home/jovyan/dimativator/projected-500m-precisions-20261013/2xChinchilla_500M_projected_wd1e-2_precisions_20261013/llama500M_apollo_fp8_full_wd1e-2_2xC_warmup2000_2gpu_20261013/ckpts/67911/main.pt',
    'AdamW': '/home/jovyan/dimativator/figure6a-fix-decay-downstream-20261013/figure6a_fix_decay_downstream_20261013/adamw_fp8_states_fix_decay_global128/ckpts/75457/main.pt',
}


def delete_authorized() -> dict:
    mount = Path('/workspace-SR006.nfs3')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(mount, flags)
    try:
        for component in TARGET.relative_to(mount).parts[:-1]:
            child = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        before = os.statvfs(directory)
        try:
            info = os.stat(TARGET.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return dict(path=str(TARGET), status='already_absent', before_available_bytes=before.f_bavail*before.f_frsize)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != EXPECTED_SIZE:
            raise RuntimeError('Target does not match the inventoried regular singly linked checkpoint')
        os.unlink(TARGET.name, dir_fd=directory)
        after = os.statvfs(directory)
        # Verify through the same directory descriptor, without following symlinks.
        try:
            os.stat(TARGET.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            absent = True
        else:
            absent = False
        return dict(path=str(TARGET), status='deleted' if absent else 'unexpected_remaining_file',
                    size_bytes=info.st_size, allocated_bytes=info.st_blocks*512,
                    inode=info.st_ino, before_available_bytes=before.f_bavail*before.f_frsize,
                    after_available_bytes=after.f_bavail*after.f_frsize, timestamp=time.time(),
                    other_directory_entries=os.listdir(directory))
    finally:
        os.close(directory)


def inspect_checkpoint(name: str, filename: str) -> dict:
    import torch
    try:
        checkpoint = torch.load(filename, map_location='meta', weights_only=True, mmap=True)
        model = checkpoint.get('model', checkpoint.get('state_dict', {}))
        names = list(model)
        moe_names = [key for key in names if any(part in key.lower() for part in ('experts', 'router', 'block_sparse_moe'))]
        mlp_names = [key for key in names if re.search(r'(mlp|feed_forward)\.(w1|w2|c_fc|c_proj|gate_proj|up_proj|down_proj)\.weight$', key)]
        classification = 'MoE' if moe_names else ('dense' if mlp_names else 'unresolved')
        blocks = {int(match.group(1)) for key in names if (match := re.search(r'(?:h|layers|blocks)\.(\d+)\.', key))}
        return dict(name=name, path=filename, classification=classification, model_tensor_names=len(names),
                    transformer_blocks=len(blocks), expert_or_router_names=moe_names[:10],
                    dense_mlp_examples=[dict(name=key, shape=list(model[key].shape)) for key in mlp_names[:4]],
                    checkpoint_keys=list(checkpoint), iteration=checkpoint.get('itr', checkpoint.get('iter')),
                    deleted=False)
    except Exception as error:
        return dict(name=name, path=filename, classification='unresolved', error=repr(error), deleted=False)


if __name__ == '__main__':
    storage = enforce(OUTPUT)
    result = dict(authorized_instruction='remove Old Slim-NorMuon 500M checkpoint, step 67,911',
                  deletion=delete_authorized(), storage_before=storage, timestamp=time.time())
    target = OUTPUT / 'authorized_slim_checkpoint_cleanup.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    result['checkpoint_architectures'] = [inspect_checkpoint(name, path) for name, path in INSPECT.items()]
    result['storage_after'] = enforce(OUTPUT)
    target.write_text(json.dumps(result, indent=2)+'\n')
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(target), repo_id='DimaTivator/effpretrain_ckpts',
        path_in_repo='mode_connectivity_20m_20261103/'+target.name,
        commit_message='Authorized single-checkpoint deletion and read-only architecture audit')
    print(json.dumps(result), flush=True)
