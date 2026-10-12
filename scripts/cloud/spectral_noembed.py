"""Finite matched spectral-WD sweep, with logs and in-memory final ranks only."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import rank_profiles as ranks

CAMPAIGN = 'spectral_noembed_20261012'
ROOT = Path(os.environ.get('SPECTRAL_RESULT_ROOT', '/tmp/spectral-noembed-results')) / CAMPAIGN


def run(coef: float, smoke: bool = False, gpu_smoke: bool = False) -> None:
    import main as trainer
    import optim.base as loop
    from models.compress import model_effective_ranks

    label = f'llama124m_spectral_noembed_coef{coef:g}' + ('_gpu_smoke' if gpu_smoke else ('_smoke' if smoke else ''))
    folder = ROOT / label
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    print(f'LOG_STORAGE {folder} free_bytes={free}', flush=True)
    if free < 1024**3:
        raise RuntimeError('Insufficient local log storage: require 1 GiB free')
    ranks.ROOT = folder
    ranks.CAMPAIGN = CAMPAIGN
    metrics = folder / 'metrics.jsonl'
    if metrics.exists():
        metrics.rename(folder / f'metrics_attempt_{time.time_ns()}.jsonl')
    original_init = trainer.wandb.init

    def log(values: dict, *args, **kwargs) -> None:
        numeric = {k: v for k, v in values.items() if isinstance(v, (int, float))}
        numeric['wall_time'] = time.time()
        with metrics.open('a') as stream:
            stream.write(json.dumps(numeric, allow_nan=False) + '\n')
        loop.wandb.run.log(values, *args, **kwargs)

    def init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        loop.wandb.log = log
        return run

    trainer.wandb.init = init
    excluded: set[int] = set()
    get_model = trainer.get_model

    def make_model(config):
        model = get_model(config)
        excluded.update((id(model.transformer.wte.weight), id(model.lm_head.weight)))
        return model

    trainer.get_model = make_model
    original_optimizer = trainer.AdamWSpectralL1Reg

    class NoEmbeddingSpectral(original_optimizer):
        def __init__(self, groups, **kwargs):
            split = []
            for group in groups:
                regularized = [p for p in group['params'] if id(p) not in excluded]
                unregularized = [p for p in group['params'] if id(p) in excluded]
                if regularized:
                    split.append({**group, 'params': regularized})
                if unregularized:
                    split.append({**group, 'params': unregularized, 'spectral_l1_reg_coef': 0.0,
                                  'weight_decay': 0.0})
            groups[:] = split  # Keep the trainer's scheduler group list aligned.
            super().__init__(groups, **kwargs)
            for group in self.param_groups:
                for p in group['params']:
                    if id(p) in excluded:
                        assert group['spectral_l1_reg_coef'] == group['weight_decay'] == 0
            print('EMBEDDING_REGULARIZATION_DISABLED', flush=True)

    trainer.AdamWSpectralL1Reg = NoEmbeddingSpectral
    original_eval = loop.eval_and_log

    def eval_and_rank(tokens, curr_iter, epoch, model, val_reader, type_ctx, backend, cfg, opt, **kwargs):
        original_eval(tokens, curr_iter, epoch, model, val_reader, type_ctx, backend, cfg, opt, **kwargs)
        raw = backend.get_raw_model(model)
        er = model_effective_ranks(raw)
        log({'iter': curr_iter, **er})
        print(f'MEAN_EFFECTIVE_RANK iter={curr_iter} value={er["effective_rank/mean_weighted"]:.6f}', flush=True)
        if not smoke and not gpu_smoke:
            ranks.upload_verified(metrics, f'{ranks.PREFIX}/{CAMPAIGN}/{label}/metrics.jsonl')

    loop.eval_and_log = eval_and_rank
    iterations = 8 if gpu_smoke else (4 if smoke else 19000)
    data = ranks.DATA_ROOT
    if smoke:
        import numpy as np
        data = folder / 'tiny_tokens'
        data.mkdir(exist_ok=True)
        for split in ('train', 'val'):
            np.arange(4096, dtype=np.uint16).tofile(data / (split + '.bin'))
    else:
        ranks.verify_data(data)
    sys.argv = ['main.py', '--experiment_name', label, '--results_base_folder', str(ROOT),
                '--model', 'llama', '--dataset', 'finewebedu', '--tokenized_data_dir', str(data),
                '--opt', 'adamw-spectral-l1-reg', '--weight_decay', '0', '--spectral_l1_reg_coef', str(coef),
                '--lr', '0.005', '--iterations', str(iterations), '--warmup_steps', '2' if smoke or gpu_smoke else '2000',
                '--n_embd', '32' if smoke else '768', '--n_head', '4' if smoke else '12',
                '--n_layer', '1' if smoke else '12', '--batch_size', '1' if smoke else '32',
                '--acc_steps', '1' if smoke else '4', '--sequence_length', '32' if smoke else '1024',
                '--eval_batch_size', '1' if smoke else '64', '--eval_batches', '1' if smoke else '64',
                '--grad_clip', '0.5', '--seed', '0', '--data_seed', '1337', '--scheduler', 'cos',
                '--dropout', '0', '--beta1', '0.9', '--beta2', '0.95', '--finewebedu_max_files', '5',
                '--eval_interval', '2' if smoke or gpu_smoke else '137', '--log_interval', '1' if smoke or gpu_smoke else '4',
                '--latest_ckpt_interval', '0', '--permanent_ckpt_interval', '0', '--effective_rank_interval', '0',
                '--wandb', '--wandb_project', 'spectral-noembed', '--wandb_entity', 'andrey',
                '--hf_checkpoint_repo', ranks.REPO]
    if smoke:
        sys.argv += ['--device', 'cpu', '--dtype', 'float32']
    args, parser = trainer.get_args()

    def final_measurement(model, opt, scheduler, itr, repo_id, path_in_repo):
        raw = model.module if hasattr(model, 'module') else model
        payload = ranks.measure(raw.state_dict(), vars(args).copy(), label, 'spectral_wd_noembed', coef,
                                'in-memory final weights, no checkpoint', os.environ.get('RANK_CODE_REVISION', ''),
                                'cpu' if smoke else 'cuda:0')
        payload['training_data_sha256'] = ranks.DATA_HASHES if not smoke else {}
        payload['excluded_regularization'] = ['transformer.wte.weight', 'lm_head.weight']
        blocks = defaultdict(list)
        for row in payload['rows']:
            if row['block_index'] is not None:
                blocks[row['block_index']].append(row)
        payload['block_summary'] = [dict(block_index=b,
            mean_effective_rank=sum(r['effective_rank'] for r in rows) / len(rows),
            mean_effective_rank_percent=sum(r['effective_rank_percent'] for r in rows) / len(rows),
            mean_weighted_effective_rank=sum(r['effective_rank']*r['full_rank'] for r in rows)/sum(r['full_rank'] for r in rows))
            for b, rows in sorted(blocks.items())]
        ranks.persist(payload, publish=False)
        assert not list(folder.rglob('*.pt')), 'Checkpoint saving must remain disabled'
        if not smoke and not gpu_smoke:
            for path in (metrics, folder / 'summary.json', folder / (label + '.json'),
                         folder / (label + '.csv'), folder / (label + '_grouped.csv')):
                ranks.upload_verified(path, f'{ranks.PREFIX}/{CAMPAIGN}/{label}/{path.name}')
        ranks.write_json(folder / 'SUCCESS.json', dict(status='complete', iteration=itr, checkpoints_saved=0))
        if not smoke:
            ranks.upload_verified(folder / 'SUCCESS.json', f'{ranks.PREFIX}/{CAMPAIGN}/{label}/SUCCESS.json')
        print('NO_CHECKPOINTS_VERIFIED', flush=True)

    trainer.upload_checkpoint_to_hf = final_measurement
    trainer.main(args, parser)
    loop.wandb.finish()
    print('SWEEP_RUN_COMPLETE ' + label, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coef', type=float, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--gpu-smoke', action='store_true')
    options = parser.parse_args()
    torch.set_num_threads(4)
    run(options.coef, options.smoke, options.gpu_smoke)
