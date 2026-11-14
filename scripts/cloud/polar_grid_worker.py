"""Ten independent single-GPU queues for the approved 102-run AdamW grid."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import queue
import signal
import subprocess
import sys
import time
import threading

from polar_grid_guard import BOOTSTRAP, ROOT, snapshot

DATA = Path('/tmp/polar-grid-tokens')


def grid() -> list[dict]:
    runs = []
    for method, values in (
        ('l2', (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
        ('spectral', (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)),
        ('zero_wd', (0.0,)),
    ):
        for lr in (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01):
            for wd in values:
                runs.append(dict(method=method, lr=lr, wd=wd, seed=0))
    assert len(runs) == 102
    for index, run in enumerate(runs):
        run['name'] = f'{index:03d}_{run["method"]}_lr{run["lr"]:g}_wd{run["wd"]:g}'
    random.Random(0).shuffle(runs)
    return runs


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def command(run: dict, root: Path, smoke: bool) -> list[str]:
    arguments = [sys.executable, 'scripts/cloud/polar_grid_train.py',
        '--experiment_name', run['name'], '--results_base_folder', str(root / 'runs'),
        '--device', 'cpu' if smoke else 'cuda:0', '--model', 'llama',
        '--dataset', 'finewebedu', '--datasets_dir', str(DATA),
        '--tokenized_data_dir', str(root / 'smoke_tokens') if smoke else str(DATA),
        '--require_tokenized_data', '--finewebedu_max_files', '5',
        '--opt', 'adamw-spectral-l1-reg' if run['method'] == 'spectral' else 'adamw',
        '--lr', str(run['lr']), '--weight_decay', str(0.0 if run['method'] == 'spectral' else run['wd']),
        '--iterations', '6' if smoke else '19000', '--warmup_steps', '2' if smoke else '2000',
        '--n_embd', '32' if smoke else '768', '--n_head', '2' if smoke else '12',
        '--n_layer', '1' if smoke else '12', '--sequence_length', '16' if smoke else '1024',
        '--batch_size', '1' if smoke else '16', '--acc_steps', '2' if smoke else '8',
        '--grad_clip', '0.5', '--seed', '0', '--data_seed', '1337',
        '--scheduler', 'cos', '--dropout', '0', '--beta1', '0.9', '--beta2', '0.95',
        '--latest_ckpt_interval', '0', '--permanent_ckpt_interval', '0',
        '--log_interval', '1' if smoke else '100', '--eval_interval', '6' if smoke else '500',
        '--eval_batches', '2' if smoke else '64',
        '--effective_rank_interval', '6' if smoke else '1000', '--minimal_metrics']
    if run['method'] == 'spectral':
        arguments += ['--spectral_l1_reg_coef', str(run['wd']), '--spectral_l1_svt_interval', '0',
                      '--spectral_polar_express', '--spectral_decay_groups_only']
    if smoke:
        arguments += ['--vocab_size', '128']
    return arguments


def stop(process: subprocess.Popen) -> dict:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    return dict(pid=process.pid, returncode=process.returncode)


def run_guarded(process: subprocess.Popen, lane_dir: Path, run: dict, completed: list) -> None:
    """A separate watchdog remains live even if the coordinator blocks on NFS writes."""
    done = threading.Event()
    failures = []
    messages = queue.Queue(maxsize=8)

    def emit(message: str) -> None:
        try:
            messages.put_nowait(message)
        except queue.Full:
            pass

    def write_messages() -> None:
        while not done.is_set() or not messages.empty():
            try:
                print(messages.get(timeout=1), flush=True)
            except queue.Empty:
                continue

    writer = threading.Thread(target=write_messages, daemon=True)
    writer.start()

    def watch() -> None:
        while not done.is_set():
            try:
                guard = snapshot()
            except Exception as error:
                guard = dict(ok=False, stop_requested=False, error=repr(error), timestamp=time.time())
            if not guard['ok'] or guard['stop_requested']:
                failures.append(guard)
                stopped = stop(process)
                emit('КРИТИЧЕСКИ МАЛО МЕСТА НА CLOUD.RU ИЛИ НЕДОСТУПНА ТЕЛЕМЕТРИЯ. '
                     'ОТСЛЕЖИВАЕМЫЙ ПРОЦЕСС ОСТАНОВЛЕН. ' + json.dumps(stopped))
                emit('DISK_GUARD=' + json.dumps(guard))
                # Stop the exact owned child before touching any shared filesystem.
                save(ROOT / 'STOP.json', dict(guard=guard, stopped=stopped))
                return
            emit('DISK_GUARD=' + json.dumps(guard))
            done.wait(30)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        save(lane_dir / 'status.json', dict(state='running', run=run, pid=process.pid,
                                           completed=completed, timestamp=time.time()))
        print('RUN_START=' + json.dumps(dict(**run, pid=process.pid)), flush=True)
        process.wait()
    finally:
        done.set()
        if process.poll() is None:
            print('STOPPED=' + json.dumps(stop(process)), flush=True)
        thread.join(timeout=1)
        writer.join(timeout=1)
    if failures:
        raise RuntimeError('Storage guard stopped the lane; no checkpoints exist to delete')


def worker(lane: int, smoke: bool = False) -> None:
    root = ROOT / f'cpu_smoke_{int(time.time())}' if smoke else ROOT
    guard = snapshot(pending=100_000_000)
    print('DISK_GUARD=' + json.dumps(guard), flush=True)
    if not guard['ok']:
        raise RuntimeError('Storage reserve or telemetry unavailable')
    root.mkdir(parents=True, exist_ok=True)
    lane_dir = root / f'lane{lane}'
    lane_dir.mkdir(exist_ok=True)
    with (lane_dir / 'STARTED.json').open('x') as stream:
        json.dump(dict(lane=lane, pid=os.getpid(), timestamp=time.time()), stream)
    if smoke:
        import numpy as np
        tokens = root / 'smoke_tokens'
        tokens.mkdir(exist_ok=True)
        for split in ('train', 'val'):
            (np.arange(256, dtype=np.uint16) % 128).tofile(tokens / f'{split}.bin')
        runs = [dict(name=f'smoke_{method}', method=method, lr=0.001, wd=0.0 if method == 'zero_wd' else 0.01, seed=0)
                for method in ('l2', 'spectral', 'zero_wd')]
    else:
        runs = grid()[lane::10]
        if not all((DATA / name).is_file() for name in ('train.bin', 'val.bin')):
            guard = snapshot(pending=100_000_000, temporary_bytes=7_536_512_038)
            if not guard['ok'] or guard['stop_requested']:
                raise RuntimeError('Insufficient space for temporary training tokens')
            process = subprocess.Popen([sys.executable, 'scripts/cloud/polar_grid_data.py'], start_new_session=True)
            run_guarded(process, lane_dir, dict(name='prepare_tokens'), [])
            if process.returncode:
                raise RuntimeError('Token download or hash verification failed')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(lane_dir / 'plan.json', dict(lane=lane, runs=runs, revision=revision,
                                    checkpoints=False, bootstrap_logs=BOOTSTRAP))
    completed = []
    failed = []
    for run in runs:
        guard = snapshot(pending=100_000_000)
        print('DISK_GUARD=' + json.dumps(guard), flush=True)
        if not guard['ok'] or guard['stop_requested']:
            raise RuntimeError('Storage guard prevents the next run')
        output = root / 'runs' / run['name']
        if output.exists():
            raise RuntimeError('Refusing to overwrite an existing run: ' + str(output))
        output.mkdir(parents=True)
        args = command(run, root, smoke)
        save(output / 'config.json', dict(**run, command=args, revision=revision, checkpoints=False))
        with (output / 'train.log').open('x') as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            run_guarded(process, lane_dir, run, completed)
        checkpoints = list(output.rglob('*.pt')) + list(output.rglob('*.safetensors'))
        if checkpoints:
            raise RuntimeError('Checkpoint prohibition violated: ' + str(checkpoints))
        if process.returncode:
            tail = (output / 'train.log').read_text()[-10000:]
            print(tail, flush=True)
            failed.append(dict(**run, returncode=process.returncode, reason=tail[-1500:]))
            save(lane_dir / 'failures.json', dict(failed=failed))
            numeric = any(word in tail.lower() for word in ('non-finite', 'nonfinite', 'nan', 'floatingpointerror', 'failed to converge'))
            infrastructure = any(word in tail.lower() for word in ('out of memory', 'modulenotfounderror', 'importerror', 'filenotfounderror', 'checkpoint writes are prohibited'))
            if infrastructure or not numeric:
                save(ROOT / 'STOP.json', dict(reason='Infrastructure failure', run=run, returncode=process.returncode, timestamp=time.time()))
                raise RuntimeError('Infrastructure failure; stopping this attempt before further configurations')
            continue
        final = json.loads((output / 'final.json').read_text())
        expected = 6 if smoke else 19000
        if final['step'] != expected or not final['final'] or not math.isfinite(final['val_loss']):
            failed.append(dict(**run, reason='Invalid final validation result'))
            save(lane_dir / 'failures.json', dict(failed=failed))
            continue
        completed.append(dict(**run, **final))
        save(lane_dir / 'results.json', dict(completed=completed, checkpoints=0))
        print('RUN_COMPLETE=' + json.dumps(completed[-1]), flush=True)
    save(lane_dir / 'status.json', dict(state='completed' if not failed else 'completed_with_failures', completed=completed, failed=failed, timestamp=time.time()))
    if smoke and (failed or len(completed) != 3):
        raise RuntimeError('CPU smoke verification failed')
    print('LANE_COMPLETE=' + json.dumps(dict(lane=lane, completed=len(completed), failed=len(failed), checkpoints=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=int, choices=range(10), default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    worker(args.lane, args.smoke)
