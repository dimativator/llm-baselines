"""Read compact complete curves, baseline, and fresh persisted worker guards."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['POLAR_GRID_ROOT'] = '/home/jovyan/shares/SR006.nfs3/dimativator-schatten4-124m-20261116'
from polar_grid_guard import ROOT, snapshot

BASELINE = Path('/home/jovyan/shares/SR006.nfs3/dimativator-polar-grid-124m-20261114-b16/runs/021_l2_lr0.002_wd0.3')


def read_run(folder: Path) -> dict:
    row = dict(name=folder.name)
    for name in ('config.json', 'final.json'):
        if (folder / name).is_file():
            row[name] = json.loads((folder / name).read_text())
    metrics = folder / 'metrics.jsonl'
    row['metrics'] = []
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            record = json.loads(line)
            keys = ('step', 'train_loss', 'val_loss', 'final', 'effective_rank/mean_weighted')
            small = {key: record[key] for key in keys if key in record}
            if len(small) > 1:
                row['metrics'].append(small)
    log = folder / 'train.log'
    row['recent_log'] = log.read_text()[-2500:] if log.is_file() else ''
    return row


if __name__ == '__main__':
    guard = snapshot(pending=100_000_000)
    print('DISK_GUARD=' + json.dumps(guard), flush=True)
    result = dict(timestamp=time.time(), guard=guard, root=str(ROOT), lanes=[], runs=[], smokes=[],
                  baseline=read_run(BASELINE), checkpoint_files=[])
    for folder in sorted(ROOT.glob('lane*')):
        row = dict(lane=folder.name)
        for name in ('status.json', 'results.json', 'failures.json', 'plan.json', 'latest-disk.json'):
            if (folder / name).is_file():
                row[name] = json.loads((folder / name).read_text())
        result['lanes'].append(row)
    for folder in sorted((ROOT / 'runs').glob('*')):
        if folder.is_dir():
            result['runs'].append(read_run(folder))
    for folder in sorted(ROOT.glob('cpu_smoke*')):
        for status in folder.glob('lane*/status.json'):
            result['smokes'].append(dict(path=str(status), status=json.loads(status.read_text())))
    for suffix in ('*.pt', '*.pth', '*.ckpt', '*.safetensors'):
        result['checkpoint_files'] += [str(path) for path in ROOT.rglob(suffix)]
    payload = json.dumps(result, allow_nan=False).encode()
    encoded = base64.b64encode(gzip.compress(payload)).decode()
    chunks = [encoded[i:i+1200] for i in range(0, len(encoded), 1200)]
    print('POLAR_GRID_INSPECTION_META=' + json.dumps(dict(chunks=len(chunks), sha256=hashlib.sha256(payload).hexdigest())), flush=True)
    for index, chunk in enumerate(chunks):
        print(f'POLAR_GRID_INSPECTION_CHUNK={index}:{chunk}', flush=True)
