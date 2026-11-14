"""Read only this campaign's minimal metrics from NFS3."""
import json
import base64
import gzip
import hashlib
import os
from pathlib import Path
import time

from polar_grid_guard import ROOT, BOOTSTRAP, snapshot


if __name__ == '__main__':
    guard = snapshot()
    print('DISK_GUARD=' + json.dumps(guard), flush=True)
    if not guard['ok']:
        raise RuntimeError('Storage telemetry unavailable or below reserve')
    result = dict(timestamp=time.time(), root=str(ROOT), lanes=[], smokes=[], checkpoint_files=[])
    for folder in sorted(ROOT.glob('lane*')):
        row = dict(lane=folder.name)
        for name in ('status.json', 'results.json', 'failures.json', 'plan.json'):
            path = folder / name
            if path.is_file():
                row[name] = json.loads(path.read_text())
        status = row.get('status.json', {})
        run = status.get('run', {}).get('name')
        if run and run != 'prepare_tokens':
            output = ROOT / 'runs' / run
            metrics = output / 'metrics.jsonl'
            row['recent_metrics'] = [json.loads(line) for line in metrics.read_text().splitlines()[-4:]] if metrics.exists() else []
            log = output / 'train.log'
            row['recent_log'] = log.read_text()[-2500:] if log.exists() else ''
        result['lanes'].append(row)
    for folder in sorted(ROOT.glob('cpu_smoke*')):
        status = folder / 'lane0' / 'status.json'
        if status.exists():
            result['smokes'].append(dict(path=str(folder), status=json.loads(status.read_text())))
    for suffix in ('*.pt', '*.safetensors'):
        result['checkpoint_files'] += [str(path) for path in ROOT.rglob(suffix)]
    bootstrap = Path(BOOTSTRAP) / 'lm-mpi-job-47803121-7278-46f1-a1a3-70d60e423b1c'
    result['first_job_bootstrap'] = []
    if bootstrap.exists():
        for parent, folders, files in os.walk(bootstrap, followlinks=False):
            if len(Path(parent).relative_to(bootstrap).parts) >= 2:
                folders[:] = []
            for name in files[:10]:
                path = Path(parent) / name
                if path.is_symlink() or not path.is_file():
                    continue
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 4000))
                    tail = stream.read(4000).decode(errors='replace')
                result['first_job_bootstrap'].append(dict(path=str(path), tail=tail))
            if len(result['first_job_bootstrap']) >= 20:
                break
    payload = json.dumps(result, allow_nan=False).encode()
    encoded = base64.b64encode(gzip.compress(payload)).decode()
    chunks = [encoded[index:index+1200] for index in range(0, len(encoded), 1200)]
    print('POLAR_GRID_INSPECTION_META=' + json.dumps(dict(chunks=len(chunks), sha256=hashlib.sha256(payload).hexdigest())), flush=True)
    for index, chunk in enumerate(chunks):
        print(f'POLAR_GRID_INSPECTION_CHUNK={index}:{chunk}', flush=True)
