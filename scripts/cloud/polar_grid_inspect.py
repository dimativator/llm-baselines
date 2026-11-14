"""Read only this campaign's minimal metrics from NFS3."""
import json
from pathlib import Path
import time

from polar_grid_guard import ROOT, snapshot


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
    print('POLAR_GRID_INSPECTION=' + json.dumps(result, allow_nan=False), flush=True)
