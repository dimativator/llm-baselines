"""Publish current log-only artifacts from shared storage; never touch weights."""
from pathlib import Path
import json
import rank_profiles as ranks

root = Path('/workspace-SR006.nfs3/dimativator/results/spectral_noembed_20261012')
ranks.CAMPAIGN = 'spectral_noembed_20261012'
for folder in sorted(root.glob('llama124m_spectral_noembed_coef*')):
    if folder.name.endswith('_smoke'):
        print('SMOKE', folder.name, (folder / 'SUCCESS.json').exists(), flush=True)
        continue
    metrics = folder / 'metrics.jsonl'
    if metrics.exists():
        records = []
        for line in metrics.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass  # Last append may be in flight.
        target = folder / 'metrics_snapshot.jsonl'
        target.write_text(''.join(json.dumps(row) + '\n' for row in records))
        ranks.upload_verified(target, f'{ranks.PREFIX}/{ranks.CAMPAIGN}/{folder.name}/metrics.jsonl')
        latest = max((r.get('iter', 0) for r in records), default=0)
        print('PROGRESS', folder.name, latest, 'complete=' + str((folder / 'SUCCESS.json').exists()), flush=True)
    for name in ('SUCCESS.json', 'summary.json', folder.name + '.json', folder.name + '.csv', folder.name + '_grouped.csv'):
        path = folder / name
        if path.exists():
            ranks.upload_verified(path, f'{ranks.PREFIX}/{ranks.CAMPAIGN}/{folder.name}/{name}')
    print('CHECKPOINT_COUNT', folder.name, len(list(folder.rglob('*.pt'))), flush=True)
