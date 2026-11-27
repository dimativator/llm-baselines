"""Read a compact snapshot of the matched grid from the shared home volume."""
import json
from pathlib import Path
import time

root = Path('/home/jovyan/dimativator/swd-grid-1m-20261127-r2')
def records(name):
    # Ignore only an unfinished last line during concurrent append.
    raw = (root/name).read_bytes().split(b'\n')[:-1]
    return [json.loads(line) for line in raw if line]
events = records('progress.jsonl')
endpoints = [r for r in events if r['event']=='endpoint']
latest = next(r for r in reversed(events) if r['event']=='train')
disk = {r['path']:[r['timestamp'],r.get('available_bytes')] for r in records('disk.jsonl')}
out = dict(now=time.time(), count=len(endpoints),files=len(list(root.glob('lambda_*.pt'))),train=latest,disk=disk)
if endpoints:
    r=endpoints[-1]
    out['last']=[r['time'],r['coefficient'],r['validation']['loss'],r['ranks']['mean_effective_rank']]
if len(endpoints)>1:
    out['seconds_per_model']=(endpoints[-1]['time']-endpoints[0]['time'])/(len(endpoints)-1)
print('GRID_SNAPSHOT='+json.dumps(out),flush=True)
