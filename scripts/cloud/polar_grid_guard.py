"""Bounded, exact-byte Cloud filesystem measurements for the checkpoint-free grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RESERVE = 5_000_000_000
ROOT = Path('/home/jovyan/shares/SR006.nfs3/dimativator-polar-grid-124m-20261114')
BOOTSTRAP = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
PATHS = ('/', '/tmp', '/home/jovyan', '/home/jovyan/shares/SR006.nfs2', BOOTSTRAP, str(ROOT))
DATA_CANDIDATES = (
    '/home/jovyan/finewebedu_h200/tokenized',
    '/workspace-SR006.nfs3/dimativator/finewebedu_h200/tokenized',
    '/home/jovyan/shares/SR006.nfs3/dimativator/finewebedu_h200/tokenized',
)
PROBE = '''
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1])
if 'SR006.nfs2' in str(p) and not p.exists():
    raise OSError('Required NFS2 mount unavailable')
while not p.exists() and p != p.parent: p=p.parent
p=p.resolve(strict=True)
s=os.statvfs(p)
m=p
while not m.is_mount() and m != m.parent: m=m.parent
print(json.dumps(dict(path=sys.argv[1],measured_path=str(p),mount=str(m),
    device=os.stat(p).st_dev,available_bytes=s.f_bavail*s.f_frsize,timestamp=time.time())))
'''


def snapshot(pending: int = 10_000_000) -> dict:
    rows = []
    for path in PATHS:
        process = subprocess.Popen([sys.executable, '-c', PROBE, path],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            output, error = process.communicate(timeout=5)
            if process.returncode:
                raise OSError(error[-500:])
            row = json.loads(output)
            row['ok'] = row['available_bytes'] > RESERVE + pending
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            process.kill()
            for stream in (process.stdout, process.stderr):
                stream.close()
            row = dict(path=path, timestamp=time.time(), error=str(error), ok=False)
        rows.append(row)
    return dict(timestamp=time.time(), reserve_bytes=RESERVE, pending_bytes=pending,
                paths=rows, ok=all(row['ok'] for row in rows))


def preflight() -> None:
    record = snapshot(pending=100_000_000)
    record['data_candidates'] = {}
    for directory in DATA_CANDIDATES:
        try:
            result = subprocess.run([sys.executable, '-c',
                'import json,os,sys; print(json.dumps({n:os.path.getsize(sys.argv[1]+"/"+n) for n in ("train.bin","val.bin")}))',
                directory], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                record['data_candidates'][directory] = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, ValueError):
            pass
    print('POLAR_GRID_PREFLIGHT=' + json.dumps(record), flush=True)


if __name__ == '__main__':
    preflight()
