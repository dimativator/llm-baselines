import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--pair', type=int, choices=range(5), required=True)
pair = parser.parse_args().pair
meta = json.loads(Path(__file__).with_name('angles30_manifest.json').read_text())
labels = meta['labels'][pair:pair+2]
home = Path('/home/jovyan')
root = home/'dimativator/subspace-30m-20261126'/'__'.join(labels)
copies = [root/(label+'.pt') for label in labels]
lock = threading.RLock()
done = threading.Event()

def cleanup():
    with lock:
        for p in copies:
            if p.is_file() and not p.is_symlink():
                p.unlink()
        print('CLEANUP', [p.name for p in copies if p.exists()], flush=True)

def guard(pending=0):
    readings = {}
    for path in [home, home/'mlsub-logs', Path('/workspace-SR006.nfs2'), Path('/'), Path('/tmp')]:
        fs = os.statvfs(path)
        readings[str(path)] = fs.f_bavail*fs.f_frsize
    print('STORAGE', time.time(), readings, flush=True)
    if min(readings.values()) <= 5_000_000_000 or readings[str(home)] <= 5_000_000_000+pending:
        raise RuntimeError('STORAGE RESERVE')

def monitor():
    while not done.wait(30):
        try:
            guard()
        except Exception as exc:
            print('DISK GUARD STOP', repr(exc), flush=True)
            cleanup()
            os._exit(74)

def load(label, p):
    guard(1_202_000_000)
    req = urllib.request.Request(os.environ['RELAY_URL']+'/'+p.name,
                                 headers={'Authorization':'Bearer '+os.environ['RELAY_KEY']})
    with urllib.request.urlopen(req, timeout=90) as response, p.open('xb') as f:
        size = int(response.headers['Content-Length'])
        assert 0 < size <= 120_140_000
        total = 0
        while block := response.read(1024*1024):
            total += len(block)
            if total > size:
                raise ValueError('Unexpected checkpoint length')
            f.write(block)
        assert total == size
    state = torch.load(p, map_location='cpu', weights_only=True)['model']
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    assert digest.hexdigest() == meta['hashes'][label], label
    return state

def analyze(a,b):
    ua,sa,va = torch.linalg.svd(a,full_matrices=False)
    ub,sb,vb = torch.linalg.svd(b,full_matrices=False)
    sm = torch.linalg.svdvals((a+b)/2)
    res = {}
    for k in (16,32,64,128):
        cs = [torch.linalg.svdvals(ua[:,:k].T @ ub[:,:k]),
              torch.linalg.svdvals(va[:k] @ vb[:k].T)]
        res[str(k)] = dict(
            mean_angles_deg=[c.clamp(0,1).acos().rad2deg().mean().item() for c in cs],
            mean_sin2=[(1-c.clamp(0,1).square()).mean().item() for c in cs],
            tail_mass=[(s[k:].sum()/s.sum()).item() for s in (sa,sm,sb)])
    return res

guard(1_202_000_000)
root.mkdir(parents=True,exist_ok=False)
threading.Thread(target=monitor,daemon=True).start()
try:
    states = [load(label,p) for label,p in zip(labels,copies)]
    result = {}
    assert torch.cuda.device_count() == 1
    for name in meta['shapes']:
        result[name] = analyze(*(state[name].float().cuda() for state in states))
        print('MATRIX', json.dumps([name,result[name]],separators=(',',':')), flush=True)
    (root/'results.json').write_text(json.dumps(result,allow_nan=False))
    print('COMPLETE',len(result),'peak_gpu_bytes',torch.cuda.max_memory_allocated(),flush=True)
finally:
    done.set()
    cleanup()
