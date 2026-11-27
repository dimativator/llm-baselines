"""Remove only the ten named stage files, and verify the campaign checkpoint count."""
import json
import os
from pathlib import Path
import stat
import time

home = Path('/home/jovyan')
root = home/'dimativator/subspace-30m-20261126'
assert os.path.ismount(home) or os.stat(home).st_dev != os.stat('/').st_dev
assert root.resolve() == home.resolve()/'dimativator/subspace-30m-20261126'
labels = ['l2','swd01','swd03','swd05','swd07','swd1']
deleted = {}
for a,b in zip(labels, labels[1:]):
    directory = root/(a+'__'+b)
    assert not directory.is_symlink()
    for label in (a,b):
        f = directory/(label+'.pt')
        try:
            info = f.lstat()
        except FileNotFoundError:
            continue
        assert stat.S_ISREG(info.st_mode)
        f.unlink()
        deleted[str(f.relative_to(root))] = info.st_size
remaining = sorted(str(p.relative_to(root)) for p in root.rglob('*.pt'))
assert not remaining, remaining
receipt = dict(timestamp=time.time(), deleted=deleted, remaining=remaining)
print(json.dumps(receipt), flush=True)
if root.is_dir():
    (root/'cleanup-verified.json').write_text(json.dumps(receipt))
