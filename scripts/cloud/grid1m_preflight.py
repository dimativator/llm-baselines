import json
import os
from pathlib import Path
import time

home=Path('/home/jovyan')
data=Path('/workspace-SR006.nfs3/dimativator/spectral-wd-matched-tokenized')
readings={}
for path,write in [(home,480000000),(home/'mlsub-logs',480000000),(Path('/'),70000000),
                   (Path('/tmp'),70000000),(Path('/workspace-SR006.nfs2'),0),(data,0)]:
    st=os.statvfs(path)
    readings[str(path)]=st.f_bavail*st.f_frsize
    assert readings[str(path)]>5000000000+write, (path,readings)
for name in ('train.bin','val.bin'):
    with (data/name).open('rb') as stream:
        assert len(stream.read(2))==2
record=dict(time=time.time(),available=readings)
print(json.dumps(record),flush=True)
(home/'dimativator/grid1m_preflight_20261127.json').write_text(json.dumps(record))
