"""Read-only confirmation of production progress when the scheduler logs are rate-limited."""
import json
from pathlib import Path
import time

root=Path('/home/jovyan/dimativator/swd-grid-1m-20261127')
for attempt in range(120):
    if (root/'progress.jsonl').is_file():
        for line in (root/'progress.jsonl').read_text().splitlines():
            try:
                row=json.loads(line)
            except ValueError:
                continue
            if row['event']=='train' and row['index']==1 and row['step']>=1:
                print(json.dumps(row))
                raise SystemExit(0)
    time.sleep(10)
raise SystemExit('No optimizer-step record found during this observation')
