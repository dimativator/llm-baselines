"""Read-only storage inventory, restricted to known roots and excluding NFS2."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import stat
import time

from connectivity_guard import enforce

OUTPUT = Path('/workspace-SR006.nfs3/dimativator/mode-connectivity-20m-20261103')
ROOTS = (
    ('user_nfs3', '/workspace-SR006.nfs3/dimativator'),
    ('user_home', '/home/jovyan/dimativator'),
    ('known_training_data', '/home/jovyan/finewebedu_h200'),
    ('shared_pip_cache', '/home/jovyan/.cache/pip'),
    ('shared_hf_cache', '/home/jovyan/.cache/huggingface'),
)


def inventory(label: str, name: str) -> dict:
    root = Path(name)
    result = dict(label=label, root=name, started=time.time(), directories=[], largest_files=[], errors=[])
    try:
        root_info = root.lstat()
    except OSError as error:
        result['errors'].append(str(error))
        return result
    if not stat.S_ISDIR(root_info.st_mode):
        result['errors'].append('Root is not a direct directory; symlink roots are not followed')
        return result
    device = root_info.st_dev
    totals = defaultdict(lambda: dict(allocated_bytes=0, apparent_bytes=0, files=0))
    categories = defaultdict(int)
    largest = []
    seen = set()
    pending = [root]
    last_guard = time.monotonic()
    while pending:
        if time.monotonic() - last_guard > 30:
            enforce(OUTPUT)
            last_guard = time.monotonic()
        folder = pending.pop()
        try:
            entries = os.scandir(folder)
            with entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    if info.st_dev != device or stat.S_ISLNK(info.st_mode):
                        continue
                    path = Path(entry.path)
                    if 'nfs2' in str(path).lower():
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(path)
                        continue
                    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) in seen:
                        continue
                    seen.add((info.st_dev, info.st_ino))
                    parts = path.relative_to(root).parts
                    allocated = info.st_blocks * 512
                    for depth in range(min(len(parts)-1, 2) + 1):
                        key = '/'.join(parts[:depth]) or '.'
                        totals[key]['allocated_bytes'] += allocated
                        totals[key]['apparent_bytes'] += info.st_size
                        totals[key]['files'] += 1
                    suffix = path.suffix.lower()
                    categories[suffix or '(no extension)'] += allocated
                    largest.append(dict(path=str(path), allocated_bytes=allocated, size_bytes=info.st_size,
                                        modified=info.st_mtime, hardlinks=info.st_nlink))
                    if len(largest) > 100:
                        largest.sort(key=lambda row: row['allocated_bytes'], reverse=True)
                        del largest[60:]
        except OSError as error:
            if len(result['errors']) < 30:
                result['errors'].append(dict(path=str(folder), error=str(error)))
    result['directories'] = [dict(path=str(root / key) if key != '.' else name, **values)
                             for key, values in sorted(totals.items(), key=lambda item:item[1]['allocated_bytes'], reverse=True)]
    result['largest_files'] = sorted(largest, key=lambda row:row['allocated_bytes'], reverse=True)[:40]
    result['extension_allocated_bytes'] = dict(sorted(categories.items(), key=lambda item:item[1], reverse=True))
    result['completed'] = time.time()
    return result


if __name__ == '__main__':
    storage = enforce(OUTPUT, pending=1_000_000)
    results = [inventory(label, path) for label, path in ROOTS]
    record = dict(timestamp=time.time(), storage_before=storage, storage_after=enforce(OUTPUT),
                  scope='Read-only, no symlink traversal, no NFS2, no files deleted', roots=results)
    target = OUTPUT / ('storage_inventory_' + str(int(record['timestamp'])) + '.json')
    target.write_text(json.dumps(record, indent=2) + '\n')
    from huggingface_hub import HfApi
    commit = HfApi().upload_file(path_or_fileobj=str(target), repo_id='DimaTivator/effpretrain_ckpts',
        path_in_repo='mode_connectivity_20m_20261103/' + target.name,
        commit_message='Read-only storage inventory, NFS2 excluded')
    print(json.dumps(dict(result=target.name, revision=commit.oid,
        summaries=[dict(label=r['label'], directories=r['directories'][:15], errors=r['errors']) for r in results])), flush=True)
