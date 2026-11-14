"""Reuse the recorded, hash-verified FineWeb-Edu tokens in ephemeral local storage."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path('/tmp/polar-grid-tokens')
REVISION = '88651c8d873261b060667ad8b6780bc84f7053bd'
HASHES = {'train.bin': '5a4c532cf1a142834f918f509fc678c178d8edc0df70eaffecc93a955b2f98af',
          'val.bin': 'd3025934e78388200ace8ab058eef34a0dd5a0a3e961d08d637665f436c4996e'}
SIZES = {'train.bin': 7535855100, 'val.bin': 656938}


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        stat = os.statvfs(ROOT)
        if stat.f_bavail * stat.f_frsize <= 5_100_000_000 + SIZES[name]:
            raise RuntimeError('Insufficient exact-byte reserve for token download')
        path = Path(hf_hub_download('DimaTivator/spectral-wd-matched-tokenized', name,
                    repo_type='dataset', revision=REVISION, local_dir=ROOT))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != SIZES[name] or digest.hexdigest() != expected:
            raise RuntimeError('Token data checksum mismatch: ' + name)
        print('DATA_VERIFIED=' + json.dumps(dict(file=name, bytes=SIZES[name], sha256=expected)), flush=True)
