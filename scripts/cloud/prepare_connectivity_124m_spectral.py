"""Verify storage, corpus, model and two-process CPU smoke for the new campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from connectivity_campaign import ROOT, prepare, preflight, publish, write_json
from connectivity_guard import enforce

if __name__ == '__main__':
    enforce(ROOT, pending=2_200_000_000)
    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='connectivity-spectral-smoke-') as folder:
        env = dict(os.environ, CONNECTIVITY_ROOT=folder, WANDB_MODE='disabled')
        try:
            subprocess.run([sys.executable, str(Path(__file__).with_name('connectivity_campaign.py')), '--smoke'],
                           env=env, check=True)
        except subprocess.CalledProcessError:
            failures = {str(p.relative_to(folder)): p.read_text()[-12000:]
                        for p in Path(folder).glob('*/worker.log')}
            write_json(ROOT/'smoke_failure.json', failures)
            publish([ROOT/'smoke_failure.json'], verify=True)
            raise
        proof = json.loads((Path(folder)/'COMPLETE.json').read_text())
        assert proof['checkpoints_remaining'] == 0 and proof['total_paths'] == 6
        proof['synthetic_cpu_only'] = True
        proof['world_size'] = 2
        write_json(ROOT/'smoke_verified.json', proof)
    prepare(False)
    import torch
    from connectivity_experiment import Llama, config
    torch.set_num_threads(4)
    model = Llama(config())
    count = sum(p.numel() for p in model.parameters())
    assert count == 123587328 and model.transformer.wte.weight is model.lm_head.weight
    write_json(ROOT/'model_verified.json', dict(parameters=count, config=vars(model.config),
        tied_embeddings=True, timestamp=time.time()))
    publish([ROOT/'model_verified.json', ROOT/'smoke_verified.json'], verify=True)
    preflight()
