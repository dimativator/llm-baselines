"""One process, one GPU, 100 matched endpoint trainings and path diagnostics."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import time

import numpy as np
import requests
import tiktoken
import torch

import connectivity_experiment as e
from connectivity_guard import _probe

HOME = Path('/home/jovyan')
ROOT = HOME/'dimativator/swd-grid-1m-20261127'
DATA_URL = 'https://huggingface.co/datasets/DimaTivator/spectral-wd-matched-tokenized/resolve/88651c8d873261b060667ad8b6780bc84f7053bd'
COUNT = 1058016
BATCH, LENGTH = 64, 256
STEPS = math.ceil(20 * COUNT / (BATCH * LENGTH))
TOKENS = STEPS * BATCH * LENGTH
RESERVE = 5_000_000_000
LOCK = threading.RLock()


def note(event, **fields):
    row = dict(event=event, time=time.time(), **fields)
    print(json.dumps(row, allow_nan=False), flush=True)
    if ROOT.is_dir():
        with (ROOT/'progress.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')


def guard(home_write=0, tmp_write=0):
    with LOCK:
        for p, reserved in ((HOME, home_write), (HOME/'mlsub-logs',home_write),
                            (Path('/tmp'),tmp_write), (Path('/'),tmp_write),
                            (Path('/workspace-SR006.nfs2'),0)):
            row = _probe(str(p))
            free = row.get('available_bytes')
            if free is None or free <= RESERVE+reserved:
                print('DISK_STOP',json.dumps(row),flush=True)
                if free is not None and free < RESERVE and ROOT.is_dir() and ROOT.resolve()==HOME.resolve()/'dimativator/swd-grid-1m-20261127' and ROOT.stat().st_dev==p.stat().st_dev:
                    files=[ROOT/f'lambda_{i:03d}.{ext}' for i in range(1,101) for ext in ('pt','part')]
                    files=[f for f in files if not f.is_symlink() and f.is_file() and f.stat().st_nlink==1]
                    for f in sorted(files,key=lambda f:f.stat().st_mtime_ns):
                        f.unlink()
                        after=_probe(str(p))
                        print('EMERGENCY_DELETE',str(f),json.dumps(after),flush=True)
                        if after.get('available_bytes',0)>RESERVE:
                            break
                os._exit(74)
            if ROOT.is_dir():
                with (ROOT/'disk.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')


def watch():
    while True:
        time.sleep(30)
        guard()


def prepare_data(tmp):
    original=tiktoken.get_encoding('gpt2')
    ranks={s:i for s,i in original._mergeable_ranks.items() if i<4095}
    small=tiktoken.Encoding(name='grid1m_bpe4096',pat_str=original._pat_str,
                            mergeable_ranks=ranks,special_tokens={'<|endoftext|>':4095})
    for split,needed in [('train',TOKENS+4*BATCH*LENGTH),('val',192*1024)]:
        source=tmp/(split+'.source')
        headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']}
        expected=e.DATA_HASHES['val.bin']
        size=656938
        if split=='train':
            size=44040192
            headers['Range']=f'bytes=0-{size-1}'
            expected='87de5bb869a839f19de85fe60f61f3a3815a1929079b0abf0644b395bc7ada65'
        digest=hashlib.sha256()
        with requests.get(f'{DATA_URL}/{split}.bin',headers=headers,stream=True,timeout=90) as response:
            if response.status_code!=(206 if split=='train' else 200):
                raise RuntimeError(f'{split} HTTP {response.status_code}')
            with source.open('wb') as f:
                for buf in response.iter_content(1<<20):
                    if f.tell()+len(buf)>size:
                        raise ValueError('Exceeded bounded source size')
                    digest.update(buf)
                    f.write(buf)
        if source.stat().st_size!=size or digest.hexdigest()!=expected:
            raise ValueError('FineWeb-Edu split verification failed: '+split)
        arr=np.memmap(source,mode='r',dtype=np.uint16)
        path=tmp/source.name
        written=start=0
        with path.open('wb') as out:
            for offset in range(0,len(arr),65536):
                ends=np.flatnonzero(arr[offset:offset+65536]==50256)+offset
                for end in ends:
                    text=original.decode(arr[start:end].tolist(),errors='strict')
                    ids=small.encode_ordinary(text)+[4095]
                    part=np.asarray(ids[:needed-written],dtype=np.uint16)
                    out.write(part.tobytes())
                    written+=len(part)
                    start=end+1
                    if written==needed:
                        break
                if written==needed:
                    break
        if written!=needed:
            raise ValueError('Incomplete split '+split)
        note('data_verified',split=split,tokens=written,source_sha256=digest.hexdigest(),
             prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save_endpoint(model,i,metrics):
    with LOCK:
        guard(home_write=COUNT*4+2_000_000)
        part=ROOT/f'lambda_{i:03d}.part'
        dest=part.with_suffix('.pt')
        weights={n:p.detach().cpu() for n,p in model.named_parameters()}
        torch.save(dict(weights=weights,metrics=metrics,config=vars(model.config)),part)
        part.replace(dest)
        note('endpoint',**metrics,bytes=dest.stat().st_size)


def train(i,train_data,val_data,init_hash):
    cfg=e.config()
    cfg.n_embd,cfg.n_layer,cfg.n_head,cfg.multiple_of=96,6,3,32
    cfg.sequence_length,cfg.vocab_size=LENGTH,4096
    torch.manual_seed(0)
    model=e.Llama(cfg).cuda()
    assert sum(p.numel() for p in model.parameters())==COUNT
    h=e.digest_model(model)
    assert init_hash is None or h==init_hash
    optimizer=e.AdamWSpectralL1Reg(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=0,
                                   spectral_l1_reg_coef=i/100,svt_interval=0)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,total_steps=STEPS,
        pct_start=round(STEPS*2000/19000)/STEPS,anneal_strategy='cos',cycle_momentum=False,
        div_factor=100,final_div_factor=1)
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        x,y=train_data.get(step,'cuda')
        with e.precision(False):
            loss=model(x,y)['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step+1 in (1,STEPS) or (step+1)%250==0:
            note('train',index=i,step=step+1,ce=loss.item())
    metrics=dict(index=i,coefficient=i/100,tokens=TOKENS,init_sha256=h,
                 weight_sha256=e.digest_model(model),validation=e.evaluate(model,val_data,False,64),
                 ranks=e.rank_summary(dict(model.named_parameters())))
    save_endpoint(model,i,metrics)
    return model,h


def analyze(model,val):
    # Evaluate measured endpoints and predictions of withheld grid points.
    names=dict(model.named_parameters())
    def load(i):
        d=torch.load(ROOT/f'lambda_{i:03d}.pt',map_location='cpu',weights_only=True)
        assert e.digest_parameters(d['weights'])==d['metrics']['weight_sha256']
        return d
    with torch.no_grad():
        rows=[]
        for i in range(1,101):
            item=load(i)
            rows.append(item['metrics'])
        for stride in (1,2,5):
            for i in range(1+stride,101-stride,stride):
                left,mid,right=[load(j)['weights'] for j in (i-stride,i,i+stride)]
                err=scale=dot=a2=b2=0.
                for n,p in names.items():
                    a=mid[n]-left[n]
                    b=right[n]-mid[n]
                    pred=(left[n]+right[n])/2
                    err+=(pred-mid[n]).square().sum().item()
                    scale+=(right[n]-left[n]).square().sum().item()
                    dot+=(a*b).sum().item()
                    a2+=a.square().sum().item()
                    b2+=b.square().sum().item()
                    p.copy_(pred)
                note('midpoint',index=i,stride=stride,relative_error=math.sqrt(err/max(scale,1e-30)),
                     tangent_cos=dot/max(math.sqrt(a2*b2),1e-30),pred_ce=e.evaluate(model,val,False,64)['loss'],
                     pred_rank=e.rank_summary(names)['mean_effective_rank'],
                     true_ce=rows[i-1]['validation']['loss'],true_rank=rows[i-1]['ranks']['mean_effective_rank'])
        e.write_json(ROOT/'endpoints.json',rows)
        note('complete',checkpoints=len(list(ROOT.glob('lambda_*.pt'))))


if __name__=='__main__':
    assert torch.cuda.device_count()==1
    guard(home_write=480_000_000,tmp_write=120_000_000)
    ROOT.mkdir(parents=True,exist_ok=False)
    threading.Thread(target=watch,daemon=True).start()
    note('start',gpu=torch.cuda.get_device_name(0),parameters=COUNT,steps=STEPS,tokens=TOKENS,
         batch=BATCH,length=LENGTH,seed=0,data_seed=1337)
    with tempfile.TemporaryDirectory(dir='/tmp',prefix='swd-grid1m-') as folder:
        tmp=Path(folder)
        prepare_data(tmp)
        reader=e.Tokens(tmp/'train.bin',BATCH,LENGTH,1337)
        val=e.Tokens(tmp/'val.bin',8,LENGTH,2026)
        init_hash=None
        for i in range(1,101):
            model,init_hash=train(i,reader,val,init_hash)
        analyze(model,val)
