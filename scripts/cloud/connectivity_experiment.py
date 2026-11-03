"""Matched Llama endpoints and fixed-endpoint quadratic Bezier connections."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from models.llama import Llama
from optim.adamw_spectral_L1_reg import AdamWSpectralL1Reg
from connectivity_guard import enforce

LABELS = ("l2", "swd05", "swd07", "swd1")
COEFFICIENTS = dict(l2=0.1, swd05=0.5, swd07=0.7, swd1=1.0)
DATA_HASHES = {"train.bin": "5a4c532cf1a142834f918f509fc678c178d8edc0df70eaffecc93a955b2f98af",
               "val.bin": "d3025934e78388200ace8ab058eef34a0dd5a0a3e961d08d637665f436c4996e"}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config(smoke: bool = False) -> SimpleNamespace:
    size = os.environ.get("CONNECTIVITY_MODEL_SIZE", "20m")
    if size not in ("20m", "124m"):
        raise ValueError("Unsupported connectivity model size: " + size)
    width, layers, heads = (768, 12, 12) if size == "124m" else (256, 8, 4)
    return SimpleNamespace(n_embd=32 if smoke else width, n_layer=1 if smoke else layers,
                           n_head=2 if smoke else heads, vocab_size=128 if smoke else 50304,
                           sequence_length=16 if smoke else 1024, multiple_of=256,
                           rmsnorm_eps=1e-5, bias=False, dropout=0.0, init_std=0.02,
                           untied_embeds=False, moe=False, parallel_block=False)


def make_model(smoke: bool) -> Llama:
    torch.manual_seed(0)
    return Llama(config(smoke)).to("cpu" if smoke else "cuda")


def digest_model(model: torch.nn.Module) -> str:
    return digest_parameters(dict(model.named_parameters()))


def digest_parameters(parameters: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters.items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Tokens:
    """Same shuffled, non-overlapping sequence sampling as the repository DataReader."""

    def __init__(self, path: Path, batch: int, length: int, seed: int = 1337):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch, self.length, self.seed = batch, length, seed
        self.epoch = -1
        self.epoch_batches = ((len(self.data) - length - 1) // length - 1) // batch
        if self.epoch_batches < 1:
            raise ValueError("Not enough tokens for one minibatch")

    def get(self, step: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, index = divmod(step, self.epoch_batches)
        if self.epoch != epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self.order = rng.permutation((len(self.data) - self.length - 1) // self.length - 1)
            self.offset = rng.integers(self.length)
            self.epoch = epoch
        starts = self.order[index * self.batch:(index + 1) * self.batch] * self.length + self.offset
        xy = np.stack([self.data[i:i + self.length + 1] for i in starts]).astype(np.int64)
        return (torch.from_numpy(xy[:, :-1].copy()).to(device),
                torch.from_numpy(xy[:, 1:].copy()).to(device))


def precision(smoke: bool):
    return contextlib.nullcontext() if smoke else torch.autocast("cuda", dtype=torch.bfloat16)


class Recorder:
    def __init__(self, root: Path, stage: str, metadata: dict, smoke: bool):
        self.path = root / stage / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = None
        if not smoke and os.environ.get("WANDB_API_KEY"):
            import wandb
            self.run = wandb.init(project="ns_weights", entity="andrey", group=root.name,
                                  name=root.name + "_" + stage, config=metadata,
                                  dir="/tmp", settings=wandb.Settings(init_timeout=120))

    def log(self, payload: dict) -> None:
        payload = dict(timestamp=time.time(), **payload)
        with self.path.open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
        print("METRIC=" + json.dumps(payload), flush=True)
        if self.run:
            self.run.log(payload)

    def finish(self) -> None:
        if self.run:
            self.run.finish()


@torch.no_grad()
def evaluate(model: Llama, reader: Tokens, smoke: bool, batches: int,
             parameters: dict | None = None) -> dict:
    model.eval()
    loss, correct, total = 0.0, 0, 0
    device = str(next(model.parameters()).device)
    for index in range(batches):
        x, y = reader.get(index, device)
        with precision(smoke):
            if parameters is None:
                output = model(x, y, get_logits=True)
            else:
                output = torch.func.functional_call(model, parameters, (x, y),
                                                     dict(get_logits=True), tie_weights=True)
        tokens = y.numel()
        loss += output["loss"].item() * tokens
        correct += (output["logits"].argmax(-1) == y).sum().item()
        total += tokens
    mean = loss / total
    return dict(loss=mean, perplexity=math.exp(mean), error_percent=100 * (1 - correct / total), tokens=total)


def rank_summary(parameters: dict[str, torch.Tensor]) -> dict:
    rows = []
    with torch.no_grad():
        for name, weight in parameters.items():
            if weight.ndim != 2 or ".h." not in name:
                continue
            singular = torch.linalg.svdvals(weight.float())
            p = singular / singular.sum().clamp_min(1e-30)
            erank = (-(p * p.clamp_min(1e-30).log()).sum()).exp().item()
            rows.append(dict(name=name, effective_rank=erank, full_rank=min(weight.shape)))
    return dict(matrices=rows, mean_effective_rank=sum(r["effective_rank"] for r in rows) / len(rows))


def train_endpoint(root: Path, label: str, data: Path, smoke: bool) -> None:
    model = make_model(smoke)
    cfg, device = model.config, str(next(model.parameters()).device)
    count = sum(p.numel() for p in model.parameters())
    batch, accumulation = (1, 1) if smoke else (16, 8)
    steps = 4 if smoke else math.ceil(20 * count / (batch * accumulation * cfg.sequence_length))
    warmup = 2 if smoke else round(steps * 2000 / 19000)
    params = dict(model.named_parameters())
    groups = [{**g, "params": [params[n] for n in g["params"]]} for g in model.get_parameter_group_specs(cfg)]
    if label == "l2":
        optimizer = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    else:
        optimizer = AdamWSpectralL1Reg(groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                                       weight_decay=0, spectral_l1_reg_coef=COEFFICIENTS[label], svt_interval=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps,
                    pct_start=warmup / steps, anneal_strategy="cos", cycle_momentum=False,
                    div_factor=100, final_div_factor=1)
    metadata = dict(model=vars(cfg), unique_parameters=count, steps=steps, warmup=warmup,
                    batch=batch, accumulation=accumulation, seed=0, data_seed=1337,
                    tokens=steps * batch * accumulation * cfg.sequence_length,
                    optimizer=type(optimizer).__name__, coefficient=COEFFICIENTS[label],
                    initial_model_sha256=digest_model(model), lr=0.001,
                    spectral_applies_to_embedding=label != "l2")
    write_json(root / label / "config.json", metadata)
    recorder = Recorder(root, label, metadata, smoke)
    train = Tokens(data / "train.bin", batch, cfg.sequence_length)
    validation = Tokens(data / "val.bin", 1 if smoke else 8, cfg.sequence_length, 2026)
    started = time.monotonic()
    for step in range(steps + 1):
        if step % (2 if smoke else 200) == 0 or step == steps:
            result = evaluate(model, validation, smoke, 1 if smoke else 16)
            recorder.log(dict(step=step, val_loss=result["loss"], elapsed=time.monotonic() - started))
        if step == steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accumulation):
            x, y = train.get(step * accumulation + micro, device)
            with precision(smoke):
                loss = model(x, y)["loss"] / accumulation
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        if step % (1 if smoke else 20) == 0:
            recorder.log(dict(step=step + 1, train_loss=loss_sum, lr=optimizer.param_groups[0]["lr"],
                              tokens=(step + 1) * batch * accumulation * cfg.sequence_length,
                              elapsed=time.monotonic() - started))
    result = dict(metadata=metadata, validation=result, ranks=rank_summary(dict(model.named_parameters())),
                  final_model_sha256=digest_model(model), elapsed=time.monotonic() - started)
    checkpoint = root / "checkpoints" / (label + ".pt")
    enforce(root, pending=4 * count + 16_000_000, smoke=smoke)
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(dict(model={k: p.detach().cpu() for k, p in model.named_parameters()}, config=vars(cfg)), checkpoint)
    write_json(root / label / "final.json", result)
    recorder.finish()


def interpolate(left: dict, right: dict, control: dict, t: float, straight: bool = False) -> dict:
    if t == 0:
        return dict(left)
    if t == 1:
        return dict(right)
    if straight:
        return {name: (1 - t) * left[name] + t * right[name] for name in left}
    return {name: (1 - t)**2 * left[name] + 2 * t * (1 - t) * control[name] + t*t * right[name] for name in left}


def train_path(root: Path, pair: str, data: Path, smoke: bool) -> None:
    left_label, right_label = pair.split("__")
    model = make_model(smoke)
    device = str(next(model.parameters()).device)
    for p in model.parameters():
        p.requires_grad_(False)
    endpoints = []
    for label in (left_label, right_label):
        checkpoint = torch.load(root / "checkpoints" / (label + ".pt"), map_location=device, weights_only=True)
        endpoints.append({k: v.detach().requires_grad_(False) for k, v in checkpoint["model"].items()})
    left, right = endpoints
    endpoint_hashes = [digest_parameters(endpoint) for endpoint in endpoints]
    originals = [{k: v.clone() for k, v in endpoint.items()} for endpoint in endpoints] if smoke else None
    control = {k: torch.nn.Parameter((left[k] + right[k]) / 2) for k in left}
    if smoke:
        assert model.transformer.wte.weight is model.lm_head.weight
        assert all(torch.equal(interpolate(left, right, control, 0)[k], left[k]) for k in left)
        assert all(torch.equal(interpolate(left, right, control, 1)[k], right[k]) for k in left)
        initial_bezier = interpolate(left, right, control, 0.37)
        initial_straight = interpolate(left, right, control, 0.37, straight=True)
        assert all(torch.allclose(initial_bezier[k], initial_straight[k], atol=1e-7) for k in left)
    batch, accumulation = (1, 1) if smoke else (16, 8)
    count = sum(p.numel() for p in model.parameters())
    steps = 4 if smoke else math.ceil(20 * count / (batch * accumulation * model.config.sequence_length))
    warmup = 2 if smoke else round(steps * 100 / 3006)
    optimizer = torch.optim.Adam(list(control.values()), lr=1e-3, betas=(0.9, 0.95), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps,
                      pct_start=warmup / steps, anneal_strategy="cos",
                      cycle_momentum=False, div_factor=100, final_div_factor=1)
    train = Tokens(data / "train.bin", batch, model.config.sequence_length)
    # Start after the endpoint training stream for curve fitting.
    offset = 0 if smoke else steps * accumulation
    metadata = dict(pair=pair, steps=steps, warmup=warmup, unique_parameters=count,
                    tokens=steps * batch * accumulation * model.config.sequence_length,
                    lr=0.001, batch=batch, accumulation=accumulation,
                    objective="unregularized training cross_entropy, t uniform[0,1]",
                    curve="quadratic Bezier, fixed endpoints", train_stream_offset_microsteps=offset)
    write_json(root / pair / "config.json", metadata)
    recorder = Recorder(root, pair, metadata, smoke)
    generator = np.random.default_rng(20261103)
    started = time.monotonic()
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t = float(generator.uniform())
        loss_sum = 0.0
        for micro in range(accumulation):
            x, y = train.get(offset + step * accumulation + micro, device)
            parameters = interpolate(left, right, control, t)
            with precision(smoke):
                loss = torch.func.functional_call(model, parameters, (x, y), tie_weights=True)["loss"] / accumulation
            loss.backward()
            loss_sum += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(list(control.values()), 0.5)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite curve gradient")
        if smoke:
            assert norm > 0
        optimizer.step()
        scheduler.step()
        if step % (1 if smoke else 20) == 0 or step == steps - 1:
            recorder.log(dict(step=step + 1, curve_train_loss=loss_sum, t=t, elapsed=time.monotonic() - started))
    if smoke:
        assert all(torch.equal(endpoint[k], original[k]) for endpoint, original in zip(endpoints, originals) for k in endpoint)
        assert any(not torch.equal(control[k], (left[k] + right[k]) / 2) for k in left)
    control = {k: p.detach() for k, p in control.items()}
    readers = {split: Tokens(data / (split + ".bin"), 1 if smoke else 8,
                            model.config.sequence_length, 2026) for split in ("train", "val")}
    grid = np.linspace(0, 1, 3 if smoke else 51)
    rows = []
    for kind in ("straight", "bezier"):
        for t in grid:
            weights = interpolate(left, right, control, float(t), straight=kind == "straight")
            row = dict(curve=kind, t=float(t))
            for split, reader in readers.items():
                row[split] = evaluate(model, reader, smoke, 1 if smoke else 16, weights)
            rows.append(row)
            recorder.log(dict(curve=kind, grid_t=float(t), grid_val_loss=row["val"]["loss"],
                              grid_train_loss=row["train"]["loss"]))
            write_json(root / pair / "curves.json", rows)
    surfaces = []
    for a in np.linspace(-0.1, 1.1, 3 if smoke else 21):
        for b in np.linspace(-0.1, 0.65, 3 if smoke else 17):
            weights = {k: left[k] + float(a) * (right[k] - left[k]) + float(b) * (control[k] - (left[k]+right[k])/2) for k in left}
            evaluation = evaluate(model, readers["val"], smoke, 1, weights)
            surfaces.append(dict(a=float(a), b=float(b), **evaluation))
        write_json(root / pair / "surface.json", surfaces)
    rank_rows = []
    for t in np.linspace(0, 1, 3 if smoke else 5):
        rank_rows.append(dict(t=float(t), **rank_summary(interpolate(left, right, control, float(t)))))
    summary = {}
    for kind in ("straight", "bezier"):
        subset = [row for row in rows if row["curve"] == kind]
        summary[kind] = {}
        for split in ("train", "val"):
            l0, l1 = subset[0][split]["loss"], subset[-1][split]["loss"]
            summary[kind][split] = dict(endpoint0=l0, endpoint1=l1,
                max_loss=max(row[split]["loss"] for row in subset),
                barrier_above_worse_endpoint=max(row[split]["loss"] for row in subset) - max(l0,l1),
                barrier_above_chord=max(row[split]["loss"] - ((1-row["t"])*l0+row["t"]*l1) for row in subset))
    result = dict(metadata=metadata, summary=summary, ranks=rank_rows,
                  elapsed=time.monotonic()-started, endpoint_parameters_frozen=True,
                  endpoint_sha256=endpoint_hashes,
                  control_checkpoint_saved=False, surface_evaluation_tokens=surfaces[0]["tokens"])
    if [digest_parameters(endpoint) for endpoint in endpoints] != endpoint_hashes:
        raise RuntimeError("Curve fitting mutated frozen endpoint weights")
    write_json(root / pair / "final.json", result)
    recorder.finish()
