"""Cloud measurements and matched AdamW runs, with ephemeral checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

REPO = "DimaTivator/effpretrain_ckpts"
PREFIX = "cf_bruteforce_124M"
CAMPAIGN = "rank_profiles_20261012"
ROOT = Path(os.environ.get("RANK_RESULT_ROOT", "/workspace-SR006.nfs3/dimativator/results/" + CAMPAIGN))
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def upload_verified(path: Path, destination: str) -> str:
    api = HfApi()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = api.upload_file(path_or_fileobj=str(path), path_in_repo=destination,
                             repo_id=REPO, commit_message=f"{CAMPAIGN}: {path.name}")
    remote = api.get_paths_info(REPO, [destination], revision=commit.oid)[0]
    if remote.size != path.stat().st_size:
        raise RuntimeError("HF upload size verification failed")
    if remote.lfs is not None:
        if remote.lfs.sha256 != digest:
            raise RuntimeError("HF upload SHA256 verification failed")
    else:
        with tempfile.TemporaryDirectory(prefix="rank_verify_", dir="/tmp") as cache:
            downloaded = hf_hub_download(REPO, destination, revision=commit.oid, cache_dir=cache)
            if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != digest:
                raise RuntimeError("HF upload content verification failed")
    print(f"UPLOAD_VERIFIED {destination} revision={commit.oid} sha256={digest}", flush=True)
    return commit.oid


@torch.no_grad()
def measure(state: dict, config: dict, label: str, family: str, coef: float,
            source: str, revision: str, device: str) -> dict:
    rows = []
    spectra = {}
    for key, weight in state.items():
        if not key.endswith(".weight") or weight.ndim != 2:
            continue
        name = key.removesuffix(".weight")
        # Embeddings and the tied head are measured but excluded from the
        # training-compatible mean over transformer matrices.
        block_match = re.search(r"(?:^|\.)h\.(\d+)\.(.+)", name)
        layer_type = block_match.group(2) if block_match else name.split(".")[-1]
        block = int(block_match.group(1)) if block_match else None
        singular = torch.linalg.svdvals(weight.to(device=device, dtype=torch.float32))
        positive = singular[singular > 0]
        if positive.numel():
            p = positive / positive.sum()
            erank = torch.exp(-(p * p.log()).sum()).item()
            stable = (singular.square().sum() / singular[0].square()).item()
        else:
            erank, stable = 0.0, 0.0
        full = min(weight.shape)
        rows.append(dict(model_size=f"{config['n_embd']}x{config['n_layer']}",
                         family=family, coefficient=coef, experiment=label,
                         iteration=int(config["iterations"]), parameter=key,
                         block_index=block, layer_type=layer_type,
                         rows=weight.shape[0], cols=weight.shape[1], full_rank=full,
                         effective_rank=erank, effective_rank_percent=100 * erank / full,
                         stable_rank=stable, included_in_transformer_mean=block is not None))
        spectra[key] = singular.cpu().tolist()
        print(f"RANK {label} {key} {erank:.6f} / {full}", flush=True)
        del singular, positive
    expected = config["n_layer"] * 5
    transformer = [r for r in rows if r["included_in_transformer_mean"]]
    if len(transformer) != expected:
        raise RuntimeError(f"Expected {expected} transformer matrices, got {len(transformer)}")
    weighted = sum(r["effective_rank"] * r["full_rank"] for r in transformer) / sum(r["full_rank"] for r in transformer)
    return dict(experiment=label, family=family, coefficient=coef, source=source,
                source_revision=revision, config=config, rows=rows, singular_values=spectra,
                mean_weighted_effective_rank=weighted,
                rank_definition="exp(-sum(p*log(p))), p=sigma/sum(sigma), FP32 SVD",
                zero_matrix_effective_rank=0, group_std="population (ddof=0)")


def persist(payload: dict, publish: bool = True) -> None:
    label = payload["experiment"]
    path = ROOT / (label + ".json")
    write_json(path, payload)
    rows = payload["rows"]
    csv_path = ROOT / (label + ".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["layer_type"]].append(row["effective_rank_percent"])
    grouped = []
    for layer, values in groups.items():
        mean = sum(values) / len(values)
        grouped.append(dict(family=payload["family"], coefficient=payload["coefficient"],
                            experiment=label, layer_type=layer, count=len(values),
                            mean_percent=mean,
                            std_percent=(sum((v-mean)**2 for v in values) / len(values))**0.5,
                            min_percent=min(values), max_percent=max(values)))
    group_path = ROOT / (label + "_grouped.csv")
    with group_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    if publish:
        for artifact in (path, csv_path, group_path):
            upload_verified(artifact, f"{PREFIX}/{CAMPAIGN}/{artifact.name}")
    write_json(ROOT / (label + "_DONE.json"), dict(experiment=label, status="complete"))


def spectral(device: str, size: str) -> None:
    api = HfApi()
    revision = api.model_info(REPO).sha
    paths = api.list_repo_files(REPO, revision=revision)
    checkpoints = [p for p in paths if p.startswith(PREFIX + f"/llama{size}m_")
                   and p.endswith("/ckpts/latest/main.pt")]
    checkpoints.sort(key=lambda p: float(re.search(r"_sl1_([^_]+)_", p).group(1)))
    if not checkpoints:
        raise RuntimeError("No spectral checkpoints found")
    write_json(ROOT / f"manifest_{size}m.json", dict(repo=REPO, revision=revision, checkpoints=checkpoints))
    for checkpoint in checkpoints:
        label = checkpoint.split("/")[1]
        if (ROOT / (label + "_DONE.json")).exists():
            continue
        coef = float(re.search(r"_sl1_([^_]+)_", label).group(1))
        with tempfile.TemporaryDirectory(prefix="rank_checkpoint_", dir="/tmp") as temporary:
            summary_path = hf_hub_download(REPO, f"{PREFIX}/{label}/summary.json", revision=revision, cache_dir=temporary)
            summary = json.loads(Path(summary_path).read_text())
            downloaded = hf_hub_download(REPO, checkpoint, revision=revision, cache_dir=temporary)
            loaded = torch.load(downloaded, map_location="cpu", weights_only=False)
            if loaded["itr"] != summary["args"]["iterations"]:
                raise RuntimeError(f"Not final: {label}, iteration={loaded['itr']}")
            payload = measure(loaded["model"], summary["args"], label, "spectral_wd", coef,
                              checkpoint, revision, device)
            payload["checkpoint_sha256"] = hashlib.sha256(Path(downloaded).read_bytes()).hexdigest()
            del loaded
            persist(payload)
        print(f"CHECKPOINT_REMOVED {label}", flush=True)
    print(f"SPECTRAL_COMPLETE size={size} count={len(checkpoints)}", flush=True)


def l2(coef: float, smoke: bool = False) -> None:
    import main as trainer
    from optim.utils import save_checkpoint
    label = f"llama124m_adamw_wd{coef:g}_lr5e-3_finewebedu_rankprofile_20261012"
    if not smoke and (ROOT / (label + "_DONE.json")).exists():
        print("ALREADY_COMPLETE " + label)
        return
    data = os.environ.get("TOKENIZED_DATA_DIR")
    if not data and not smoke:
        raise RuntimeError("TOKENIZED_DATA_DIR must point to the matched spectral training tokens")
    iterations = 2 if smoke else 19000
    with tempfile.TemporaryDirectory(prefix="rank_l2_", dir="/tmp") as temporary:
        if smoke:
            import numpy as np
            data = temporary + "/tokens"
            Path(data).mkdir()
            for split in ("train", "val"):
                np.arange(4096, dtype=np.uint16).tofile(Path(data) / (split + ".bin"))
        sys.argv = ["main.py", "--experiment_name", label, "--results_base_folder", temporary,
                    "--model", "llama", "--dataset", "finewebedu", "--datasets_dir", "/tmp",
                    "--tokenized_data_dir", data, "--opt", "adamw", "--weight_decay", str(coef),
                    "--lr", "0.005", "--iterations", str(iterations), "--warmup_steps", "1" if smoke else "2000",
                    "--n_embd", "768", "--n_head", "12", "--n_layer", "12",
                    "--batch_size", "1" if smoke else "64", "--acc_steps", "1" if smoke else "2",
                    "--sequence_length", "32" if smoke else "1024", "--eval_batch_size", "1" if smoke else "64",
                    "--eval_batches", "1" if smoke else "64", "--grad_clip", "0.5", "--seed", "0",
                    "--data_seed", "1337", "--scheduler", "cos", "--dropout", "0",
                    "--beta1", "0.9", "--beta2", "0.95", "--finewebedu_max_files", "5",
                    "--eval_interval", "137", "--log_interval", "4", "--latest_ckpt_interval", "0",
                    "--permanent_ckpt_interval", "0", "--effective_rank_interval", "0",
                    "--hf_checkpoint_repo", REPO, "--hf_checkpoint_path", f"{PREFIX}/{label}/ckpts/latest/main.pt"]
        if not smoke:
            sys.argv += ["--wandb", "--wandb_project", "ns_weights", "--wandb_entity", "andrey"]
        elif not torch.cuda.is_available():
            sys.argv += ["--device", "cpu", "--dtype", "float32"]
        args, parser = trainer.get_args()

        def final_checkpoint(model, opt, scheduler, itr, repo_id, path_in_repo):
            with tempfile.TemporaryDirectory(prefix="rank_final_", dir="/tmp") as final_dir:
                save_checkpoint(model, opt, scheduler, itr, Path(final_dir))
                checkpoint = Path(final_dir) / "main.pt"
                raw = model.module if hasattr(model, "module") else model
                config = vars(args).copy()
                if not smoke:
                    revision = upload_verified(checkpoint, path_in_repo)
                    payload = measure(raw.state_dict(), config, label, "adamw_l2", coef,
                                      path_in_repo, revision, "cuda:0")
                    payload["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    persist(payload)
                    summary = Path(temporary) / label / "summary.json"
                    upload_verified(summary, f"{PREFIX}/{label}/summary.json")
                else:
                    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    assert loaded["itr"] == 2
            print(f"FINAL_CHECKPOINT_REMOVED {label}", flush=True)
        trainer.upload_checkpoint_to_hf = final_checkpoint
        trainer.main(args, parser)
    print("SMOKE_COMPLETE" if smoke else "L2_COMPLETE " + label, flush=True)


def inspect() -> None:
    print("HF_IDENTITY " + HfApi().whoami()["name"])
    for base in (Path("/workspace-SR006.nfs2/dimativator"), Path("/workspace-SR006.nfs3/dimativator"), Path("/home/jovyan")):
        print(f"STORAGE {base} exists={base.exists()}")
        if base.exists():
            for folder, dirs, files in os.walk(base):
                relative = Path(folder).relative_to(base)
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                if len(relative.parts) >= 3:
                    dirs[:] = []
                if "train.bin" in files or "val.bin" in files:
                    print("TOKEN_CANDIDATE " + folder)
                    for name in ("train.bin", "val.bin"):
                        path = Path(folder) / name
                        if path.exists():
                            print(f"TOKEN_FILE {path} bytes={path.stat().st_size}")
    print("IMPORT_SMOKE")
    import main
    print("IMPORT_OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "spectral", "l2", "smoke"))
    parser.add_argument("--size", choices=("124", "257"), default="124")
    parser.add_argument("--coef", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if arguments.mode == "inspect":
        inspect()
    elif arguments.mode == "spectral":
        spectral(arguments.device, arguments.size)
    else:
        l2(arguments.coef, smoke=arguments.mode == "smoke")
