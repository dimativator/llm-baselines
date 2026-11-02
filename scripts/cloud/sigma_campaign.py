"""124M AdamW reference, spectral pretraining, and independent Sigma continuations."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(os.environ.get("SIGMA_ROOT", "/workspace-SR006.nfs3/dimativator/sigma-only-124m-20261102"))
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
RESERVE = 5_000_000_000
BOOTSTRAP = "/home/jovyan/shares/SR006.nfs3/mlsub-logs"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def space(pending: int = 0) -> dict:
    records = []
    cloud = os.environ.get("SIGMA_CPU_SMOKE") != "1"
    required = (str(ROOT), "/home/jovyan", "/tmp", "/", BOOTSTRAP) if cloud else (str(ROOT), "/tmp", "/")
    for path in required:
        if cloud and not Path("/workspace-SR006.nfs3").resolve().is_mount():
            records.append(dict(path=path, ok=False, error="NFS3 mount unavailable"))
            continue
        if cloud and path in ("/home/jovyan", BOOTSTRAP) and not Path(path).is_dir():
            records.append(dict(path=path, ok=False, error="Required path unavailable"))
            continue
        parent = Path(path)
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        stat = os.statvfs(parent)
        available = stat.f_bavail * stat.f_frsize
        records.append(dict(path=path, measured_path=str(parent), available_bytes=available,
                            device=os.stat(parent).st_dev, ok=available > RESERVE + pending))
    return dict(timestamp=time.time(), paths=records, pending_bytes=pending,
                ok=all(row["ok"] for row in records))


def enforce(pending: int = 0) -> None:
    record = space(pending)
    print("DISK=" + json.dumps(record), flush=True)
    if not record["ok"]:
        raise RuntimeError("DISK_RESERVE_UNAVAILABLE " + json.dumps(record))


def run_worker(stage: str, smoke: bool) -> None:
    import torch
    import main as trainer
    import optim.base as base
    from compression.svd_llm import collect_input_covariance
    from optim.utils import eval as evaluate
    from rank_profiles import verify_data, DATA_ROOT
    from sigma_factors import factorize, measure

    enforce()
    def watchdog() -> None:
        while True:
            time.sleep(30)
            try:
                enforce()
            except (OSError, RuntimeError) as error:
                print("DISK GUARD FAILED: STOPPING WORKER " + str(error), flush=True)
                os._exit(70)
    threading.Thread(target=watchdog, daemon=True).start()
    data = os.environ.get("TOKENIZED_DATA_DIR", str(DATA_ROOT))
    if not smoke:
        verify_data(Path(data))
    steps = 4 if smoke else (int(stage.split("_")[1]) if stage.startswith("sigma_") else 19000)
    args_list = ["main.py", "--experiment_name", stage, "--results_base_folder", str(ROOT),
                 "--model", "llama", "--dataset", "finewebedu", "--datasets_dir", "/tmp",
                 "--tokenized_data_dir", data, "--opt", "adamw-spectral-l1-reg" if stage == "spectral" else "adamw",
                 "--weight_decay", "0.1" if stage == "reference" else "0", "--spectral_l1_reg_coef", "1",
                 "--lr", "0.005", "--iterations", str(steps), "--warmup_steps", "2" if smoke else "2000",
                 "--n_embd", "32" if smoke else "768", "--n_head", "2" if smoke else "12",
                 "--n_layer", "1" if smoke else "12", "--batch_size", "1" if smoke else "32",
                 "--acc_steps", "1" if smoke else "4", "--sequence_length", "16" if smoke else "1024",
                 "--eval_batch_size", "1" if smoke else "16", "--eval_batches", "2" if smoke else "64",
                 "--grad_clip", "0.5", "--seed", "0", "--data_seed", "1337", "--scheduler", "cos",
                 "--dropout", "0", "--beta1", "0.9", "--beta2", "0.95", "--finewebedu_max_files", "5",
                 "--eval_interval", "500", "--log_interval", "1" if smoke else "20",
                 "--latest_ckpt_interval", "0", "--permanent_ckpt_interval", "0",
                 "--effective_rank_interval", "0"]
    if stage.startswith("sigma_"):
        args_list[args_list.index("--warmup_steps") + 1] = "0"
        args_list[args_list.index("--scheduler") + 1] = "none"
    if smoke:
        args_list += ["--device", "cpu", "--dtype", "float32", "--vocab_size", "128"]
        import numpy as np
        token_dir = ROOT / "smoke_tokens"
        token_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (np.arange(256, dtype=np.uint16) % 128).tofile(token_dir / (split + ".bin"))
        args_list[args_list.index("--tokenized_data_dir") + 1] = str(token_dir)
    sys.argv = args_list
    cfg, parser = trainer.get_args()
    original_train = trainer.train
    original_eval = base.eval_and_log
    original_numeric_eval = base.eval

    def numeric_eval(*args, **kwargs):
        values = original_numeric_eval(*args, **kwargs)
        write_json(ROOT / stage / "latest_validation.json", dict(timestamp=time.time(), val_loss=values[1]))
        print("VALIDATION=" + json.dumps(dict(stage=stage, val_loss=values[1], timestamp=time.time())), flush=True)
        return values

    base.eval = numeric_eval

    def logged_eval(*args, **kwargs):
        original_eval(*args, **kwargs)
        model = kwargs.get("model", args[3] if len(args) > 3 else None)
        iteration = kwargs.get("curr_iter", args[1] if len(args) > 1 else None)
        print("PROGRESS=" + json.dumps(dict(stage=stage, step=iteration, timestamp=time.time(),
              peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() and not smoke else 0)), flush=True)

    base.eval_and_log = logged_eval

    def wrapped_train(**kwargs):
        model = kwargs["model"]
        readers = kwargs["datareaders"]
        val_reader = readers["val"]
        calibration = None
        if stage.startswith("sigma_"):
            checkpoint = torch.load(ROOT / "spectral_final.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            train_reader = readers["train"]
            train_reader.set_step(0)

            def batches():
                for _ in range(2 if smoke else 16):
                    yield train_reader.sample_batch()[0]

            covariance = collect_input_covariance(model, batches(), n_batches=2 if smoke else 16, device=cfg.device)
            calibration = factorize(model, covariance)
            del covariance
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.005,
                                   betas=(0.9, 0.95), weight_decay=0)
            kwargs["opt"] = opt
            kwargs["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
            cfg.scheduler = "cos"
            # Continue from the first unseen training batch, using a fresh optimizer.
            original_set_step = train_reader.set_step
            train_reader.set_step = lambda step: original_set_step(step + (0 if smoke else 19000 * cfg.acc_steps))
            assert sum(p.requires_grad for p in model.parameters()) == len(calibration)
            write_json(ROOT / stage / "factorization.json", calibration)

        stats = original_train(**kwargs)
        model.eval()
        val_reader.set_step(0)
        context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if not smoke else __import__("contextlib").nullcontext()
        _, loss, perplexity, _, _ = evaluate(model, val_reader, cfg.device,
            max_num_batches=val_reader.num_batches(), ctx=context, cfg=cfg)
        if not __import__("math").isfinite(loss):
            raise RuntimeError("Nonfinite final validation loss")
        ranks = measure(model)
        result = dict(stage=stage, steps=steps, val_loss=loss, perplexity=perplexity, ranks=ranks,
                      factorization=calibration, args=vars(cfg), timestamp=time.time(),
                      optimizer_class=type(kwargs["opt"]).__name__,
                      scheduler_class=type(kwargs["scheduler"]).__name__,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated() if not smoke else 0)
        if stage == "spectral":
            expected = sum(p.numel() * p.element_size() for p in model.state_dict().values()) + 64_000_000
            enforce(expected)
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(cfg)},
                       ROOT / "spectral_final.pt")
        write_json(ROOT / stage / "final.json", result)
        print("FINAL=" + json.dumps({k: v for k, v in result.items() if k not in ("ranks", "args", "factorization")}), flush=True)
        return stats

    trainer.train = wrapped_train
    trainer.main(cfg, parser)
    write_json(ROOT / (stage + "_DONE.json"), dict(stage=stage, timestamp=time.time()))


def coordinate(smoke: bool = False) -> None:
    enforce()
    ROOT.mkdir(parents=True, exist_ok=True)
    active = {}
    pending = ["reference", "spectral", "sigma_1000", "sigma_3000", "sigma_5000"]
    with (ROOT / "STARTED.json").open("x") as stream:
        json.dump(dict(timestamp=time.time(), smoke=smoke), stream)
    try:
        while pending or active:
            enforce()
            for gpu, (stage, process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    write_json(ROOT / (stage + "_EXIT.json"), dict(exit_code=code, timestamp=time.time()))
                    del active[gpu]
                    if code or not (ROOT / (stage + "_DONE.json")).exists():
                        raise RuntimeError(f"Worker {stage} failed with code {code}")
            for gpu in range(1 if smoke else 2):
                if gpu in active:
                    continue
                eligible = next((s for s in pending if not s.startswith("sigma_") or (ROOT / "spectral_DONE.json").exists()), None)
                if eligible is None:
                    continue
                pending.remove(eligible)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                log = (ROOT / (eligible + ".log")).open("w")
                command = [sys.executable, __file__, "--stage", eligible]
                if smoke:
                    command.append("--smoke")
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (eligible, process, log)
                write_json(ROOT / "active.json", {str(g): dict(stage=s, pid=p.pid) for g, (s, p, _) in active.items()})
            if active:
                if not smoke:
                    telemetry = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=10)
                    print("GPU_MEMORY=" + telemetry.stdout.strip(), flush=True)
                time.sleep(5 if smoke else 30)
        finals = [json.loads((ROOT / stage / "final.json").read_text()) for stage in
                  ("reference", "spectral", "sigma_1000", "sigma_3000", "sigma_5000")]
        write_json(ROOT / "results.json", finals)
        (ROOT / "spectral_final.pt").unlink()
        write_json(ROOT / "COMPLETE.json", dict(timestamp=time.time(), checkpoint_removed=True))
    except BaseException as error:
        stopped = []
        for stage, process, log in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGSTOP)
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    stopped.append(dict(stage=stage, pid=process.pid, stopped=True))
                except (ProcessLookupError, subprocess.TimeoutExpired) as stop_error:
                    stopped.append(dict(stage=stage, pid=process.pid, stopped=process.poll() is not None,
                                        error=str(stop_error)))
            log.close()
        checkpoint = ROOT / "spectral_final.pt"
        disk = space()
        if not disk["ok"]:
            print("CRITICALLY LOW CLOUD.RU DISK SPACE: TRACKED WORKERS STOPPED. " + json.dumps(disk), flush=True)
        if not disk["ok"] and checkpoint.is_file() and not checkpoint.is_symlink():
            checkpoint.unlink()
            print("CHECKPOINT_DELETED=" + json.dumps(dict(path=str(checkpoint), before=disk, after=space())), flush=True)
        write_json(ROOT / "FAILED.json", dict(error=str(error), timestamp=time.time(), disk=disk, stopped_workers=stopped))
        print("CAMPAIGN_FAILED=" + str(error), flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["reference", "spectral", "sigma_1000", "sigma_3000", "sigma_5000"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        print("SPACE_SNAPSHOT=" + json.dumps(space()), flush=True)
        for path in ("/workspace-SR006.nfs2", "/home/jovyan/shares/SR006.nfs2"):
            if Path(path).exists():
                stat = os.statvfs(path)
                print("NFS2=" + json.dumps(dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize)), flush=True)
            else:
                print("NFS2=" + json.dumps(dict(path=path, error="Unavailable")), flush=True)
        for stage in ("reference", "spectral", "sigma_1000", "sigma_3000", "sigma_5000"):
            for name in ("latest_validation.json", "factorization.json", "final.json"):
                path = ROOT / stage / name
                if path.is_file():
                    print("ARTIFACT=" + json.dumps(dict(stage=stage, name=name, value=json.loads(path.read_text()))), flush=True)
            log = ROOT / (stage + ".log")
            if log.is_file():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    print("LOG=" + json.dumps(dict(stage=stage, tail=stream.read().decode(errors="replace").splitlines()[-12:])), flush=True)
        for name in ("COMPLETE.json", "FAILED.json", "active.json"):
            path = ROOT / name
            if path.is_file():
                print("MARKER=" + json.dumps(dict(name=name, value=json.loads(path.read_text()))), flush=True)
        expected = {"reference": 19000, "spectral": 19000, "sigma_1000": 1000,
                    "sigma_3000": 3000, "sigma_5000": 5000}
        results_path = ROOT / "results.json"
        results = json.loads(results_path.read_text()) if results_path.is_file() else []
        stages = [row.get("stage") for row in results]
        consistent = len(results) == 5 and set(stages) == set(expected)
        for row in results:
            final = ROOT / str(row.get("stage")) / "final.json"
            consistent = consistent and final.is_file() and row == json.loads(final.read_text())
            consistent = consistent and row.get("steps") == expected.get(row.get("stage")) and math.isfinite(row.get("val_loss", math.nan))
        checkpoints = []
        for folder, _, files in os.walk(ROOT, followlinks=False):
            checkpoints.extend(str(Path(folder) / name) for name in files
                               if Path(name).suffix in (".pt", ".pth", ".safetensors"))
        print("COMPLETION_AUDIT=" + json.dumps(dict(timestamp=time.time(),
              results_exists=results_path.is_file(), results_consistent=consistent,
              stages=stages, checkpoint_paths=checkpoints,
              sole_checkpoint_exists=(ROOT / "spectral_final.pt").exists())), flush=True)
        return
    if args.stage:
        run_worker(args.stage, args.smoke)
    else:
        coordinate(args.smoke)


if __name__ == "__main__":
    main()
