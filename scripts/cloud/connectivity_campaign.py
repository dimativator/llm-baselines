"""Two two-GPU lanes, bounded storage, six pairs, and verified result export."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from connectivity_guard import space, enforce, cleanup_checkpoints

ROOT = Path(os.environ.get("CONNECTIVITY_ROOT", "/workspace-SR006.nfs3/dimativator/mode-connectivity-20m-20261103"))
DATA = Path(os.environ.get("TOKENIZED_DATA_DIR", "/workspace-SR006.nfs3/dimativator/spectral-wd-matched-tokenized"))
HF_REPO = "DimaTivator/effpretrain_ckpts"
PREFIX = "mode_connectivity_20m_20261103"
LABELS = ("l2", "swd05", "swd07", "swd1")
PAIRS = [a + "__" + b for a, b in itertools.combinations(LABELS, 2)]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish(paths: list[Path], verify: bool = False) -> None:
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    import tempfile
    operations = []
    hashes = {}
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.suffix not in (".json", ".jsonl", ".log"):
            continue
        if "checkpoints" in path.relative_to(ROOT).parts:
            raise RuntimeError("Checkpoint export forbidden")
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        operations.append(CommitOperationAdd(path_in_repo=PREFIX + "/" + relative, path_or_fileobj=content))
        hashes[relative] = hashlib.sha256(content).hexdigest()
    if not operations:
        return
    api = HfApi()
    commit = api.create_commit(repo_id=HF_REPO, operations=operations,
                               commit_message=PREFIX + " result telemetry")
    if verify:
        with tempfile.TemporaryDirectory(prefix="connectivity_verify_") as cache:
            for relative, expected in hashes.items():
                fetched = hf_hub_download(HF_REPO, PREFIX + "/" + relative, revision=commit.oid, cache_dir=cache)
                if hashlib.sha256(Path(fetched).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Result export checksum mismatch: " + relative)
    print("RESULT_EXPORT=" + json.dumps(dict(revision=commit.oid, files=len(operations), verified=verify)), flush=True)


def prepare(smoke: bool) -> Path:
    if smoke:
        import numpy as np
        data = ROOT / "smoke_tokens"
        data.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (np.arange(8192, dtype=np.uint16) % 128).tofile(data / (split + ".bin"))
        return data
    from connectivity_experiment import DATA_HASHES
    for name, expected in DATA_HASHES.items():
        digest = hashlib.sha256()
        with (DATA / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError("Training data checksum mismatch: " + name)
    return DATA


def stop(active: dict) -> list[dict]:
    stopped = []
    for gpu, (stage, process, log) in active.items():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for gpu, (stage, process, log) in active.items():
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        log.close()
        stopped.append(dict(stage=stage, pid=process.pid, exit_code=process.returncode))
    return stopped


def coordinate(lane: int, smoke: bool) -> None:
    enforce(ROOT, pending=450_000_000 if not smoke else 1_000_000, smoke=smoke)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / f"LANE_{lane}_STARTED.json").open("x") as stream:
        json.dump(dict(timestamp=time.time(), lane=lane, smoke=smoke), stream)
    data = prepare(smoke)
    endpoints = list(LABELS) if smoke else list(LABELS[lane*2:lane*2+2])
    pairs = list(PAIRS) if smoke else (PAIRS[:3] if lane == 0 else PAIRS[3:])
    pending = endpoints + pairs
    active = {}
    last_publish = 0.0
    try:
        while pending or active:
            record = space(ROOT, smoke=smoke)
            write_json(ROOT / f"disk_lane{lane}.json", record)
            print("DISK_GUARD=" + json.dumps(record), flush=True)
            if not record["ok"] or (ROOT / "ABORT.json").exists():
                raise RuntimeError("DISK_OR_CAMPAIGN_ABORT")
            for gpu, (stage, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                write_json(ROOT / (stage + "_EXIT.json"), dict(exit_code=code, pid=process.pid, timestamp=time.time()))
                if code or not (ROOT / stage / "final.json").is_file():
                    raise RuntimeError("Worker failed: " + stage + " code=" + str(code))
                write_json(ROOT / (stage + "_DONE.json"), dict(stage=stage, timestamp=time.time()))
            for gpu in range(1 if smoke else 2):
                if gpu in active:
                    continue
                eligible = next((stage for stage in pending if "__" not in stage or
                                 all((ROOT / (label + "_DONE.json")).exists() for label in LABELS)), None)
                if eligible is None:
                    continue
                pending.remove(eligible)
                folder = ROOT / eligible
                folder.mkdir(exist_ok=True)
                log = (folder / "worker.log").open("x")
                command = [sys.executable, __file__, "--worker", eligible]
                if smoke:
                    command.append("--smoke")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZED_DATA_DIR=str(data))
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (eligible, process, log)
                write_json(folder / "process.json", dict(pid=process.pid, gpu=gpu, lane=lane, command=command))
                print("STARTED=" + json.dumps(dict(stage=eligible, pid=process.pid, gpu=gpu)), flush=True)
            if not smoke and time.monotonic() - last_publish > 120:
                try:
                    selected = [ROOT / f"disk_lane{lane}.json"]
                    for stage in endpoints + pairs:
                        selected.extend((ROOT / stage).glob("*.json*"))
                        selected.append(ROOT / stage / "worker.log")
                    publish(selected)
                    last_publish = time.monotonic()
                except Exception as error:
                    print("EXPORT_RETRY=" + repr(error), flush=True)
            time.sleep(1 if smoke else 20)
        write_json(ROOT / f"LANE_{lane}_DONE.json", dict(timestamp=time.time(), lane=lane))
        if lane == 1 and not smoke:
            publish([ROOT / "LANE_1_DONE.json"])
            return
        while not smoke and not (ROOT / "LANE_1_DONE.json").exists():
            enforce(ROOT)
            if (ROOT / "ABORT.json").exists():
                raise RuntimeError("Peer lane aborted")
            time.sleep(20)
        results = {stage: json.loads((ROOT / stage / "final.json").read_text()) for stage in LABELS + tuple(PAIRS)}
        write_json(ROOT / "results.json", dict(stages=results, timestamp=time.time(),
                    code_revision=os.environ.get("CONNECTIVITY_REVISION"), data_hashes={} if smoke else __import__("connectivity_experiment").DATA_HASHES,
                    smoke=smoke, paired_initialization=True, parameter_count=results["l2"]["metadata"]["unique_parameters"]))
        if not smoke:
            publish(list(ROOT.rglob("*.json")) + list(ROOT.rglob("*.jsonl")), verify=True)
        deleted = cleanup_checkpoints(ROOT)
        remaining = list((ROOT / "checkpoints").iterdir())
        if remaining:
            raise RuntimeError("Checkpoint cleanup incomplete: " + repr(remaining))
        write_json(ROOT / "COMPLETE.json", dict(timestamp=time.time(), checkpoints_remaining=0, deleted=deleted,
                                               result_export_verified=not smoke, total_endpoints=4, total_paths=6))
        if not smoke:
            publish([ROOT / "COMPLETE.json"], verify=True)
        print("CAMPAIGN_COMPLETE checkpoints_remaining=0", flush=True)
    except BaseException as error:
        try:
            write_json(ROOT / "ABORT.json", dict(timestamp=time.time(), error=repr(error), lane=lane))
        except OSError as flag_error:
            print("ABORT_FLAG_WRITE_FAILED=" + repr(flag_error), flush=True)
        # Stopping tracked writers takes precedence over writing diagnostics.
        stopped = stop(active)
        record = space(ROOT, smoke=smoke)
        emergency = any(row.get("available_bytes", 10**30) < 5_000_000_000 for row in record["paths"])
        # Peer coordinator sees ABORT and stops its own exact child process groups.
        try:
            write_json(ROOT / f"LANE_{lane}_STOPPED.json", dict(stopped=stopped, timestamp=time.time()))
        except OSError as flag_error:
            print("STOPPED_FLAG_WRITE_FAILED=" + repr(flag_error), flush=True)
        if emergency:
            print("CRITICALLY LOW CLOUD.RU DISK SPACE: TRACKED WORKERS STOPPED. CLEANING THEIR CHECKPOINTS.", flush=True)
            if not smoke:
                peer = ROOT / f"LANE_{1-lane}_STOPPED.json"
                deadline = time.monotonic() + 60
                while not peer.exists() and time.monotonic() < deadline:
                    time.sleep(2)
                if not peer.exists():
                    print("PEER_STOP_UNCONFIRMED: OWN WORKERS STOPPED; PEER GUARD MUST STOP ITS WRITERS", flush=True)
            cleanup_checkpoints(ROOT, emergency=True)
        write_json(ROOT / f"FAILED_lane{lane}.json", dict(error=repr(error), stopped=stopped, storage=record, timestamp=time.time()))
        if not smoke:
            try:
                publish([ROOT / "ABORT.json", ROOT / f"FAILED_lane{lane}.json", ROOT / f"LANE_{lane}_STOPPED.json"], verify=True)
            except Exception as export_error:
                print("FAILURE_EXPORT_FAILED=" + repr(export_error), flush=True)
        raise


def worker(stage: str, smoke: bool) -> None:
    from connectivity_experiment import train_endpoint, train_path
    import torch
    torch.set_num_threads(2 if smoke else 4)
    if not smoke:
        torch.backends.cuda.matmul.allow_tf32 = True
    def watchdog() -> None:
        while True:
            try:
                enforce(ROOT, smoke=smoke)
                if (ROOT / "ABORT.json").exists():
                    raise RuntimeError("Peer coordinator aborted")
            except (OSError, RuntimeError) as error:
                print("WORKER_DISK_ABORT=" + repr(error), flush=True)
                os._exit(70)
            time.sleep(30)
    enforce(ROOT, smoke=smoke)
    threading.Thread(target=watchdog, daemon=True).start()
    function = train_path if "__" in stage else train_endpoint
    function(ROOT, stage, DATA, smoke)


def preflight() -> None:
    record = space(ROOT, pending=450_000_000)
    record["data_files"] = {name: (DATA / name).stat().st_size if (DATA / name).exists() else None for name in ("train.bin", "val.bin")}
    record["data_ready"] = all(record["data_files"].values())
    record["ok"] = record["ok"] and record["data_ready"]
    print("PREFLIGHT=" + json.dumps(record), flush=True)
    target = ROOT / "preflight.json"
    # Small diagnostic files only, including when storage is unavailable.
    write_json(target, record)
    publish([target], verify=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, choices=(0, 1), default=0)
    parser.add_argument("--worker", choices=LABELS + tuple(PAIRS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.worker:
        worker(args.worker, args.smoke)
    else:
        coordinate(args.lane, args.smoke)
