"""Stream existing Sigma campaign logs to W&B without restarting training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from sigma_campaign import ROOT, enforce, space, write_json

STAGES = {"reference": "swd26ref", "spectral": "swd26spec", "sigma_1000": "swd26s1k",
          "sigma_3000": "swd26s3k", "sigma_5000": "swd26s5k"}
TRAIN = re.compile(r"Train: Iter=(\d+).*?train_loss=([\d.eE+-]+) iter_dt=([\d.eE+-]+)s lr=([\d.eE+-]+)")
VALIDATION = re.compile(r">Eval: Iter=(\d+).*?val_loss=([\d.eE+-]+)")


def parse_line(line: str, pending: dict) -> dict | None:
    if "VALIDATION=" in line:
        pending["native_val"] = json.loads(line.split("VALIDATION=", 1)[1])
    match = TRAIN.search(line)
    if match:
        step, loss, duration, lr = match.groups()
        return {"iter": int(step), "train/loss": float(loss), "lr": float(lr),
                "train/step_seconds": float(duration), "tokens": int(step) * 128 * 1024}
    match = VALIDATION.search(line)
    if match:
        step, rounded_loss = match.groups()
        native = pending.pop("native_val", {})
        return {"iter": int(step), "val/loss": native.get("val_loss", float(rounded_loss)),
                "source/validation_timestamp": native.get("timestamp", 0)}
    if "PROGRESS=" in line:
        value = json.loads(line.split("PROGRESS=", 1)[1])
        return {"iter": value["step"], "memory/peak_gpu_bytes": value["peak_gpu_bytes"]}
    if "DISK=" in line:
        value = json.loads(line.split("DISK=", 1)[1])
        result = {"source/disk_timestamp": value["timestamp"], "disk/guard_ok": value["ok"]}
        for row in value["paths"]:
            if "available_bytes" in row:
                result["disk/free_bytes/" + row["path"].strip("/").replace("/", "_")] = row["available_bytes"]
        return result
    return None


def stage_relay(stage: str) -> None:
    import wandb

    logfile = ROOT / (stage + ".log")
    while not logfile.exists():
        enforce()
        if (ROOT / "FAILED.json").exists():
            return
        time.sleep(30)
    enforce()
    local = Path("/tmp/sigma-wandb") / stage
    local.mkdir(parents=True, exist_ok=True)
    run = wandb.init(entity="andrey", project="ns_weights", id=STAGES[stage], resume="allow",
                     name="sigma124m_20261102_" + stage, group="sigma-only-124m-20261102",
                     job_type="sigma-log-relay", dir=str(local),
                     config={"stage": stage, "model": "llama124m", "lr": 0.005,
                             "iterations": int(stage.split("_")[1]) if stage.startswith("sigma_") else 19000,
                             "weight_decay": 0.1 if stage == "reference" else 0,
                             "spectral_pretraining_coefficient": 1, "global_batch": 128,
                             "sequence_length": 1024, "seed": 0, "data_seed": 1337,
                             "sigma_schedule": "cosine_to_zero_without_warmup",
                             "training_job": "lm-mpi-job-39e0357e-9f2b-4454-b724-bdfe1716dce3",
                             "source": "persistent training logs, backfilled then streamed",
                             "train_loss_precision": "three decimals from original console",
                             "validation_loss_precision": "native recorded float"})
    run.define_metric("iter")
    for metric in ("train/*", "val/*", "memory/*", "lr", "tokens"):
        run.define_metric(metric, step_metric="iter")
    status = dict(url=run.url, id=run.id, stage=stage, timestamp=time.time())
    write_json(ROOT / stage / "wandb_relay.json", status)
    print("WANDB_READY=" + json.dumps(status), flush=True)
    cursor = 0
    pending = {}
    last_step = 0
    while True:
        enforce()
        with logfile.open(errors="replace") as stream:
            stream.seek(cursor)
            while True:
                start = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith("\n"):
                    stream.seek(start)
                    break
                for line in raw.replace("\r", "\n").splitlines():
                    value = parse_line(line, pending)
                    if value is not None:
                        last_step = value.get("iter", last_step)
                        run.log(value)
            cursor = stream.tell()
        run.log({"relay/heartbeat": time.time(), "source/log_mtime": logfile.stat().st_mtime,
                 "relay/last_training_step": last_step})
        final = ROOT / stage / "final.json"
        if final.exists() and (ROOT / (stage + "_DONE.json")).exists():
            payload = json.loads(final.read_text())
            transformer = [row for row in payload["ranks"] if row["transformer"]]
            denominator = sum(row["full_rank"] for row in transformer)
            summary = {"final/val_loss": payload["val_loss"], "final/perplexity": payload["perplexity"],
                       "final/iterations": payload["steps"], "final/verified": True,
                       "rank/weighted_effective_rank": sum(row["effective_rank"] * row["full_rank"] for row in transformer) / denominator,
                       "rank/weighted_rank_ratio": sum(row["effective_rank"] for row in transformer) / denominator}
            run.summary.update(summary)
            run.log({**summary, "iter": payload["steps"], "rank/per_matrix": wandb.Table(
                columns=["matrix", "effective_rank", "full_rank", "transformer"],
                data=[[row["name"], row["effective_rank"], row["full_rank"], row["transformer"]] for row in payload["ranks"]])})
            artifact = wandb.Artifact("sigma124m-20261102-" + stage, type="metrics")
            artifact.add_file(str(final), name="final.json")
            run.log_artifact(artifact)
            run.finish()
            write_json(ROOT / stage / "wandb_relay_done.json", dict(**status, finished=time.time()))
            return
        if (ROOT / "FAILED.json").exists():
            run.summary["campaign_failure"] = json.loads((ROOT / "FAILED.json").read_text())
            run.finish(exit_code=1)
            return
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.stage:
        stage_relay(args.stage)
        return
    enforce()
    with (ROOT / "wandb_relay.lock").open("x") as stream:
        json.dump(dict(pid=os.getpid(), timestamp=time.time()), stream)
    processes = [subprocess.Popen([sys.executable, __file__, "--stage", stage]) for stage in STAGES]
    while any(process.poll() is None for process in processes):
        if not space()["ok"]:
            print("CRITICALLY LOW CLOUD.RU DISK SPACE: STOPPING W&B RELAY", flush=True)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
            raise RuntimeError("Disk reserve violated")
        time.sleep(30)
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError("W&B relay workers failed: " + str(codes))


if __name__ == "__main__":
    main()
