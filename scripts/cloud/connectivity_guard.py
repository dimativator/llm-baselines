"""Fail-closed disk checks and narrowly scoped checkpoint cleanup."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

RESERVE = 5_000_000_000
PROBE_TIMEOUT = 5
ENDPOINTS = ("l2", "swd05", "swd07", "swd1")
CHECKPOINT_NAMES = frozenset(
    [name + ".pt" for name in ENDPOINTS]
    + [left + "__" + right + ".pt" for left, right in itertools.combinations(ENDPOINTS, 2)]
)
BOOTSTRAP = "/home/jovyan/shares/SR006.nfs3/mlsub-logs"

# Every filesystem operation, including resolving aliases and mount checks, runs
# in a disposable process: an unresponsive NFS mount must not hang the guard.
_PROBE = r'''
import json, os, pathlib, sys, time
requested, required_mount = sys.argv[1:3]
try:
    path = pathlib.Path(requested)
    while not path.exists() and path != path.parent:
        path = path.parent
    resolved = path.resolve(strict=True)
    required = pathlib.Path(required_mount).resolve(strict=True) if required_mount else None
    if required is not None and not required.is_mount():
        raise OSError("Required filesystem is not mounted: " + str(required))
    device = os.stat(resolved).st_dev
    if required is not None and device != os.stat(required).st_dev:
        raise OSError("Path is not on its required filesystem")
    info = os.statvfs(resolved)
    mount = resolved
    while not mount.is_mount() and mount != mount.parent:
        mount = mount.parent
    print(json.dumps(dict(path=requested, measured_path=str(resolved),
        mount=str(mount), device=device, available_bytes=info.f_bavail * info.f_frsize,
        timestamp=time.time())))
except (OSError, RuntimeError) as error:
    print(json.dumps(dict(path=requested, timestamp=time.time(), error=str(error))))
'''


def _probe(path: str, required_mount: str = "") -> dict:
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _PROBE, path, required_mount],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        stdout, _ = process.communicate(timeout=PROBE_TIMEOUT)
        if process.returncode:
            raise OSError("Filesystem probe exited " + str(process.returncode))
        record = json.loads(stdout)
        if not isinstance(record, dict) or record.get("path") != path:
            raise ValueError("Invalid filesystem probe response")
        return record
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        if process is not None and process.poll() is None:
            # Do not call unbounded wait/communicate: a process blocked on an
            # unavailable NFS mount can remain in uninterruptible kernel sleep.
            process.kill()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        return dict(path=path, timestamp=time.time(), error=str(error))


def space(root: Path, pending: int = 0, smoke: bool = False) -> dict:
    """Measure exact available bytes; pending conservatively applies to each FS."""
    if pending < 0:
        raise ValueError("pending bytes must be nonnegative")
    paths = [(str(root), "" if smoke else "/workspace-SR006.nfs3"),
             ("/", ""), ("/tmp", "")]
    if not smoke:
        paths += [("/home/jovyan", ""),
                  (BOOTSTRAP, "/workspace-SR006.nfs3"),
                  ("/workspace-SR006.nfs2", "/workspace-SR006.nfs2")]
    records = []
    for path, mount in paths:
        record = _probe(path, mount)
        available = record.get("available_bytes")
        record["ok"] = (
            "error" not in record and isinstance(available, int)
            and available > RESERVE + pending
        )
        records.append(record)
    return dict(timestamp=time.time(), paths=records, pending_bytes=pending,
                reserve_bytes=RESERVE, ok=all(row["ok"] for row in records))


def enforce(root: Path, pending: int = 0, smoke: bool = False) -> dict:
    record = space(root, pending=pending, smoke=smoke)
    print("DISK_GUARD=" + json.dumps(record, allow_nan=False), flush=True)
    if not record["ok"]:
        raise RuntimeError("DISK_RESERVE_UNAVAILABLE " + json.dumps(record))
    return record


def cleanup_checkpoints(root: Path, emergency: bool = False) -> list[dict]:
    """Delete only the ten known direct regular checkpoint files, oldest first.

    The caller stops its exact tracked workers before emergency cleanup. Logs,
    configs, metrics, unknown files, symlinks, and nested files are untouched.
    Each unlink is relative to an opened directory descriptor with O_NOFOLLOW.
    """
    root = Path(root)
    if root.is_symlink():
        raise RuntimeError("Refusing symlink campaign root")
    folder = root / "checkpoints"
    if folder.is_symlink():
        raise RuntimeError("Refusing symlink checkpoint directory")
    if not folder.exists():
        return []
    if emergency:
        initial = space(root)
        print("EMERGENCY_CLEANUP_DISK=" + json.dumps(initial), flush=True)
        if initial["ok"]:
            return []
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root, flags)
    try:
        folder_fd = os.open("checkpoints", flags, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    deleted = []
    try:
        candidates = []
        for name in os.listdir(folder_fd):
            if name not in CHECKPOINT_NAMES:
                continue
            info = os.stat(name, dir_fd=folder_fd, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                candidates.append((info.st_mtime_ns, name, info.st_ino))
        for _, name, inode in sorted(candidates):
            before = _probe(str(root))
            current = os.stat(name, dir_fd=folder_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or current.st_ino != inode or current.st_nlink != 1:
                raise RuntimeError("Checkpoint changed during cleanup: " + name)
            os.unlink(name, dir_fd=folder_fd)
            after = _probe(str(root))
            row = dict(timestamp=time.time(), deleted_path=str(folder / name),
                       size_bytes=current.st_size, before=before, after=after)
            deleted.append(row)
            print("CHECKPOINT_DELETED=" + json.dumps(row), flush=True)
            if emergency:
                storage = space(root)
                row["all_filesystems_after"] = storage
                print("EMERGENCY_CLEANUP_DISK=" + json.dumps(storage), flush=True)
                if storage["ok"]:
                    break
    finally:
        os.close(folder_fd)
    return deleted
