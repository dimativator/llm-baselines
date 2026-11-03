"""Disk reserve boundaries, telemetry failures, and checkpoint cleanup safety."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/cloud/connectivity_guard.py"
SPEC = importlib.util.spec_from_file_location("connectivity_guard", SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def telemetry(path, mount="", available=5_000_000_001):
    return dict(path=path, mount=mount or "/", device=7,
                available_bytes=available, timestamp=1.0)


class SpaceTests(unittest.TestCase):
    def test_exact_reserve_and_pending_write_boundaries(self):
        for available, pending, expected in (
            (guard.RESERVE, 0, False),
            (guard.RESERVE + 1, 0, True),
            (guard.RESERVE + 100, 100, False),
            (guard.RESERVE + 101, 100, True),
        ):
            with self.subTest(available=available, pending=pending):
                with patch.object(guard, "_probe", side_effect=lambda p, m: telemetry(p, m, available)):
                    result = guard.space(Path("/campaign"), pending, smoke=True)
                self.assertEqual(result["ok"], expected)

    def test_cloud_includes_nfs2_and_bootstrap_even_when_devices_alias(self):
        with patch.object(guard, "_probe", side_effect=telemetry) as probe:
            result = guard.space(Path("/workspace-SR006.nfs3/user/campaign"))
        self.assertTrue(result["ok"])
        paths = {row["path"] for row in result["paths"]}
        self.assertIn("/workspace-SR006.nfs2", paths)
        self.assertIn(guard.BOOTSTRAP, paths)
        self.assertIn("/home/jovyan", paths)
        self.assertEqual(probe.call_count, 6)
        self.assertTrue(all(row["device"] == 7 for row in result["paths"]))

    def test_one_unavailable_filesystem_fails_closed(self):
        def probe(path, mount=""):
            return dict(path=path, error="NFS timed out") if "nfs2" in path else telemetry(path, mount)
        with patch.object(guard, "_probe", side_effect=probe), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "DISK_RESERVE_UNAVAILABLE"):
                guard.enforce(Path("/campaign"))

    def test_negative_pending_is_rejected(self):
        with self.assertRaises(ValueError):
            guard.space(Path("/campaign"), -1)

    def test_probe_timeout_never_waits_unbounded(self):
        process = Mock(returncode=None, stdout=Mock(), stderr=Mock())
        process.communicate.side_effect = subprocess.TimeoutExpired("probe", 5)
        process.poll.return_value = None
        with patch.object(guard.subprocess, "Popen", return_value=process):
            result = guard._probe("/stalled")
        self.assertIn("error", result)
        process.kill.assert_called_once_with()
        process.communicate.assert_called_once_with(timeout=guard.PROBE_TIMEOUT)
        process.wait.assert_not_called()

    def test_probe_preserves_exact_integer_bytes(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (json.dumps(telemetry("/root", available=5_000_000_003)), "")
        with patch.object(guard.subprocess, "Popen", return_value=process):
            result = guard._probe("/root")
        self.assertEqual(result["available_bytes"], 5_000_000_003)


class CleanupTests(unittest.TestCase):
    def test_only_known_regular_direct_files_deleted_oldest_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "checkpoints"
            folder.mkdir()
            (folder / "swd1.pt").write_bytes(b"newer")
            (folder / "l2.pt").write_bytes(b"older")
            os.utime(folder / "l2.pt", ns=(1, 1))
            (folder / "unknown.pt").write_bytes(b"preserve")
            (root / "metrics.jsonl").write_text("preserve\n")
            nested = folder / "nested"
            nested.mkdir()
            (nested / "swd05.pt").write_bytes(b"preserve")
            outside = root / "outside.pt"
            outside.write_bytes(b"preserve")
            (folder / "swd05.pt").symlink_to(outside)
            os.link(outside, folder / "swd07.pt")
            with patch.object(guard, "_probe", side_effect=telemetry), contextlib.redirect_stdout(io.StringIO()):
                deleted = guard.cleanup_checkpoints(root)
            self.assertEqual([Path(row["deleted_path"]).name for row in deleted], ["l2.pt", "swd1.pt"])
            self.assertTrue((folder / "unknown.pt").exists())
            self.assertTrue((nested / "swd05.pt").exists())
            self.assertTrue((folder / "swd05.pt").is_symlink())
            self.assertTrue((folder / "swd07.pt").exists())
            self.assertEqual(outside.read_bytes(), b"preserve")
            self.assertTrue(all("before" in row and "after" in row for row in deleted))

    def test_symlink_checkpoint_directory_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "unrelated"
            target.mkdir()
            (root / "checkpoints").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink checkpoint"):
                guard.cleanup_checkpoints(root)

    def test_symlink_root_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real"
            target.mkdir()
            alias = root / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink campaign"):
                guard.cleanup_checkpoints(alias)

    def test_missing_checkpoint_directory_is_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(guard.cleanup_checkpoints(Path(temporary)), [])

    def test_emergency_stops_when_all_filesystems_recover(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "checkpoints"
            folder.mkdir()
            for name in ("l2.pt", "swd05.pt", "swd07.pt"):
                (folder / name).write_bytes(b"checkpoint")
            os.utime(folder / "l2.pt", ns=(1, 1))
            os.utime(folder / "swd05.pt", ns=(2, 2))
            measurements = [dict(ok=False), dict(ok=False), dict(ok=True)]
            with patch.object(guard, "space", side_effect=measurements) as space:
                with patch.object(guard, "_probe", side_effect=telemetry), contextlib.redirect_stdout(io.StringIO()):
                    deleted = guard.cleanup_checkpoints(root, emergency=True)
            self.assertEqual(len(deleted), 2)
            self.assertTrue((folder / "swd07.pt").exists())
            self.assertEqual(space.call_count, 3)

    def test_emergency_keeps_checkpoints_when_reserve_already_sufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "checkpoints"
            folder.mkdir()
            (folder / "l2.pt").write_bytes(b"checkpoint")
            with patch.object(guard, "space", return_value=dict(ok=True)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(guard.cleanup_checkpoints(root, emergency=True), [])
            self.assertTrue((folder / "l2.pt").exists())


if __name__ == "__main__":
    unittest.main()
