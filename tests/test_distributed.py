import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def launch(*arguments, timeout=30):
    """Bound native collectives and clean up the launcher and its children."""
    command = [
        str(Path(sys.executable).parent / "mlx.launch"),
        "--backend",
        "ring",
        "-n",
        "2",
        "--python",
        sys.executable,
        "--",
        "-m",
        "weftlm",
        *arguments,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("Two-process collective exceeded its timeout")
    assert process.returncode == 0, stdout + stderr
    return [
        json.loads(line[line.index("{") :])
        for line in stdout.splitlines()
        if "{" in line
    ]


def test_ring_probe_two_local_cpu_processes():
    rows = launch("probe", "--warmup", "1", "--repeats", "3")
    assert sorted(row["rank"] for row in rows) == [0, 1]
    for row in rows:
        assert row["passed"]
        assert row["world_size"] == 2
        assert row["payload_bytes_per_rank"] == 16640
        assert not row["wire_traffic_measured"]
        assert len(row["samples_ns"]) == 3
        assert all(sample > 0 for sample in row["samples_ns"])


def test_probe_rejects_singleton_and_invalid_payload():
    for arguments, message in [
        (["probe"], "MLX_HOSTFILE"),
        (["probe", "--payload-bytes", "3"], "multiple of four"),
    ]:
        result = subprocess.run(
            [sys.executable, "-m", "weftlm", *arguments],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert message in result.stderr
