import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def launch(*arguments, timeout=30, program=("-m", "weftlm")):
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
        *program,
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


@pytest.mark.parametrize(
    "length,dtype", [(1, "float32"), (7, "float16"), (9, "float32")]
)
def test_context_attention_both_ranks_match_independent_reference(length, dtype):
    rows = launch(
        "check-distributed",
        "--device",
        "cpu",
        "--length",
        str(length),
        "--hq",
        "4",
        "--hkv",
        "2",
        "--dim",
        "8",
        "--block-size",
        "2",
        "--dtype",
        dtype,
    )
    assert sorted(row["rank"] for row in rows) == [0, 1]
    for row in rows:
        assert row["error"]["passed"]
        assert row["error"]["finite"]
        assert row["output_shape"] == [1, 4, 1, 8]
        rank = row["rank"]
        assert row["token_range"] == [length * rank // 2, length * (rank + 1) // 2]


def test_mismatched_protocol_is_rejected_on_both_ranks():
    code = """
import json
from weftlm.distributed import initialize_group, agree_protocol
group = initialize_group()
try:
    agree_protocol(group, {"length": 7 + group.rank()})
except ValueError:
    print(json.dumps({"rank": group.rank(), "rejected": True}), flush=True)
else:
    raise AssertionError("Mismatched protocol was accepted")
"""
    rows = launch(program=("-c", code))
    assert sorted(row["rank"] for row in rows if row["rejected"]) == [0, 1]
