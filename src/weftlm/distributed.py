"""Small, explicit MLX ring collectives for the two-participant prototype."""

import hashlib
import json

import mlx.core as mx

from .results import summarize_samples
from .timing import measure


def initialize_group():
    group = mx.distributed.init(strict=True, backend="ring")
    if group.size() != 2:
        raise RuntimeError("This prototype requires exactly two launched participants")
    return group


def agree_protocol(group, protocol: dict) -> None:
    """Compare fixed-size fingerprints before any shape-dependent collective."""
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).digest()
    with mx.stream(mx.cpu):
        packet = mx.array(list(digest), dtype=mx.uint8)
        gathered = mx.distributed.all_gather(packet, group=group, stream=mx.cpu)
        rows = gathered.reshape(group.size(), len(digest)).tolist()
    if any(row != rows[0] for row in rows[1:]):
        raise ValueError("Participants disagree on the operation protocol")


def probe_collectives(*, payload_bytes=16640, warmup=3, repeats=10) -> dict:
    if payload_bytes <= 0 or payload_bytes % 4:
        raise ValueError("Payload must be a positive multiple of four bytes")
    if warmup < 0 or repeats <= 0:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    group = initialize_group()
    agree_protocol(
        group,
        {"probe_bytes": payload_bytes, "warmup": warmup, "repeats": repeats},
    )
    elements = payload_bytes // 4
    iteration = -1
    payload = None

    def prepare():
        nonlocal iteration, payload
        iteration += 1
        payload = mx.full(
            (elements,), iteration * group.size() + group.rank(), dtype=mx.float32
        )
        mx.eval(payload)

    measurements = measure(
        lambda: mx.distributed.all_gather(payload, group=group, stream=mx.cpu),
        before_each=prepare,
        device=mx.cpu,
        warmup=warmup,
        repeats=repeats,
    )
    rows = measurements.last_output.reshape(group.size(), elements)
    expected = mx.arange(group.size())[:, None] + iteration * group.size()
    passed = mx.all(rows == expected).item()
    if not passed:
        raise RuntimeError("Collective probe returned incorrect values")
    return {
        "rank": group.rank(),
        "world_size": group.size(),
        "backend": "ring",
        "device": "cpu",
        "passed": passed,
        "payload_bytes_per_rank": payload_bytes,
        "logical_received_bytes_per_rank": payload_bytes * (group.size() - 1),
        "wire_traffic_measured": False,
        "samples_ns": measurements.samples_ns,
        **summarize_samples(measurements.samples_ns),
    }
