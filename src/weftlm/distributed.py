"""Small, explicit MLX ring collectives for the two-participant prototype."""

import hashlib
import json

import mlx.core as mx

from .baselines import optimized_attention
from .fixtures import AttentionConfig, make_inputs
from .partial import PartialAttention, blocked_partial_attention, merge_partials
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


def pack_partial(part: PartialAttention) -> mx.array:
    return mx.concatenate([part.m, part.l, part.u], axis=-1)


def merge_packets(packets: mx.array, *, dtype: mx.Dtype) -> mx.array:
    if packets.ndim != 4 or packets.shape[2] != 1 or packets.shape[-1] < 3:
        raise ValueError("Statistic packets must have shape [world,Hq,1,D+2]")
    parts = [
        PartialAttention(
            packets[rank : rank + 1, :, :, :1],
            packets[rank : rank + 1, :, :, 1:2],
            packets[rank : rank + 1, :, :, 2:],
        )
        for rank in range(packets.shape[0])
    ]
    return merge_partials(parts, dtype=dtype)


def distributed_context_attention(q, k, v, *, group, block_size=4096):
    """Local KV only; gather FP32 statistics and recover the full output."""
    local = blocked_partial_attention(q, k, v, block_size=block_size)
    packets = mx.distributed.all_gather(pack_partial(local), group=group, stream=mx.cpu)
    return merge_packets(packets, dtype=q.dtype)


def distributed_head_attention(q, k, v, *, group):
    """Optimized local head groups followed by complete output collection."""
    local = optimized_attention(q, k, v)
    return mx.distributed.all_gather(local[0], group=group, stream=mx.cpu)[None]


def partition_inputs(config: AttentionConfig, *, group, mode: str):
    rank, size = group.rank(), group.size()
    if mode == "context":
        start, end = config.length * rank // size, config.length * (rank + 1) // size
        return make_inputs(config, token_range=(start, end)), {
            "token_range": [start, end],
            "kv_head_range": [0, config.hkv],
            "query_head_range": [0, config.hq],
        }
    if mode == "heads":
        if config.hkv % size:
            raise ValueError("Head splitting requires Hkv divisible by world size")
        first, last = config.hkv * rank // size, config.hkv * (rank + 1) // size
        groups = config.hq // config.hkv
        return make_inputs(config, head_range=(first, last)), {
            "token_range": [0, config.length],
            "kv_head_range": [first, last],
            "query_head_range": [first * groups, last * groups],
        }
    raise ValueError("Distributed mode must be context or heads")


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
