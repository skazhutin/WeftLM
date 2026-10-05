"""Complete distributed operations and separate component diagnostics."""

import json
import uuid
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from .baselines import optimized_attention
from .benchmark import error_metrics
from .distributed import (
    agree_protocol,
    distributed_context_attention,
    distributed_head_attention,
    merge_packets,
    pack_partial,
    partition_inputs,
)
from .fixtures import AttentionConfig, make_inputs
from .golden import load_reference
from .partial import blocked_partial_attention
from .reference import reference_attention
from .results import (
    environment_metadata,
    maximum_rank_samples,
    summarize_samples,
    write_results,
)
from .timing import measure


def barrier(group):
    mx.eval(mx.distributed.all_sum(mx.array([1]), group=group, stream=mx.cpu))


def gather_samples(samples, group):
    with mx.stream(mx.cpu):
        packet = mx.array([samples], dtype=mx.int64)
        return mx.distributed.all_gather(packet, group=group, stream=mx.cpu).tolist()


def benchmark_distributed(
    config: AttentionConfig,
    *,
    group,
    mode: str,
    output: Path,
    device=mx.gpu,
    warmup=10,
    repeats=50,
    reference_directory: Path | None = None,
    topology="unspecified",
    link_description="unspecified",
):
    if mode not in ("context", "heads"):
        raise ValueError("Distributed mode must be context or heads")
    if warmup < 0 or repeats <= 0:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    expected, reference_identity = None, "small-full-reference"
    if reference_directory is not None:
        expected, reference_identity = load_reference(reference_directory, config)
    elif config.length > 4096:
        raise ValueError(
            "Lengths above 4096 require --reference from a verified single run"
        )
    metadata = environment_metadata()
    agree_protocol(
        group,
        {
            "benchmark": asdict(config),
            "mode": mode,
            "device": str(device),
            "warmup": warmup,
            "repeats": repeats,
            "reference_identity": reference_identity,
            "mlx": metadata["mlx"],
            "source": metadata["source_fingerprint"],
            "topology": topology,
            "link": link_description,
        },
    )
    with mx.stream(mx.cpu):
        identifier = mx.array(list(uuid.uuid4().bytes), dtype=mx.uint8)
        identifiers = mx.distributed.all_gather(identifier, group=group, stream=mx.cpu)
        run_id = str(uuid.UUID(bytes=bytes(identifiers[:16].tolist())))
    (q, k, v), ownership = partition_inputs(config, group=group, mode=mode)

    def synchronize():
        barrier(group)

    operation = (
        (
            lambda: distributed_context_attention(
                q, k, v, group=group, block_size=config.block_size
            )
        )
        if mode == "context"
        else (lambda: distributed_head_attention(q, k, v, group=group))
    )
    measurements = measure(
        operation,
        device=device,
        warmup=warmup,
        repeats=repeats,
        before_each=synchronize,
    )
    participants = gather_samples(measurements.samples_ns, group)
    global_samples = maximum_rank_samples(participants)
    with mx.stream(device):
        if expected is None:
            expected = reference_attention(*make_inputs(config))
        error = error_metrics(measurements.last_output, expected, config.dtype)
    with mx.stream(mx.cpu):
        passed = (
            mx.distributed.all_sum(
                mx.array([int(error["passed"])]), group=group, stream=mx.cpu
            ).item()
            == group.size()
        )

    # Diagnostics run in separate series after total latency; they are not summed.
    diagnostics = {}
    if passed:
        local_operation = (
            (
                lambda: pack_partial(
                    blocked_partial_attention(q, k, v, block_size=config.block_size)
                )
            )
            if mode == "context"
            else (lambda: optimized_attention(q, k, v)[0])
        )
        local = measure(
            local_operation,
            device=device,
            warmup=warmup,
            repeats=repeats,
            before_each=synchronize,
        )
        communication = measure(
            lambda: mx.distributed.all_gather(
                local.last_output, group=group, stream=mx.cpu
            ),
            device=mx.cpu,
            warmup=warmup,
            repeats=repeats,
            before_each=synchronize,
        )
        assembly_operation = (
            (lambda: merge_packets(communication.last_output, dtype=q.dtype))
            if mode == "context"
            else (lambda: communication.last_output[None])
        )
        assembly = measure(
            assembly_operation,
            device=device,
            warmup=warmup,
            repeats=repeats,
            before_each=synchronize,
        )
        synchronization = measure(
            lambda: mx.distributed.all_sum(mx.array([1]), group=group, stream=mx.cpu),
            device=mx.cpu,
            warmup=warmup,
            repeats=repeats,
            before_each=synchronize,
        )
        for name, result in [
            ("local_compute", local),
            ("communication", communication),
            ("merge_or_assembly", assembly),
            ("barrier", synchronization),
        ]:
            samples = gather_samples(result.samples_ns, group)
            diagnostics[name] = {
                "participants_samples_ns": samples,
                "global_samples_ns": maximum_rank_samples(samples),
                "summary": summarize_samples(maximum_rank_samples(samples)),
            }
    payload_bytes = (
        config.hq * (config.dim + 2) * 4
        if mode == "context"
        else (config.hq // group.size()) * config.dim * q.itemsize
    )
    metadata.update(
        {
            "run_id": run_id,
            "config": asdict(config),
            "mode": mode,
            "rank": group.rank(),
            "world_size": group.size(),
            **ownership,
            "device": "gpu" if device == mx.gpu else "cpu",
            "backend": "ring",
            "topology": topology,
            "link_description": link_description,
            "data_kind": "synthetic",
            "fixture_version": 1,
            "warmup": warmup,
            "repeats": repeats,
            "error": error,
            "all_participants_passed": passed,
            "reference_identity": reference_identity,
            "peak_mlx_bytes": measurements.peak_mlx_bytes,
            "memory_scope": (
                "Full operation MLX peak; process RSS is lifetime "
                "including validation/diagnostics"
            ),
            "latency_scope": "max_participant_per_iteration",
            "barrier_scope": (
                "Before each operation, outside total timer; measured separately"
            ),
            "payload_bytes_per_rank": payload_bytes,
            "logical_received_bytes_per_rank": payload_bytes * (group.size() - 1),
            "wire_traffic_measured": False,
            "swap_after": environment_metadata()["swap"],
        }
    )
    summary = write_results(
        output, metadata, global_samples, per_rank_samples=participants
    )
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"Distributed correctness failed; artifacts: {output}")
    return summary
