"""Local attention measurements; synthetic inputs, no model weights."""

from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from .baselines import optimized_attention
from .fixtures import AttentionConfig, make_inputs
from .golden import write_reference
from .partial import context_attention
from .reference import reference_attention
from .results import environment_metadata, write_results
from .timing import measure


def error_metrics(actual: mx.array, expected: mx.array, dtype: str) -> dict:
    atol, rtol = (1e-5, 1e-4) if dtype == "float32" else (1e-3, 1e-2)
    difference = actual.astype(mx.float32) - expected.astype(mx.float32)
    finite = mx.all(mx.isfinite(difference)).item()
    return {
        "max_abs": mx.max(mx.abs(difference)).item() if finite else None,
        "rmse": mx.sqrt(mx.mean(difference * difference)).item() if finite else None,
        "finite": finite,
        "atol": atol,
        "rtol": rtol,
        "passed": mx.allclose(actual, expected, atol=atol, rtol=rtol).item(),
    }


def benchmark_local(
    config: AttentionConfig,
    *,
    mode: str,
    output: Path,
    device: mx.DeviceType = mx.gpu,
    warmup: int = 10,
    repeats: int = 50,
) -> dict:
    if mode not in ("single", "local-context"):
        raise ValueError("Local mode must be single or local-context")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metadata = environment_metadata()
    q, k, v = make_inputs(config)
    with mx.stream(device):
        operation = (
            (lambda: optimized_attention(q, k, v))
            if mode == "single"
            else (lambda: context_attention(q, k, v, block_size=config.block_size))
        )
        measurements = measure(operation, warmup=warmup, repeats=repeats, device=device)
        # Independent numerical verification is deliberately outside the timer.
        expected = reference_attention(q, k, v)
        error = error_metrics(measurements.last_output, expected, config.dtype)
    metadata.update(
        {
            "config": asdict(config),
            "mode": mode,
            "rank": 0,
            "world_size": 1,
            "device": "gpu" if device == mx.gpu else "cpu",
            "data_kind": "synthetic",
            "fixture_version": 1,
            "warmup": warmup,
            "repeats": repeats,
            "error": error,
            "peak_mlx_bytes": measurements.peak_mlx_bytes,
            "memory_scope": "MLX operation peak; process RSS is lifetime peak",
            "swap_after": environment_metadata()["swap"],
        }
    )
    summary = write_results(output, metadata, measurements.samples_ns)
    if not error["passed"]:
        raise RuntimeError(f"Attention correctness check failed; artifacts: {output}")
    if mode == "single":
        write_reference(output, config, expected)
    return summary
