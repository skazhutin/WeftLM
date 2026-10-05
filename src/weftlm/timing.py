"""Synchronous end-to-end timing of fresh MLX operations."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns

import mlx.core as mx


@dataclass(frozen=True)
class Measurements:
    samples_ns: list[int]
    peak_mlx_bytes: int
    last_output: mx.array


def measure(
    operation: Callable[[], mx.array],
    *,
    warmup: int = 10,
    repeats: int = 50,
    device: mx.DeviceType = mx.gpu,
    before_each: Callable[[], None] | None = None,
) -> Measurements:
    """Inputs must already be evaluated. Preparation/barriers are not timed.

    The callable is invoked again every iteration. Its graph construction,
    evaluation and completion synchronization are all inside the timer.
    """
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Warmup must be a nonnegative integer")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("Repeats must be a positive integer")
    samples = []
    with mx.stream(device):
        for _ in range(warmup):
            if before_each is not None:
                before_each()
            output = operation()
            mx.eval(output)
            mx.synchronize(device)
        mx.reset_peak_memory()
        for _ in range(repeats):
            mx.synchronize(device)
            if before_each is not None:
                before_each()
            start = perf_counter_ns()
            output = operation()
            mx.eval(output)
            mx.synchronize(device)
            samples.append(perf_counter_ns() - start)
    return Measurements(samples, mx.get_peak_memory(), output)
