"""Local softmax statistics and numerically stable attention merging."""

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

from .attention import attention_scale, validate_inputs


@dataclass(frozen=True)
class PartialAttention:
    """FP32 maximum/denominator [1,Hq,1,1] and numerator [1,Hq,1,D]."""

    m: mx.array
    l: mx.array  # noqa: E741 - denominator notation from the specification
    u: mx.array


def partial_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float | None = None
) -> PartialAttention:
    hq, hkv, _, dim = validate_inputs(q, k, v)
    if hq != hkv:
        raise ValueError("Local statistics currently require equal Q/KV head counts")
    scores = (q @ mx.swapaxes(k, -1, -2)) * attention_scale(dim, scale)
    maximum = mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores - maximum)
    return PartialAttention(
        maximum, mx.sum(weights, axis=-1, keepdims=True), weights @ v
    )


def merge_partials(parts: Sequence[PartialAttention]) -> mx.array:
    """Recover complete attention; local outputs must not simply be averaged."""
    if not parts:
        raise ValueError("At least one partial result is required")
    shape = parts[0].u.shape
    if len(shape) != 4 or shape[0] != 1 or shape[2] != 1 or min(shape) <= 0:
        raise ValueError("Partial numerators must have shape [1,Hq,1,D]")
    scalar_shape = (*shape[:-1], 1)
    for part in parts:
        if (
            part.u.shape != shape
            or part.m.shape != scalar_shape
            or part.l.shape != scalar_shape
        ):
            raise ValueError("All partial statistic shapes must match")
        if any(x.dtype != mx.float32 for x in (part.m, part.l, part.u)):
            raise ValueError("Partial statistics must use float32")
    maxima = mx.stack([part.m for part in parts])
    factors = mx.exp(maxima - mx.max(maxima, axis=0))
    denominator = mx.sum(factors * mx.stack([part.l for part in parts]), axis=0)
    numerator = mx.sum(factors * mx.stack([part.u for part in parts]), axis=0)
    return numerator / denominator
