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
    hq, hkv, length, dim = validate_inputs(q, k, v, allow_empty=True)
    factor = attention_scale(dim, scale)
    q, k, v = (x.astype(mx.float32) for x in (q, k, v))
    if length == 0:
        return PartialAttention(
            mx.full((1, hq, 1, 1), -float("inf")),
            mx.zeros((1, hq, 1, 1)),
            mx.zeros((1, hq, 1, dim)),
        )
    groups = hq // hkv
    queries = q.reshape(1, hkv, groups, 1, dim)
    keys = mx.expand_dims(k, 2)
    scores = ((queries @ mx.swapaxes(keys, -1, -2)) * factor).reshape(1, hq, 1, length)
    maximum = mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores - maximum)
    numerator = weights.reshape(1, hkv, groups, 1, length) @ mx.expand_dims(v, 2)
    return PartialAttention(
        maximum,
        mx.sum(weights, axis=-1, keepdims=True),
        numerator.reshape(1, hq, 1, dim),
    )


def merge_partials(
    parts: Sequence[PartialAttention], *, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """Recover complete attention; local outputs must not simply be averaged."""
    if not parts:
        raise ValueError("At least one partial result is required")
    if dtype not in (mx.float32, mx.float16):
        raise ValueError("Output dtype must be float32 or float16")
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
    maximum = mx.max(maxima, axis=0)
    safe_maximum = mx.where(mx.isfinite(maximum), maximum, 0.0)
    factors = mx.where(mx.isfinite(maxima), mx.exp(maxima - safe_maximum), 0.0)
    denominator = mx.sum(factors * mx.stack([part.l for part in parts]), axis=0)
    numerator = mx.sum(factors * mx.stack([part.u for part in parts]), axis=0)
    # An exclusively empty collection is neutral; complete attention rejects N=0.
    return (numerator / mx.where(denominator > 0, denominator, 1.0)).astype(dtype)
