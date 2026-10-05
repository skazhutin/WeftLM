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


def combine_partials(parts: Sequence[PartialAttention]) -> PartialAttention:
    """Combine statistics without normalizing, so they can be merged again."""
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
    maximum = mx.max(maxima, axis=0)
    safe_maximum = mx.where(mx.isfinite(maximum), maximum, 0.0)
    factors = mx.where(mx.isfinite(maxima), mx.exp(maxima - safe_maximum), 0.0)
    denominator = mx.sum(factors * mx.stack([part.l for part in parts]), axis=0)
    numerator = mx.sum(factors * mx.stack([part.u for part in parts]), axis=0)
    return PartialAttention(maximum, denominator, numerator)


def merge_partials(
    parts: Sequence[PartialAttention], *, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """Recover attention; an exclusively empty collection produces zeros."""
    if dtype not in (mx.float32, mx.float16):
        raise ValueError("Output dtype must be float32 or float16")
    part = combine_partials(parts)
    return (part.u / mx.where(part.l > 0, part.l, 1.0)).astype(dtype)


def blocked_partial_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    block_size: int = 4096,
    scale: float | None = None,
) -> PartialAttention:
    """Bound FP32 conversions and logits to one KV block at a time."""
    _, _, length, _ = validate_inputs(q, k, v, allow_empty=True)
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("Block size must be a positive integer")
    if length == 0:
        return partial_attention(q, k, v, scale=scale)
    combined = None
    for start in range(0, length, block_size):
        part = partial_attention(
            q,
            k[:, :, start : start + block_size],
            v[:, :, start : start + block_size],
            scale=scale,
        )
        combined = part if combined is None else combine_partials([combined, part])
        # Release evaluated block graphs instead of retaining all conversion buffers.
        mx.eval(combined.m, combined.l, combined.u)
    return combined


def context_attention(
    q: mx.array, k: mx.array, v: mx.array, *, block_size: int = 4096
) -> mx.array:
    """Complete one-device blocked attention; no distributed speed claim."""
    validate_inputs(q, k, v)
    return merge_partials(
        [blocked_partial_attention(q, k, v, block_size=block_size)], dtype=q.dtype
    )
