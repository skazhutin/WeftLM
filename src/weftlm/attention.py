"""Input contract for the single-query decode experiments."""

import math

import mlx.core as mx


def validate_inputs(
    q: mx.array, k: mx.array, v: mx.array, *, allow_empty: bool = False
) -> tuple[int, int, int, int]:
    """Validate [1, Hq, 1, D] queries and [1, Hkv, N, D] keys/values."""
    if not all(isinstance(x, mx.array) for x in (q, k, v)):
        raise TypeError("Q, K and V must be MLX arrays")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q, K and V must have four dimensions")
    if any(x.shape[0] != 1 for x in (q, k, v)) or q.shape[2] != 1:
        raise ValueError("Only batch size 1 and one query token are supported")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("K/V shapes and Q/K/V head dimensions must match")
    hq, hkv, length, dim = q.shape[1], k.shape[1], k.shape[2], q.shape[-1]
    if min(hq, hkv, dim) <= 0 or hq % hkv:
        raise ValueError(
            "Head counts and dimension must be positive; Hq must divide by Hkv"
        )
    if length == 0 and not allow_empty:
        raise ValueError("The complete KV context must not be empty")
    if q.dtype not in (mx.float32, mx.float16) or any(
        x.dtype != q.dtype for x in (k, v)
    ):
        raise ValueError("Q, K and V must use the same float32 or float16 dtype")
    return hq, hkv, length, dim


def attention_scale(dim: int, scale: float | None) -> float:
    """Resolve the usual attention scale without evaluating array data."""
    value = dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(value):
        raise ValueError("Attention scale must be finite")
    return value
