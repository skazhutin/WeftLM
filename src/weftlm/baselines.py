"""Optimized MLX attention controls, separate from the correctness reference."""

import mlx.core as mx

from .attention import attention_scale, validate_inputs


def optimized_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Use MLX's normal kernel selection and native GQA, without KV tiling."""
    _, _, _, dim = validate_inputs(q, k, v)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=attention_scale(dim, None)
    )
