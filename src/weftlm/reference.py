"""Independent full-attention reference for small correctness fixtures."""

import mlx.core as mx

from .attention import attention_scale, validate_inputs


def reference_attention(
    q: mx.array, k: mx.array, v: mx.array, *, scale: float | None = None
) -> mx.array:
    """Compute complete attention directly, without shard statistics or merging.

    Inputs are [1, Hq, 1, D] and [1, Hkv, N, D]. All KV tokens are
    visible. Query heads in consecutive groups share one KV head.
    """
    hq, hkv, _, dim = validate_inputs(q, k, v)
    dtype = q.dtype
    q, k, v = (x.astype(mx.float32) for x in (q, k, v))
    factor = attention_scale(dim, scale)
    group_size = hq // hkv
    outputs = []
    for head in range(hq):
        kv_head = head // group_size
        scores = (q[0, head, 0] @ k[0, kv_head].T) * factor
        weights = mx.softmax(scores, axis=-1)
        outputs.append(weights @ v[0, kv_head])
    return mx.stack(outputs).reshape(1, hq, 1, dim).astype(dtype)
