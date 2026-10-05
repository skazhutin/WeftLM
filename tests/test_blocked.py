import mlx.core as mx
import pytest

from weftlm.partial import blocked_partial_attention, context_attention, merge_partials
from weftlm.reference import reference_attention


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("block_size", [1, 4, 7, 4096])
def test_blocked_attention_matches_reference(dtype, block_size):
    q = mx.random.normal((1, 8, 1, 16), key=mx.random.key(4)).astype(dtype)
    k = mx.random.normal((1, 2, 13, 16), key=mx.random.key(5)).astype(dtype)
    v = mx.random.normal((1, 2, 13, 16), key=mx.random.key(6)).astype(dtype)
    out = context_attention(q, k, v, block_size=block_size)
    tol = (
        dict(rtol=1e-4, atol=1e-5)
        if dtype == mx.float32
        else dict(rtol=1e-2, atol=1e-3)
    )
    assert mx.allclose(out, reference_attention(q, k, v), **tol).item()


def test_empty_blocked_shard_is_neutral():
    q = mx.zeros((1, 4, 1, 8))
    k = v = mx.zeros((1, 1, 0, 8))
    assert mx.all(merge_partials([blocked_partial_attention(q, k, v)]) == 0).item()
    with pytest.raises(ValueError, match="empty"):
        context_attention(q, k, v)


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_invalid_block_size(block_size):
    arrays = [mx.zeros((1, 1, 1, 2)) for _ in range(3)]
    with pytest.raises(ValueError, match="positive integer"):
        blocked_partial_attention(*arrays, block_size=block_size)
