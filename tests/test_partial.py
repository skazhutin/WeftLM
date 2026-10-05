import mlx.core as mx
import pytest

from weftlm.partial import PartialAttention, merge_partials, partial_attention
from weftlm.reference import reference_attention


def fixture_arrays(hq=2, hkv=2, length=12, dim=8, seed=0, dtype=mx.float32):
    q = mx.random.normal((1, hq, 1, dim), key=mx.random.key(seed)).astype(dtype)
    k = mx.random.normal((1, hkv, length, dim), key=mx.random.key(seed + 1)).astype(
        dtype
    )
    v = mx.random.normal((1, hkv, length, dim), key=mx.random.key(seed + 2)).astype(
        dtype
    )
    return q, k, v


@pytest.mark.parametrize("seed", range(5))
def test_two_equal_shards(seed):
    q, k, v = fixture_arrays(seed=seed)
    parts = [
        partial_attention(q, k[:, :, start : start + 6], v[:, :, start : start + 6])
        for start in (0, 6)
    ]
    actual = merge_partials(parts)
    assert mx.allclose(
        actual, reference_attention(q, k, v), rtol=1e-4, atol=1e-5
    ).item()


def test_zero_query_statistics():
    q, k, v = fixture_arrays()
    part = partial_attention(mx.zeros_like(q), k, v)
    assert part.m.tolist() == [[[[0.0]], [[0.0]]]]
    assert mx.all(part.l == 12).item()
    assert mx.allclose(part.u, mx.sum(v, axis=2, keepdims=True)).item()


def test_merge_requires_partials():
    with pytest.raises(ValueError, match="At least"):
        merge_partials([])


def test_merge_rejects_mismatched_shapes():
    q, k, v = fixture_arrays()
    part = partial_attention(q, k, v)
    bad = PartialAttention(part.m, part.l, mx.zeros((1, 3, 1, 8)))
    with pytest.raises(ValueError, match="shapes"):
        merge_partials([part, bad])


@pytest.mark.parametrize("hq,hkv", [(1, 1), (4, 2), (8, 1), (32, 8)])
@pytest.mark.parametrize("cuts", [(0, 3, 12), (0, 0, 1, 5, 7, 12, 12)])
def test_unequal_and_empty_gqa_shards(hq, hkv, cuts):
    q, k, v = fixture_arrays(hq=hq, hkv=hkv)
    parts = [
        partial_attention(q, k[:, :, a:b], v[:, :, a:b]) for a, b in zip(cuts, cuts[1:])
    ]
    out = merge_partials(parts)
    assert mx.all(mx.isfinite(out)).item()
    assert mx.allclose(out, reference_attention(q, k, v), rtol=1e-4, atol=1e-5).item()


def test_all_empty_statistics_are_neutral():
    q, k, v = fixture_arrays(hq=4, hkv=1, length=0)
    part = partial_attention(q, k, v)
    out = merge_partials([part, part])
    assert mx.all(mx.isfinite(out)).item()
    assert mx.all(out == 0).item()


@pytest.mark.parametrize("winner", [0, 1])
def test_concentrated_attention(winner):
    q = mx.ones((1, 1, 1, 4)) * 100
    k = mx.concatenate(
        [mx.ones((1, 1, 3, 4)) * (100 if winner == i else -100) for i in range(2)],
        axis=2,
    )
    v = mx.concatenate([mx.ones((1, 1, 3, 4)) * i for i in range(2)], axis=2)
    out = merge_partials(
        [
            partial_attention(q, k[:, :, :3], v[:, :, :3]),
            partial_attention(q, k[:, :, 3:], v[:, :, 3:]),
        ]
    )
    assert mx.all(out == winner).item()
    assert mx.allclose(out, reference_attention(q, k, v)).item()
