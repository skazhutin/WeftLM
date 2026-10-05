import math

import mlx.core as mx
import pytest

from weftlm.reference import reference_attention


def scalar_reference(q, k, v):
    """Tiny Python float64 oracle, independent of all MLX attention code."""
    q, k, v = q.tolist()[0], k.tolist()[0], v.tolist()[0]
    dim = len(q[0][0])
    groups = len(q) // len(k)
    result = []
    for head, query in enumerate(q):
        keys, values = k[head // groups], v[head // groups]
        scores = [
            math.fsum(a * b for a, b in zip(query[0], key)) / math.sqrt(dim)
            for key in keys
        ]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = math.fsum(weights)
        result.append(
            [
                [
                    math.fsum(w * value[d] for w, value in zip(weights, values))
                    / denominator
                    for d in range(dim)
                ]
            ]
        )
    return mx.array([result], dtype=mx.float32)


def test_one_token_returns_value():
    q = mx.array([[[[2.0, -1.0]]]])
    k = mx.array([[[[100.0, 5.0]]]])
    v = mx.array([[[[7.0, -3.0]]]])
    assert reference_attention(q, k, v).tolist() == v.tolist()


def test_zero_query_returns_mean():
    q = mx.zeros((1, 1, 1, 2))
    k = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    v = mx.array([[[[2.0, 4.0], [6.0, 8.0]]]])
    assert reference_attention(q, k, v).tolist() == [[[[4.0, 6.0]]]]


@pytest.mark.parametrize("hq,hkv", [(1, 1), (4, 2), (4, 1)])
def test_matches_independent_scalar_oracle(hq, hkv):
    q = mx.random.normal((1, hq, 1, 4), key=mx.random.key(11))
    k = mx.random.normal((1, hkv, 7, 4), key=mx.random.key(12))
    v = mx.random.normal((1, hkv, 7, 4), key=mx.random.key(13))
    assert mx.allclose(
        reference_attention(q, k, v), scalar_reference(q, k, v), rtol=1e-4, atol=1e-5
    ).item()


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 2, 4), (1, 1, 3, 4), (1, 1, 3, 4)),
        ((2, 2, 1, 4), (2, 1, 3, 4), (2, 1, 3, 4)),
        ((1, 2, 2, 4), (1, 1, 3, 4), (1, 1, 3, 4)),
        ((1, 3, 1, 4), (1, 2, 3, 4), (1, 2, 3, 4)),
        ((1, 2, 1, 4), (1, 1, 3, 4), (1, 1, 2, 4)),
        ((1, 2, 1, 4), (1, 1, 3, 8), (1, 1, 3, 8)),
        ((1, 2, 1, 4), (1, 1, 0, 4), (1, 1, 0, 4)),
    ],
)
def test_invalid_shapes(shapes):
    with pytest.raises(ValueError):
        reference_attention(*(mx.zeros(shape) for shape in shapes))


def test_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="float32"):
        reference_attention(
            *(mx.zeros((1, 1, 1, 2), dtype=mx.float16) for _ in range(3))
        )


def test_rejects_nonfinite_scale():
    arrays = [mx.zeros((1, 1, 1, 2)) for _ in range(3)]
    with pytest.raises(ValueError, match="finite"):
        reference_attention(*arrays, scale=float("nan"))
