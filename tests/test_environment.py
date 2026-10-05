"""Check that the installed MLX backend can evaluate a small computation."""

import mlx.core as mx


def test_mlx_cpu_matmul() -> None:
    with mx.stream(mx.cpu):
        left = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        right = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
        result = mx.matmul(left, right)
        mx.eval(result)

    assert result.tolist() == [[19.0, 22.0], [43.0, 50.0]]
