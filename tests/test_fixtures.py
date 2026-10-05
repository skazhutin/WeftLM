from dataclasses import replace

import mlx.core as mx
import pytest

from weftlm.fixtures import AttentionConfig, make_inputs


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_token_and_head_partitions_reconstruct_identical_inputs(dtype):
    config = AttentionConfig(length=4101, hq=4, hkv=2, dim=4, dtype=dtype)
    q, k, v = make_inputs(config)
    a = make_inputs(config, token_range=(0, 4093))
    b = make_inputs(config, token_range=(4093, 4101))
    assert mx.array_equal(a[0], q).item()
    for full, left, right in zip((k, v), a[1:], b[1:]):
        assert mx.array_equal(full, mx.concatenate([left, right], axis=2)).item()
    a = make_inputs(config, head_range=(0, 1))
    b = make_inputs(config, head_range=(1, 2))
    for full, left, right in zip((q, k, v), a, b):
        assert mx.array_equal(full, mx.concatenate([left, right], axis=1)).item()


def test_seed_changes_data_but_kernel_block_size_does_not():
    config = AttentionConfig(length=7, hq=2, hkv=1, dim=4)
    a = make_inputs(config)
    b = make_inputs(replace(config, block_size=1))
    assert all(mx.array_equal(x, y).item() for x, y in zip(a, b))
    assert not mx.array_equal(a[0], make_inputs(replace(config, seed=1))[0]).item()


def test_empty_local_token_range():
    config = AttentionConfig(length=7, hq=2, hkv=1, dim=4)
    q, k, v = make_inputs(config, token_range=(3, 3))
    assert q.shape == (1, 2, 1, 4)
    assert k.shape == v.shape == (1, 1, 0, 4)


@pytest.mark.parametrize(
    "kwargs", [{"length": 0}, {"hq": 3, "hkv": 2}, {"dtype": "int32"}, {"dim": -1}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AttentionConfig(**kwargs)
