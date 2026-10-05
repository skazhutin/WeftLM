import mlx.core as mx
import pytest

from weftlm.baselines import optimized_attention
from weftlm.fixtures import AttentionConfig, make_inputs
from weftlm.reference import reference_attention


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("hq,hkv", [(2, 2), (4, 1), (32, 8)])
def test_optimized_control_matches_independent_reference(dtype, hq, hkv):
    arrays = make_inputs(
        AttentionConfig(length=17, hq=hq, hkv=hkv, dim=128, dtype=dtype)
    )
    actual = optimized_attention(*arrays)
    expected = reference_attention(*arrays)
    tol = (
        dict(rtol=1e-4, atol=1e-5) if dtype == "float32" else dict(rtol=1e-2, atol=1e-3)
    )
    assert actual.dtype == arrays[0].dtype
    assert mx.allclose(actual, expected, **tol).item()
