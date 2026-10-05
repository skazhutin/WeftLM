import mlx.core as mx
import pytest

from weftlm.timing import measure


def test_every_repeat_reexecutes_and_evaluates_operation():
    calls = []
    barriers = []

    def operation():
        calls.append(len(calls) + 1)
        return mx.array(calls[-1]) * 2

    result = measure(
        operation,
        warmup=3,
        repeats=5,
        device=mx.cpu,
        before_each=lambda: barriers.append(len(calls)),
    )
    assert calls == list(range(1, 9))
    assert len(barriers) == 8
    assert len(result.samples_ns) == 5
    assert all(isinstance(value, int) and value > 0 for value in result.samples_ns)
    assert result.last_output.item() == 16


@pytest.mark.parametrize("kwargs", [{"warmup": -1}, {"repeats": 0}, {"repeats": 1.5}])
def test_invalid_protocol(kwargs):
    with pytest.raises(ValueError):
        measure(lambda: mx.array(1), device=mx.cpu, **kwargs)
