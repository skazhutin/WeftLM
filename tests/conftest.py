"""Keep automated correctness tests independent of a Metal GPU."""

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def cpu_stream():
    with mx.stream(mx.cpu):
        yield
