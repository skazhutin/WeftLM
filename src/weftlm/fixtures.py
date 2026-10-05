"""Canonical CPU-generated fixtures independent of token/head partitioning."""

from dataclasses import dataclass
from hashlib import blake2s

import mlx.core as mx

FIXTURE_BLOCK_SIZE = 4096


@dataclass(frozen=True)
class AttentionConfig:
    length: int = 4096
    hq: int = 32
    hkv: int = 8
    dim: int = 128
    dtype: str = "float16"
    seed: int = 0
    block_size: int = 4096

    def __post_init__(self):
        for name in ("length", "hq", "hkv", "dim", "block_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hq % self.hkv:
            raise ValueError("Hq must be divisible by Hkv")
        if self.dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")

    @property
    def mlx_dtype(self) -> mx.Dtype:
        return getattr(mx, self.dtype)


def _key(config: AttentionConfig, kind: str, head: int, block: int) -> mx.array:
    identity = f"weftlm-fixture-v1:{config.seed}:{kind}:{head}:{block}:{config.dim}"
    seed = int.from_bytes(blake2s(identity.encode(), digest_size=4).digest(), "little")
    return mx.random.key(seed)


def make_inputs(
    config: AttentionConfig,
    *,
    token_range: tuple[int, int] | None = None,
    head_range: tuple[int, int] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Generate only owned KV, with identical global samples in every mode.

    Head ranges select KV heads and their consecutive query-head groups.
    Token ranges may be empty. CPU RNG avoids GPU-specific normal sampling.
    """
    start, end = token_range if token_range is not None else (0, config.length)
    first, last = head_range if head_range is not None else (0, config.hkv)
    if not 0 <= start <= end <= config.length:
        raise ValueError("Token range is outside the global context")
    if not 0 <= first < last <= config.hkv:
        raise ValueError("Head range must select at least one valid KV head")
    groups = config.hq // config.hkv
    with mx.stream(mx.cpu):
        queries = [
            mx.random.normal((1, config.dim), key=_key(config, "q", head, 0))
            for head in range(first * groups, last * groups)
        ]
        q = mx.stack(queries).reshape(1, (last - first) * groups, 1, config.dim)

        def cache(kind: str) -> mx.array:
            heads = []
            for head in range(first, last):
                blocks = []
                for index in range(
                    start // FIXTURE_BLOCK_SIZE,
                    (end + FIXTURE_BLOCK_SIZE - 1) // FIXTURE_BLOCK_SIZE,
                ):
                    block = mx.random.normal(
                        (FIXTURE_BLOCK_SIZE, config.dim),
                        key=_key(config, kind, head, index),
                    )
                    low = max(start - index * FIXTURE_BLOCK_SIZE, 0)
                    high = min(end - index * FIXTURE_BLOCK_SIZE, FIXTURE_BLOCK_SIZE)
                    if high > low:
                        piece = block[low:high].astype(config.mlx_dtype)
                        mx.eval(piece)
                        blocks.append(piece)
                heads.append(
                    mx.concatenate(blocks)
                    if blocks
                    else mx.zeros((0, config.dim), dtype=config.mlx_dtype)
                )
            return mx.stack(heads)[None]

        k, v = cache("k"), cache("v")
        q = q.astype(config.mlx_dtype)
        mx.eval(q, k, v)
    return q, k, v
