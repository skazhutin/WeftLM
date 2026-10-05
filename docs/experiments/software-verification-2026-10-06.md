# Prototype software checkpoint — 2026-10-06

Roadmap steps 1–14 are implemented and verified. The computational source at
[`73b5735`](https://github.com/skazhutin/WeftLM/commit/73b5735d7aa83496d0b610fd5ba56e00aae43f47)
passed [macOS ARM64 CPU CI](https://github.com/skazhutin/WeftLM/actions/runs/37379769981).
Each preceding step was separately committed, pushed and checked in CI before
the next implementation task began. See [the roadmap](../roadmap.md) for SHAs.

## Verified software

- `uv sync --locked`, Ruff lint/format and 91 CPU tests pass.
- Wheel and sdist build; the built wheel imports the package and distributed
  benchmark module from an isolated Python 3.12 environment.
- Two native local ring participants pass collective and reference checks,
  including FP16/FP32, empty/unequal context slices and complete GQA head groups.
- Mismatched protocols fail on both participants before shape-dependent exchange.
- Tests bound launcher execution, validate raw artifact reductions, reject
  damaged series, check reference identity and retain allocation-failure records.
- The supplied specification is copied byte-for-byte; environment, caches and
  build outputs are excluded from tracked files.

## Physical local GPU checks

On this MacBookPro18,1 (Apple M1 Pro, 16 GB, macOS 27.0), the GPU small-reference
checks passed for both context and heads with N=17, Hq=32, Hkv=8, D=128, FP16.
The two participants each returned the complete output and passed the stated
FP16 tolerance against the independent reference.

Both GPU benchmark modes also passed at N=8193, Hq=4, Hkv=2, D=8, FP16,
block size 4096, one warmup and three measured repeats. Validation used a
reference sidecar from a separately verified single operation. Both rank
summaries reported successful correctness and per-iteration maximum timing.
These smoke checks tested the source subsequently committed as `73b5735`.
Their temporary artifacts were not retained as performance evidence.

Equivalent reproduction commands, with fresh output directories:

```sh
uv run --locked python -m weftlm bench --device gpu --mode single \
  --lengths 8193 --hq 4 --hkv 2 --dim 8 --warmup 1 --repeats 3 \
  --output results/gpu-smoke-reference
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm bench --device gpu --mode context --lengths 8193 \
  --hq 4 --hkv 2 --dim 8 --warmup 1 --repeats 3 \
  --reference results/gpu-smoke-reference --topology local-processes \
  --link-description "loopback TCP" --output results/gpu-smoke-context
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm bench --device gpu --mode heads --lengths 8193 \
  --hq 4 --hkv 2 --dim 8 --warmup 1 --repeats 3 \
  --reference results/gpu-smoke-reference --topology local-processes \
  --link-description "loopback TCP" --output results/gpu-smoke-heads
```

## Remaining experiment

These checks used two processes sharing one physical Mac. They verify the
software path, and do not establish two-Mac latency, memory capacity or model
throughput. The [preliminary single-Mac campaign](local-2026-10-06.md) has
retained raw measurements, but its system swap limits performance conclusions.

Step 15 is waiting for preparation and access to the second physical Mac.
Then verify hardware/transport, run physical correctness probes and perform
the three-mode campaign with each Mac's standalone control (steps 16–18).
The [distributed instructions](../distributed.md) contain the preparation
contract and native launcher commands. Model, Exo and optimization work remains
conditional on those measurements. No model weights have been downloaded.
