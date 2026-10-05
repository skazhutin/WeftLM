# WeftLM

Experimental context-parallel inference for long-context LLMs on Apple Silicon.

## Status

The independent reference, stable local KV merging, GQA, FP16/FP32 and bounded
KV blocks are implemented. The research CLI supports local measurements and
two-participant context/head partitioning through MLX ring/TCP. CPU integration
tests and GPU checks use two local processes. Physical two-Mac measurements
are pending. The project will
measure whether context parallelism offers useful latency or memory benefits across
multiple Macs.

See the [software verification checkpoint](docs/experiments/software-verification-2026-10-06.md)
for completed checks and the physical-hardware boundary.

## Goals

- Context-parallel attention across Apple Silicon devices
- Sharded KV-cache experiments
- MLX/JACCL distributed benchmarks
- Comparison with existing tensor/pipeline-parallel approaches
- Potential integration with EXO

## Requirements

- A Mac with Apple Silicon and macOS 14 or later
- Python 3.12 for the development environment
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

The runtime dependency is pinned to MLX 0.32.3. Resolved development dependencies
are recorded in `uv.lock`.

## Setup

```sh
git clone https://github.com/skazhutin/WeftLM.git
cd WeftLM
uv sync --locked
uv run --locked python -c "import weftlm; from importlib.metadata import version; print(version('weftlm'))"
```

`uv` uses `.python-version` to select Python 3.12 and creates the local `.venv`.
The package is installed in editable mode for development.

## Development checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv build
```

The environment test evaluates a small MLX matrix multiplication on CPU and
checks its result. The suite also covers attention correctness, partitioned
fixtures, fresh evaluation, raw artifacts and two local CPU processes.

GitHub Actions runs these checks on macOS ARM64 with Python 3.12 for pushes to
`main` and pull requests. CI uses the CPU backend for the environment test and
also verifies that the built wheel imports in an isolated environment.

To check the Metal GPU backend on a physical Mac:

```sh
uv run --locked python - <<'PY'
import mlx.core as mx

with mx.stream(mx.gpu):
    left = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    right = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
    result = mx.matmul(left, right)
    mx.eval(result)
assert result.tolist() == [[19.0, 22.0], [43.0, 50.0]]
print("MLX GPU check passed")
PY
```

`uv build` produces a wheel and source distribution in `dist/`. The local
environment, caches, and build outputs are excluded from Git.

## Project layout

```text
src/weftlm/           Experimental attention and benchmark modules
tests/               Numerical, CLI and local distributed integration checks
docs/project_spec.md  Original project specification in Russian
```

The [project specification](docs/project_spec.md) describes the experiment,
comparison contract, and measurement protocol. The roadmap distinguishes
verified implementation from pending physical experiments and conditional work.

The [implementation roadmap](docs/roadmap.md) tracks individual tasks and verified
commits. The small reference is available as
`weftlm.reference.reference_attention`; it supports one query token and GQA
without copying KV heads, and is intended for correctness rather than timing.

## Research roadmap

### Local commands

```sh
uv run --locked python -m weftlm check --device gpu
uv run --locked python -m weftlm bench --mode single --output results/single-run
uv run --locked python -m weftlm bench --mode local-context --output results/local-context-run
uv run --locked python -m weftlm summarize --input results/single-run
```

Use `--device cpu` for small correctness/CLI checks, and `--help` for all options.
Benchmarks default to 10 warmups and 50 measurements at 4k, 16k, 64k, 128k and
256k tokens. Inputs are generated outside timing. Each result directory contains
raw `samples.jsonl`, `metadata.json` and `summary.csv` (nearest-rank p95).
Verified `single` runs also save a small independent `reference.json` output.
Output directories must be new. `local-context` is a blocked calculation on one
Mac; it is not a measurement of two-node context parallelism.

The [first local campaign](docs/experiments/local-2026-10-06.md) includes raw
measurements and explicitly records system swap/memory-pressure limitations.

### Distributed commands

```sh
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm check-distributed --device cpu --mode context
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm check-distributed --device cpu --mode heads
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm bench --mode context --lengths 4096 --topology local-processes \
  --link-description "loopback TCP" --output results/context-local-debug
```

Use `--mode heads` for the optimized head-split control. Each rank writes its
own directory; total latency is reduced by maximum across participants for
each iteration before computing median/p95. Long lengths require `--reference`
from a verified single run. See [distributed instructions](docs/distributed.md)
for artifacts, diagnostics, failure handling and physical-node preparation.

### Planned physical experiments

1. **Prepare two Macs.** Record hardware and connection, make both nodes reachable,
   install the same checkout/lock and verify the transport and numerical results.
2. **Decode attention on two physical Macs.** Keep each node's KV shard local and
   exchange partial results through MLX Distributed. Verify the available
   transport separately; two processes on one Mac are a debugging tool.
3. **Compare three modes.** Measure optimized attention on each Mac separately,
   splitting by heads on two Macs, and splitting by context on two Macs. Include
   communication and merging in total latency, make the complete output
   available on each participant, and save raw timings, error, and memory data.

These are synthetic decode experiments for one attention layer, with one query
token. They do not establish whole-model generation speed or supported model
context length. Prefill, a real model, and possible Exo integration follow the
initial measurements. A server, management UI, device discovery, and a custom
network protocol are outside the initial scope.

## License

Apache-2.0. See [LICENSE](LICENSE).
