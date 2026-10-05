# WeftLM

Experimental context-parallel inference for long-context LLMs on Apple Silicon.

## Status

Initial research scaffold. The Python package and development environment are ready;
attention algorithms and distributed benchmarks are planned next. The project will
measure whether context parallelism offers useful latency or memory benefits across
multiple Macs.

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
checks its result. It verifies the dependency setup; it does not test attention
correctness or distributed performance yet.

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
src/weftlm/           Python package; no computational API yet
tests/               Environment checks and future correctness tests
docs/project_spec.md  Original project specification in Russian
```

The [project specification](docs/project_spec.md) describes the experiment,
comparison contract, and measurement protocol. Its implementation tasks are
planned work, not features already present in this scaffold.

## Research roadmap

1. **Correctness on one Mac.** Implement an independent full-attention reference,
   local shard statistics, and stable merging of KV shards. Cover grouped-query
   attention without duplicating KV, unequal or empty shards, concentrated
   attention, and FP32 before a lower-precision working format.
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
