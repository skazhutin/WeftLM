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
